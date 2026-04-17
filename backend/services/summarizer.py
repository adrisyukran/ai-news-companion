"""
AI News Companion - Summarizer Service

Provides article summarization with three summary types:
- Short summary (1-2 lines)
- Medium summary (3-5 lines)
- Headline (single compelling headline)

Features:
- Long article handling via chunking (summarize chunks, then summarize summaries)
- Hallucination prevention through strict prompt engineering
- Multi-format support via ParserService (URL, PDF, DOCX, TXT)
"""
import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass

from backend.services.llm_service import NanoGPTService
from backend.services.parser_service import ParserService, ParserError
from backend.config import CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS

logger = logging.getLogger(__name__)


@dataclass
class ChunkSummary:
    """Container for a chunk summary result."""
    chunk_index: int
    summary: str


class SummarizerError(Exception):
    """Exception raised when summarization fails."""
    pass


class SummarizerService:
    """
    Service for generating multiple types of summaries from articles.
    
    Supports three summary formats:
    1. Short summary (1-2 lines): Quick overview
    2. Medium summary (3-5 lines): More detailed coverage
    3. Headline: Single compelling headline
    
    For long articles, uses a two-stage summarization approach:
    1. Split article into chunks
    2. Summarize each chunk individually
    3. Combine chunk summaries and generate final summaries
    """
    
    # System prompt for summarization (prevents hallucinations)
    SYSTEM_PROMPT = """You are a precise news summarizer. Your task is to summarize articles STRICTLY based on the provided content.

IMPORTANT RULES:
1. ONLY use information explicitly stated in the article
2. NEVER add information, opinions, or details not present in the original text
3. NEVER guess or infer information not supported by the article
4. Use neutral, factual language typical of news reporting
5. Focus on key facts: who, what, when, where, why, and how
6. IGNORE advertisements, sponsored content, promotions, navigation menus, sidebar content, newsletter signup prompts, social media links, and other out-of-context elements
7. Focus ONLY on the main article content

If the article does not contain enough information for a particular summary type, clearly state what information is missing rather than inventing details."""
    
    # Prompt templates for each summary type
    SHORT_SUMMARY_PROMPT = """Based on the following article, provide a SHORT SUMMARY of 1-2 lines.
The short summary should capture the ESSENCE of the article in brief form.

ARTICLE:
---
{article_text}
---

SHORT SUMMARY (1-2 lines only):"""

    MEDIUM_SUMMARY_PROMPT = """Based on the following article, provide a MEDIUM SUMMARY of 3-5 lines.
The medium summary should provide a more detailed overview while remaining concise.

ARTICLE:
---
{article_text}
---

MEDIUM SUMMARY (3-5 lines):"""

    HEADLINE_PROMPT = """You are a professional news editor with 20+ years of experience at major news organizations. Your task is to create a compelling, factual headline for the article below.

CRITICAL OUTPUT RULES - FOLLOW EXACTLY:
- Output ONLY the headline text - absolutely NOTHING else
- Do NOT include quotation marks, colons, labels, brackets, or any prefix/suffix
- Do NOT explain your reasoning or add any commentary
- Do NOT output phrases like "Headline:", "News:", "Title:", or "Here is the headline:"
- Your entire response must be ONLY the headline itself

STRICT HEADLINE REQUIREMENTS:
- Must be a proper, professional news headline (6-14 words)
- Must start with a capital letter
- Must use active voice when possible (e.g., "Company Launches Product" not "Product Launched by Company")
- Must be specific and informative - include key entities (names, places, organizations) when available
- Must NOT be a question
- Must NOT use generic phrases like "Article about...", "Story on...", "News regarding..."
- Must NOT use clickbait language like "You Won't Believe...", "Shocking...", "Amazing..."
- NO quotation marks, colons, semicolons, or trailing punctuation (periods, exclamation marks)
- CRITICAL: Must be a COMPLETE thought - NEVER end with conjunctions, prepositions, or articles
- NEVER end with words like: "a", "an", "the", "because", "and", "with", "of", "for", "to", "in", "on", "at", "by", "from"
- The headline must be a standalone, complete statement that makes sense on its own

GENERIC WORDS TO AVOID - these make headlines meaningless:
- "Something", "something new", "something big"
- "Latest", "recent", "new" (unless paired with specific subject)
- Standalone "News", "Story", "Article", "Update", "Report"
- "Discusses", "talks about", "looks at" (weak verbs)

STRONG HEADLINE VERBS TO USE:
- Launches, Unveils, Announces, Reveals, Introduces
- Expands, Acquires, Partners, Invests, Commits
- Discovers, Develops, Creates, Achieves, Breaks
- Faces, Confronts, Addresses, Tackles, Overcomes
- Rises, Falls, Surges, Drops, Rebounds (for markets/data)

GOOD HEADLINE EXAMPLES (study these patterns):
- "Tech Giant Unveils Revolutionary AI Assistant for Healthcare"
- "Global Markets Rally as Inflation Shows Signs of Cooling"
- "Scientists Discover Breakthrough Treatment for Rare Disease"
- "Electric Vehicle Sales Surge 40 Percent in Third Quarter"
- "Major Automaker Announces All-Electric Lineup by 2030"
- "Researchers Develop New Battery Technology for Grid Storage"
- "Federal Reserve Holds Interest Rates Steady Amid Economic Uncertainty"
- "Startup Raises $500 Million Series F for AI-Powered Drug Discovery"
- "Climate Summit Reaches Historic Agreement on Emissions Reduction"
- "Tech Company Faces Antitrust Lawsuit Over Market Dominance"

BAD HEADLINE EXAMPLES (NEVER produce these):
- "News:" or "News Summary" (generic label)
- "What is AI?" (question format)
- "A summary of the article" (vague, meaningless)
- "Headline: Breaking News" (includes label)
- "The company announced something new today" (too vague)
- "Latest developments in the story" (no substance)
- "Article discusses important topic" (generic phrase)
- "Something big is happening" (meaningless clickbait)
- "You won't believe this news" (clickbait)
- "Breaking: Something amazing just happened" (vague clickbait)
- "Company announces new product because" (incomplete - ends with conjunction)
- "Market rises as investors look at" (incomplete - ends with preposition)
- "Scientists discover a" (incomplete - ends with article)

ARTICLE:
---
{article_text}
---

Remember: You are a professional editor. Create ONE compelling, specific headline that is a COMPLETE thought. Output ONLY the headline text with no additional content."""

    # Prompt for summarizing individual chunks (first stage of chunked summarization)
    CHUNK_SUMMARY_PROMPT = """Summarize the following article excerpt in 2-3 sentences.
Focus ONLY on the key information in this excerpt.
Do NOT add information not present in the excerpt.
IGNORE any advertisements, sponsored content, or navigation elements.

EXCERPT:
---
{excerpt_text}
---

SUMMARY:"""

    # Prompt for combining chunk summaries (second stage)
    COMBINED_SUMMARY_PROMPT = """Based on the following partial summaries of an article, create the final summaries.
Each partial summary covers a different section of the article.
Combine ALL information from all partial summaries.
IMPORTANT: The HEADLINE MUST follow strict requirements below.

PARTIAL SUMMARIES:
---
{partial_summaries}
---

Now create the final summaries:

1. SHORT SUMMARY (1-2 lines capturing the essence):
[Your short summary here]

2. MEDIUM SUMMARY (3-5 lines with more detail):
[Your medium summary here]

3. HEADLINE - STRICT REQUIREMENTS:
   - Must be a proper news headline (NOT a question, NOT a phrase)
   - Must be 8-12 words maximum
   - Must start with a capital letter
   - Must use active voice
   - NO quotation marks or colons
   [Your headline here]"""

    def __init__(
        self,
        llm_service: Optional[NanoGPTService] = None,
        parser_service: Optional[ParserService] = None,
        chunk_size: int = CHUNK_SIZE_TOKENS,
        chunk_overlap: int = CHUNK_OVERLAP_TOKENS,
    ):
        """
        Initialize the SummarizerService.
        
        Args:
            llm_service: NanoGPTService instance (creates new if not provided)
            parser_service: ParserService instance (creates new if not provided)
            chunk_size: Maximum tokens per chunk for long articles
            chunk_overlap: Token overlap between chunks
        """
        self.llm_service = llm_service or NanoGPTService()
        self.parser_service = parser_service or ParserService()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        # Simple estimation: ~4 characters per token for English
        return len(text) // 4
    
    def _preprocess_article_text(self, text: str) -> str:
        """
        Preprocess article text by removing advertisements and cleaning up formatting.
        
        Uses regular expressions to remove common ad patterns and excessive whitespace.
        
        Args:
            text: Raw article text
            
        Returns:
            Cleaned article text with ads removed
        """
        import re
        
        # Remove common advertisement patterns
        ad_patterns = [
            # Advertisement labels
            r'(?i)^\s*advertisement\s*$',
            r'(?i)^\s*ad\s*$',
            r'(?i)^\s*sponsored\s*$',
            r'(?i)^\s*sponsored content\s*$',
            r'(?i)^\s*partner content\s*$',
            r'(?i)^\s*promoted\s*$',
            r'(?i)^\s*promotion\s*$',
            
            # Newsletter/signup prompts
            r'(?i)^\s*subscribe to our newsletter\s*$',
            r'(?i)^\s*sign up for our newsletter\s*$',
            r'(?i)^\s*get the latest news\s*$',
            r'(?i)^\s*join our mailing list\s*$',
            r'(?i)^\s*email newsletter\s*$',
            r'(?i)^\s*subscribe now\s*$',
            r'(?i)^\s*stay informed\s*$',
            
            # Social media/navigation elements
            r'(?i)^\s*share this article\s*$',
            r'(?i)^\s*share on (twitter|facebook|linkedin|reddit)\s*$',
            r'(?i)^\s*follow us on\s*$',
            r'(?i)^\s*related articles?\s*$',
            r'(?i)^\s*read more\s*$',
            r'(?i)^\s*continue reading\s*$',
            
            # Generic ad-like patterns
            r'(?i)^\s*\[ad(vertisement)?\]\s*$',
            r'(?i)^\s*from our sponsors\s*$',
            r'(?i)^\s*special offer\s*$',
            r'(?i)^\s*limited time offer\s*$',
        ]
        
        cleaned_text = text
        
        # Remove each ad pattern
        for pattern in ad_patterns:
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.MULTILINE)
        
        # Remove lines that are primarily promotional (contain certain keywords)
        promotional_keywords = [
            r'(?i)^.*buy now.*$',
            r'(?i)^.*click here to.*$',
            r'(?i)^.*learn more about.*$',
            r'(?i)^.*get started today.*$',
            r'(?i)^.*limited time.*$',
            r'(?i)^.*act now.*$',
            r'(?i)^.*don\'t miss.*$',
            r'(?i)^.*exclusive offer.*$',
        ]
        
        for pattern in promotional_keywords:
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.MULTILINE)
        
        # Clean up excessive blank lines (replace 3+ consecutive newlines with 2)
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
        
        # Clean up lines with only whitespace
        cleaned_text = re.sub(r'^\s+$', '', cleaned_text, flags=re.MULTILINE)
        
        # Strip leading/trailing whitespace from the entire text
        cleaned_text = cleaned_text.strip()
        
        return cleaned_text
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks with overlap for processing.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        estimated_tokens = self._estimate_tokens(text)
        
        # If text fits in single chunk, return as-is
        if estimated_tokens <= self.chunk_size:
            return [text]
        
        chunks = []
        chars_per_token = 4
        chunk_size_chars = self.chunk_size * chars_per_token
        overlap_chars = self.chunk_overlap * chars_per_token
        
        start = 0
        while start < len(text):
            end = start + chunk_size_chars
            chunk = text[start:end]
            
            # Try to break at sentence boundary for cleaner chunks
            if end < len(text):
                search_start = int(len(chunk) * 0.8)
                for ending in ['. ', '.\n', '! ', '!\n', '? ', '?\n']:
                    last_ending = chunk.rfind(ending, search_start)
                    if last_ending != -1:
                        chunk = chunk[:last_ending + len(ending)]
                        break
            
            chunks.append(chunk.strip())
            start = end - overlap_chars if end < len(text) else len(text)
        
        return chunks
    
    async def _summarize_chunk(self, chunk: str, chunk_index: int) -> ChunkSummary:
        """
        Generate a summary for a single chunk.
        
        Args:
            chunk: Text chunk to summarize
            chunk_index: Index of the chunk for logging
            
        Returns:
            ChunkSummary with the summary text
        """
        logger.info(f"Processing chunk {chunk_index + 1}")
        
        # Preprocess chunk to remove any ad patterns
        cleaned_chunk = self._preprocess_article_text(chunk)
        
        response = await self.llm_service.complete(
            prompt=self.CHUNK_SUMMARY_PROMPT.format(excerpt_text=cleaned_chunk),
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.3,  # Lower temperature for more factual summaries
            max_tokens=200,
        )
        
        return ChunkSummary(
            chunk_index=chunk_index,
            summary=response.content.strip()
        )
    
    async def _generate_final_summaries(
        self,
        partial_summaries: List[ChunkSummary]
    ) -> Tuple[str, str, str]:
        """
        Generate final summaries from partial chunk summaries.
        
        Args:
            partial_summaries: List of chunk summaries
            
        Returns:
            Tuple of (short_summary, medium_summary, headline)
        """
        # Combine all partial summaries
        combined_text = "\n\n".join([
            f"[Chunk {cs.chunk_index + 1}]: {cs.summary}"
            for cs in partial_summaries
        ])
        
        logger.info(f"Generating final summaries from {len(partial_summaries)} partial summaries")
        
        response = await self.llm_service.complete(
            prompt=self.COMBINED_SUMMARY_PROMPT.format(partial_summaries=combined_text),
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.5,
            max_tokens=500,
        )
        
        # Parse the response to extract the three summary types
        return self._parse_combined_response(response.content)
    
    def _parse_combined_response(self, response_text: str) -> Tuple[str, str, str]:
        """
        Parse the combined summary response to extract individual summaries.
        
        Args:
            response_text: Raw response text from LLM
            
        Returns:
            Tuple of (short_summary, medium_summary, headline)
        """
        lines = response_text.strip().split('\n')
        
        short_summary = ""
        medium_summary = ""
        headline = ""
        
        current_section = None
        current_content = []
        
        def save_section():
            nonlocal current_section, current_content
            if current_section == 'short' and current_content:
                nonlocal short_summary
                short_summary = ' '.join(current_content).strip()
            elif current_section == 'medium' and current_content:
                nonlocal medium_summary
                medium_summary = ' '.join(current_content).strip()
            elif current_section == 'headline' and current_content:
                nonlocal headline
                headline = ' '.join(current_content).strip()
        
        for line in lines:
            line = line.strip()
            
            if not line:
                continue
            
            # Skip known header patterns like [Your short summary here]
            if '[your' in line.lower():
                continue
            
            # Detect section headers
            lower_line = line.lower()
            
            # Check for section markers
            if lower_line.startswith('1. short summary') or lower_line.startswith('short summary') or 'short summary (1-2' in lower_line:
                save_section()
                current_section = 'short'
                current_content = []
            elif lower_line.startswith('2. medium summary') or lower_line.startswith('medium summary') or 'medium summary (3-5' in lower_line:
                save_section()
                current_section = 'medium'
                current_content = []
            elif lower_line.startswith('3. headline') or lower_line.startswith('headline (single') or 'headline (single' in lower_line:
                save_section()
                current_section = 'headline'
                current_content = []
            elif current_section:
                current_content.append(line)
        
        # Save the last section
        save_section()
        
        # If parsing failed, use the full response as short summary
        if not short_summary:
            short_summary = response_text.strip()[:500]
        
        if not headline:
            # Try to extract headline from the text
            headline = self._extract_headline(response_text)
        
        if not medium_summary:
            medium_summary = short_summary
        
        # Validate headline using the robust validation logic
        # This ensures consistency with direct summarization path
        headline = self._validate_headline(headline, short_summary)
        
        return short_summary, medium_summary, headline
    
    def _extract_headline(self, text: str) -> str:
        """
        Attempt to extract a headline from text.
        
        Args:
            text: Input text
            
        Returns:
            Extracted headline or first line
        """
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if lines:
            # Return the longest line as potential headline
            return max(lines, key=len)[:200]
        return "Summary"
    
    async def _generate_summaries_direct(self, article_text: str) -> Tuple[str, str, str]:
        """
        Generate all three summary types directly for a single article.
        
        Args:
            article_text: Full article text
            
        Returns:
            Tuple of (short_summary, medium_summary, headline)
        """
        logger.info("Generating summaries directly (single pass)")
        
        # Generate short summary
        short_response = await self.llm_service.complete(
            prompt=self.SHORT_SUMMARY_PROMPT.format(article_text=article_text),
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.3,
            max_tokens=150,
        )
        short_summary = short_response.content.strip()
        
        # Generate medium summary
        medium_response = await self.llm_service.complete(
            prompt=self.MEDIUM_SUMMARY_PROMPT.format(article_text=article_text),
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.3,
            max_tokens=400,
        )
        medium_summary = medium_response.content.strip()
        
        # Generate headline
        headline_response = await self.llm_service.complete(
            prompt=self.HEADLINE_PROMPT.format(article_text=article_text),
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.5,
            max_tokens=50,
        )
        headline = headline_response.content.strip()
        
        # Post-process headline validation
        headline = self._validate_headline(headline, short_summary)
        
        return short_summary, medium_summary, headline
    
    def _is_generic_headline(self, headline: str) -> bool:
        """
        Check if a headline is too generic or meaningless.
        
        Args:
            headline: Headline to check
            
        Returns:
            True if headline is generic/meaningless, False otherwise
        """
        if not headline or not headline.strip():
            return True
        
        lower = headline.lower().strip()
        
        # Generic words/phrases that make headlines meaningless
        generic_patterns = [
            # Pure labels
            r'^news[:\s]*$',
            r'^headline[:\s]*$',
            r'^title[:\s]*$',
            r'^summary[:\s]*$',
            r'^story[:\s]*$',
            r'^article[:\s]*$',
            r'^update[:\s]*$',
            r'^report[:\s]*$',
            r'^breaking news[:\s]*$',
            r'^news story[:\s]*$',
            r'^news summary[:\s]*$',
            r'^news article[:\s]*$',
            
            # Vague phrases
            r'^a?n?\s*(summary|article|story|report|update)\s*(of|on|about)?\s*the?\s*(article|story|news)?$',
            r'^article about',
            r'^story about',
            r'^news about',
            r'^report on',
            r'^update on',
            r'^something (new|big|important)',
            r'^latest (news|update|developments)',
            r'^developments? (in|on)',
            r'^discusses',
            r'^talks? about',
            r'^looks? at',
            r'^covers',
            
            # Clickbait patterns
            r'^you (won\'t|will)\s*(not)?\s*believe',
            r'^shocking',
            r'^amazing',
            r'^incredible',
            r'^(this|that)\s*is',
            
            # Question format
            r'^what\s+(is|are|was|were)',
            r'^how\s+(to|does|did|can|could)',
            r'^why\s+(is|are|was|were|does|did)',
            r'^when\s+(is|are|was|were)',
            r'^who\s+(is|are|was|were)',
            r'\?$',
        ]
        
        import re
        for pattern in generic_patterns:
            if re.search(pattern, lower):
                return True
        
        # Check if headline is too short (less than 3 words)
        words = headline.split()
        if len(words) < 3:
            return True
        
        # Check if headline contains only generic words
        generic_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                        'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                        'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                        'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                        'through', 'during', 'before', 'after', 'above', 'below',
                        'between', 'under', 'again', 'further', 'then', 'once',
                        'news', 'story', 'article', 'report', 'update', 'summary'}
        
        content_words = [w.lower().strip('.,!?;:"\'') for w in words
                        if w.lower().strip('.,!?;:"\'') not in generic_words]
        
        # If less than 2 content words, it's too generic
        if len(content_words) < 2:
            return True
        
        return False
    
    def _extract_fallback_headline(self, short_summary: str) -> str:
        """
        Extract a meaningful headline from a short summary.
        
        Args:
            short_summary: Short summary text
            
        Returns:
            A meaningful headline extracted from the summary
        """
        if not short_summary or not short_summary.strip():
            return "News Story"
        
        # Try to extract the first sentence or meaningful phrase
        summary = short_summary.strip()
        
        # Split into sentences
        import re
        sentences = re.split(r'[.!?]+', summary)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if sentences:
            # Use the first sentence as the base
            first_sentence = sentences[0].strip()
            
            # Clean up: remove leading conjunctions or articles
            first_sentence = re.sub(r'^(and|but|or|so|yet|the|a|an)\s+', '',
                                   first_sentence, flags=re.IGNORECASE)
            
            # Take first 10-12 words for headline length
            words = first_sentence.split()
            headline_words = words[:12]
            
            # Check if we're cutting off at an incomplete word and adjust
            incomplete_endings = {
                "because", "and", "but", "or", "so", "yet", "although", "though", "while", "whereas",
                "with", "of", "for", "to", "in", "on", "at", "by", "from", "into", "upon", "within",
                "without", "through", "throughout", "across", "around", "about", "after", "before",
                "during", "under", "over", "above", "below", "between", "among", "beside", "beyond",
                "a", "an", "the",
                "as", "than", "that", "which", "who", "whom", "whose", "what", "when", "where", "why",
                "how", "if", "unless", "until", "since",
            }
            
            # If the last word is incomplete, trim backwards until we find a complete ending
            while headline_words and headline_words[-1].strip().lower().rstrip('.,!?;:"\'') in incomplete_endings:
                headline_words = headline_words[:-1]
            
            # If we trimmed too much, try to include more from the original sentence
            if len(headline_words) < 4 and len(words) > 12:
                # Try a different cut point - look for a complete clause
                for i in range(min(8, len(words)), len(words)):
                    if words[i].strip().lower().rstrip('.,!?;:"\'') not in incomplete_endings:
                        headline_words = words[:i+1]
                        break
            
            headline = ' '.join(headline_words)
            
            # Capitalize first letter of each major word (title case-ish)
            # But keep articles/prepositions lowercase unless first word
            minor_words = {'a', 'an', 'the', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or', 'but', 'is', 'are', 'was', 'were'}
            title_words = []
            for i, word in enumerate(headline_words):
                if i == 0 or word.lower() not in minor_words:
                    title_words.append(word.capitalize() if len(word) > 2 else word.upper() if len(word) <= 2 else word.capitalize())
                else:
                    title_words.append(word.lower())
            headline = ' '.join(title_words)
            
            # Ensure it starts with a capital letter
            if headline and headline[0].islower():
                headline = headline[0].upper() + headline[1:]
            
            # Remove trailing punctuation
            headline = headline.rstrip('.,!?;:')
            
            # Ensure minimum length
            if len(headline.split()) < 3 and len(sentences) > 1:
                # Try combining first two sentences
                combined = ' '.join(sentences[:2])
                words = combined.split()[:12]
                
                # Again, check for incomplete endings
                while words and words[-1].strip().lower().rstrip('.,!?;:"\'') in incomplete_endings:
                    words = words[:-1]
                
                headline = ' '.join(words).rstrip('.,!?;:')
            
            return headline if headline.strip() else "News Story"
        
        # Fallback: use first 10 words of summary
        words = summary.split()[:10]
        
        # Check for incomplete endings and adjust
        incomplete_endings = {
            "because", "and", "but", "or", "so", "yet", "although", "though", "while", "whereas",
            "with", "of", "for", "to", "in", "on", "at", "by", "from", "into", "upon", "within",
            "without", "through", "throughout", "across", "around", "about", "after", "before",
            "during", "under", "over", "above", "below", "between", "among", "beside", "beyond",
            "a", "an", "the",
        }
        
        while words and words[-1].strip().lower().rstrip('.,!?;:"\'') in incomplete_endings:
            words = words[:-1]
        
        headline = ' '.join(words).rstrip('.,!?;:')
        return headline if headline.strip() else "News Story"
    
    def _ends_with_incomplete_word(self, headline: str) -> bool:
        """
        Check if a headline ends with an incomplete word (conjunction, preposition, or article).
        
        Args:
            headline: Headline to check
            
        Returns:
            True if headline ends with an incomplete word, False otherwise
        """
        if not headline or not headline.strip():
            return False
        
        # Words that indicate an incomplete thought when at the end
        incomplete_endings = {
            # Conjunctions
            "because", "and", "but", "or", "so", "yet", "although", "though", "while", "whereas",
            # Prepositions
            "with", "of", "for", "to", "in", "on", "at", "by", "from", "into", "upon", "within",
            "without", "through", "throughout", "across", "around", "about", "after", "before",
            "during", "under", "over", "above", "below", "between", "among", "beside", "beyond",
            # Articles
            "a", "an", "the",
            # Other incomplete indicators
            "as", "than", "that", "which", "who", "whom", "whose", "what", "when", "where", "why",
            "how", "if", "unless", "until", "since", "while",
        }
        
        words = headline.strip().split()
        if not words:
            return False
        
        # Get the last word, stripped of punctuation
        last_word = words[-1].strip().lower().rstrip('.,!?;:"\'')
        
        return last_word in incomplete_endings
    
    def _validate_headline(self, headline: str, short_summary: str) -> str:
        """
        Validate and clean up generated headline with robust fallback logic.
        
        Args:
            headline: Generated headline
            short_summary: Short summary as fallback source
            
        Returns:
            Validated headline
        """
        # Strip any leading/trailing whitespace
        headline = headline.strip()
        
        # Remove any残留 from LLM output (labels, prefixes that LLM might add)
        # Remove common prefixes LLM might accidentally include
        prefixes_to_remove = [
            "headline:", "headline", "headline:",
            "here's the headline:", "here is the headline:",
            "title:", "title",
            "news:", "news",
            "breaking:", "breaking news:",
        ]
        lower_headline = headline.lower()
        for prefix in prefixes_to_remove:
            if lower_headline.startswith(prefix):
                headline = headline[len(prefix):].strip()
                lower_headline = headline.lower()
        
        # Remove surrounding quotation marks
        if (headline.startswith('"') and headline.endswith('"')) or \
           (headline.startswith("'") and headline.endswith("'")):
            headline = headline[1:-1]
        
        # Remove trailing colons, periods, and other punctuation
        headline = headline.rstrip(':.,!?;')
        
        # Strip again after all cleaning
        headline = headline.strip()
        
        # Check if headline is generic/meaningless
        if self._is_generic_headline(headline):
            logger.debug(f"Headline detected as generic: '{headline}'")
            headline = ""  # Trigger fallback
        
        # Check if headline is too short (< 5 words) - use fallback
        if headline and len(headline.split()) < 5:
            logger.debug(f"Headline too short ({len(headline.split())} words): '{headline}'")
            headline = ""  # Trigger fallback
        
        # Check if headline ends with an incomplete word (conjunction, preposition, article)
        if headline and self._ends_with_incomplete_word(headline):
            logger.debug(f"Headline ends with incomplete word: '{headline}'")
            headline = ""  # Trigger fallback
        
        # Final fallback for empty, too short, or generic headline
        if not headline or not headline.strip():
            headline = self._extract_fallback_headline(short_summary)
            logger.debug(f"Using fallback headline: '{headline}'")
        
        # Final cleanup: ensure no trailing punctuation or quotes
        headline = headline.strip().rstrip('.,!?;:"\'')
        
        # Last resort fallback - MUST never return empty string
        if not headline or not headline.strip():
            headline = "News Story"
        
        return headline
    
    async def summarize(self, source: str, source_type: str = "text") -> Tuple[str, str, str]:
        """
        Summarize an article from any supported source.
        
        Args:
            source: The source (URL, file path, or text content)
            source_type: Type of source ('url', 'file_path', or 'text')
            
        Returns:
            Tuple of (short_summary, medium_summary, headline)
            
        Raises:
            SummarizerError: If summarization fails
        """
        try:
            # Extract text from the source
            article_text = self.parser_service.parse(source)
            
            if not article_text or not article_text.strip():
                raise SummarizerError("No text content extracted from the source")
            
            # Preprocess the text to remove ads and clean up formatting
            article_text = self._preprocess_article_text(article_text)
            
            if not article_text or not article_text.strip():
                raise SummarizerError("No text content remaining after preprocessing")
            
            logger.info(f"Processing article ({len(article_text)} characters)")
            
            # Check if chunking is needed
            estimated_tokens = self._estimate_tokens(article_text)
            
            if estimated_tokens <= self.chunk_size:
                # Single-pass summarization
                return await self._generate_summaries_direct(article_text)
            else:
                # Chunked summarization for long articles
                return await self._summarize_long_article(article_text)
                
        except ParserError as e:
            raise SummarizerError(f"Failed to parse source: {e}")
        except Exception as e:
            logger.error(f"Summarization error: {e}")
            raise SummarizerError(f"Summarization failed: {e}")
    
    async def _summarize_long_article(self, article_text: str) -> Tuple[str, str, str]:
        """
        Summarize a long article using chunking strategy.
        
        Strategy:
        1. Split article into chunks
        2. Summarize each chunk individually
        3. Combine chunk summaries and generate final summaries
        
        Args:
            article_text: Full article text
            
        Returns:
            Tuple of (short_summary, medium_summary, headline)
        """
        logger.info(f"Article too long ({self._estimate_tokens(article_text)} tokens), using chunking strategy")
        
        # Split into chunks
        chunks = self._chunk_text(article_text)
        logger.info(f"Split article into {len(chunks)} chunks")
        
        # Summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            summary = await self._summarize_chunk(chunk, i)
            chunk_summaries.append(summary)
        
        # Generate final summaries from chunk summaries
        return await self._generate_final_summaries(chunk_summaries)
    
    async def summarize_url(self, url: str) -> Tuple[str, str, str]:
        """
        Summarize an article from a URL.
        
        Args:
            url: URL of the article
            
        Returns:
            Tuple of (short_summary, medium_summary, headline)
        """
        return await self.summarize(url, source_type="url")
    
    async def summarize_file(self, file_path: str) -> Tuple[str, str, str]:
        """
        Summarize an article from a local file.
        
        Args:
            file_path: Path to the file (PDF, DOCX, TXT)
            
        Returns:
            Tuple of (short_summary, medium_summary, headline)
        """
        return await self.summarize(file_path, source_type="file_path")
    
    async def summarize_text(self, text: str) -> Tuple[str, str, str]:
        """
        Summarize plain text content.
        
        Args:
            text: Plain text to summarize
            
        Returns:
            Tuple of (short_summary, medium_summary, headline)
        """
        return await self.summarize(text, source_type="text")
