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

    HEADLINE_PROMPT = """Based on the following article, create a SINGLE COMPELLING HEADLINE.
The headline MUST be non-empty (at least 1 character) and should be news-style, informative, and engaging.
If you cannot create a headline from the article content, respond with a brief descriptive phrase.

ARTICLE:
---
{article_text}
---

HEADLINE (single line only, no quotation marks, MUST be non-empty):"""

    # Prompt for summarizing individual chunks (first stage of chunked summarization)
    CHUNK_SUMMARY_PROMPT = """Summarize the following article excerpt in 2-3 sentences.
Focus ONLY on the key information in this excerpt.
Do NOT add information not present in the excerpt.

EXCERPT:
---
{excerpt_text}
---

SUMMARY:"""

    # Prompt for combining chunk summaries (second stage)
    COMBINED_SUMMARY_PROMPT = """Based on the following partial summaries of an article, create the final summaries.
Each partial summary covers a different section of the article.
Combine ALL information from all partial summaries.
IMPORTANT: The HEADLINE MUST be non-empty (at least 1 character).

PARTIAL SUMMARIES:
---
{partial_summaries}
---

Now create the final summaries:

1. SHORT SUMMARY (1-2 lines capturing the essence):
[Your short summary here]

2. MEDIUM SUMMARY (3-5 lines with more detail):
[Your medium summary here]

3. HEADLINE (single compelling headline, no quotation marks, MUST be non-empty):
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
        
        response = await self.llm_service.complete(
            prompt=self.CHUNK_SUMMARY_PROMPT.format(excerpt_text=chunk),
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
        
        # Final fallback for empty headline - MUST never return empty string
        if not headline or not headline.strip():
            # Use first few words of short_summary as fallback headline
            if short_summary and short_summary.strip():
                words = short_summary.split()[:8]
                headline = ' '.join(words) + ('...' if len(short_summary.split()) > 8 else '')
            else:
                headline = "News Summary"
        
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
        
        # Fallback for empty headline - MUST never return empty string
        if not headline or not headline.strip():
            # Use first few words of short_summary as fallback headline
            if short_summary and short_summary.strip():
                words = short_summary.split()[:8]
                headline = ' '.join(words) + ('...' if len(short_summary.split()) > 8 else '')
            else:
                headline = "News Summary"
        
        return short_summary, medium_summary, headline
    
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
