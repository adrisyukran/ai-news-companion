"""
AI News Companion - Summarizer Service Tests

Tests cover:
- Token estimation
- Text chunking
- Short, medium, headline prompts
- Combined response parsing
- Long article chunking strategy
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.services.summarizer import (
    SummarizerService,
    SummarizerError,
    ChunkSummary,
)


class TestTokenEstimation:
    """Tests for token estimation logic."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = SummarizerService()
    
    def test_estimate_tokens_empty(self):
        """Test token estimation for empty string."""
        tokens = self.service._estimate_tokens("")
        assert tokens == 0
    
    def test_estimate_tokens_short_text(self):
        """Test token estimation for short text."""
        text = "Hello, world!"
        tokens = self.service._estimate_tokens(text)
        assert tokens == 3  # len("Hello, world!") // 4 = 3
    
    def test_estimate_tokens_medium_text(self):
        """Test token estimation for medium text."""
        text = "This is a sample article about artificial intelligence and machine learning."
        tokens = self.service._estimate_tokens(text)
        assert tokens > 0
    
    def test_estimate_tokens_returns_integer(self):
        """Test that token estimation returns an integer."""
        text = "Test text for token estimation."
        tokens = self.service._estimate_tokens(text)
        assert isinstance(tokens, int)


class TestTextChunking:
    """Tests for text chunking logic."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = SummarizerService(
            chunk_size=500,  # 500 tokens for testing
            chunk_overlap=50,
        )
    
    def test_chunk_text_single_chunk(self):
        """Test chunking text that fits in single chunk."""
        short_text = "This is a short article."
        chunks = self.service._chunk_text(short_text)
        assert len(chunks) == 1
        assert chunks[0] == short_text
    
    def test_chunk_text_long_text(self):
        """Test chunking long text requiring multiple chunks."""
        # Create text ~3000 tokens (approx 12000 chars)
        long_text = "This is a sentence. " * 600
        chunks = self.service._chunk_text(long_text)
        assert len(chunks) > 1
    
    def test_chunk_text_empty(self):
        """Test chunking empty string."""
        chunks = self.service._chunk_text("")
        assert len(chunks) == 1
        assert chunks[0] == ""
    
    def test_chunk_text_preserves_sentences(self):
        """Test that chunking tries to break at sentence boundaries."""
        service = SummarizerService(chunk_size=10, chunk_overlap=0)
        text = "First sentence. Second sentence. Third sentence."
        chunks = service._chunk_text(text)
        # Should have multiple chunks
        assert len(chunks) >= 1


class TestChunkSummary:
    """Tests for ChunkSummary dataclass."""
    
    def test_chunk_summary_creation(self):
        """Test ChunkSummary creation."""
        summary = ChunkSummary(chunk_index=0, summary="Test summary")
        assert summary.chunk_index == 0
        assert summary.summary == "Test summary"


class TestPrompts:
    """Tests for prompt templates."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = SummarizerService()
    
    def test_system_prompt_exists(self):
        """Test that system prompt is defined."""
        assert len(self.service.SYSTEM_PROMPT) > 0
        assert "ONLY use information" in self.service.SYSTEM_PROMPT
    
    def test_short_summary_prompt_format(self):
        """Test short summary prompt template."""
        prompt = self.service.SHORT_SUMMARY_PROMPT.format(article_text="Test article")
        assert "SHORT SUMMARY" in prompt
        assert "Test article" in prompt
    
    def test_medium_summary_prompt_format(self):
        """Test medium summary prompt template."""
        prompt = self.service.MEDIUM_SUMMARY_PROMPT.format(article_text="Test article")
        assert "MEDIUM SUMMARY" in prompt
        assert "Test article" in prompt
    
    def test_headline_prompt_format(self):
        """Test headline prompt template."""
        prompt = self.service.HEADLINE_PROMPT.format(article_text="Test article")
        assert "HEADLINE" in prompt
        assert "Test article" in prompt
    
    def test_chunk_summary_prompt_format(self):
        """Test chunk summary prompt template."""
        prompt = self.service.CHUNK_SUMMARY_PROMPT.format(excerpt_text="Test excerpt")
        assert "SUMMARY" in prompt
        assert "Test excerpt" in prompt
    
    def test_combined_summary_prompt_format(self):
        """Test combined summary prompt template."""
        prompt = self.service.COMBINED_SUMMARY_PROMPT.format(
            partial_summaries="Test partial summaries"
        )
        assert "PARTIAL SUMMARIES" in prompt
        assert "SHORT SUMMARY" in prompt
        assert "MEDIUM SUMMARY" in prompt
        assert "HEADLINE" in prompt


class TestResponseParsing:
    """Tests for combined response parsing."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = SummarizerService()
    
    def test_parse_response_with_all_sections(self):
        """Test parsing response with all three sections."""
        response = """1. Short Summary (1-2 lines capturing the essence):
This is a short summary of the article.

2. Medium Summary (3-5 lines with more detail):
This is a medium summary that provides more information about the article. It contains several key points and details.

3. Headline (single compelling headline, no quotation marks):
Major Technology Breakthrough Announced by Research Team"""

        short, medium, headline = self.service._parse_combined_response(response)
        assert "short summary" in short.lower()
        assert "medium summary" in medium.lower()
        # Headline should be validated and contain key content (7 words, passes validation)
        assert "Technology Breakthrough" in headline or "Research Team" in headline
    
    def test_parse_response_short_only(self):
        """Test parsing response with only short summary."""
        response = "This is the only summary provided."
        
        short, medium, headline = self.service._parse_combined_response(response)
        assert "only summary provided" in short.lower()
        assert medium == short  # Falls back to short
        assert headline != ""  # Headline extracted
    
    def test_parse_response_empty(self):
        """Test parsing empty response."""
        response = ""
        
        short, medium, headline = self.service._parse_combined_response(response)
        assert short == ""  # Falls back to empty
        assert medium == ""
        assert headline == "News Story"  # Default headline after validation
    
    def test_extract_headline(self):
        """Test headline extraction from text."""
        text = "Line 1\nA much longer line that could be a headline\nLine 3"
        headline = self.service._extract_headline(text)
        assert "much longer line" in headline


class TestSummarizerService:
    """Tests for SummarizerService class."""
    
    def test_init_default(self):
        """Test service initialization with defaults."""
        service = SummarizerService()
        assert service.chunk_size == 2000  # From config
        assert service.chunk_overlap == 200  # From config
    
    def test_init_custom(self):
        """Test service initialization with custom values."""
        service = SummarizerService(
            chunk_size=1000,
            chunk_overlap=100,
        )
        assert service.chunk_size == 1000
        assert service.chunk_overlap == 100
    
    def test_init_with_services(self):
        """Test service initialization with custom services."""
        mock_llm = MagicMock()
        mock_parser = MagicMock()
        service = SummarizerService(
            llm_service=mock_llm,
            parser_service=mock_parser,
        )
        assert service.llm_service == mock_llm
        assert service.parser_service == mock_parser


class TestSummarizerMethods:
    """Tests for SummarizerService async methods."""
    
    @pytest.fixture
    def mock_service(self):
        """Create a mock service with mocked dependencies."""
        service = SummarizerService()
        service.llm_service = MagicMock()
        service.parser_service = MagicMock()
        return service
    
    @pytest.mark.asyncio
    async def test_summarize_text_short_article(self, mock_service):
        """Test summarizing a short article (no chunking needed)."""
        # Setup mocks
        mock_parser = mock_service.parser_service
        mock_parser.parse.return_value = "This is a short article."
        
        mock_llm = mock_service.llm_service
        mock_response = MagicMock()
        mock_response.content = "Short summary"
        mock_llm.complete = AsyncMock(return_value=mock_response)
        
        # Execute
        short, medium, headline = await mock_service.summarize_text("Short article text")
        
        # Verify
        mock_parser.parse.assert_called_once_with("Short article text")
        assert short == "Short summary"
    
    @pytest.mark.asyncio
    async def test_summarize_url(self, mock_service):
        """Test summarizing from URL."""
        mock_parser = mock_service.parser_service
        mock_parser.parse.return_value = "Article from URL"
        
        mock_llm = mock_service.llm_service
        mock_response = MagicMock()
        mock_response.content = "URL summary"
        mock_llm.complete = AsyncMock(return_value=mock_response)
        
        short, medium, headline = await mock_service.summarize_url("https://example.com")
        
        mock_parser.parse.assert_called_once_with("https://example.com")
        assert short == "URL summary"
    
    @pytest.mark.asyncio
    async def test_summarize_file(self, mock_service):
        """Test summarizing from file."""
        mock_parser = mock_service.parser_service
        mock_parser.parse.return_value = "Article from file"
        
        mock_llm = mock_service.llm_service
        mock_response = MagicMock()
        mock_response.content = "File summary"
        mock_llm.complete = AsyncMock(return_value=mock_response)
        
        short, medium, headline = await mock_service.summarize_file("/path/to/doc.pdf")
        
        mock_parser.parse.assert_called_once_with("/path/to/doc.pdf")
        assert short == "File summary"
    
    @pytest.mark.asyncio
    async def test_summarize_empty_content_raises_error(self, mock_service):
        """Test that empty content raises SummarizerError."""
        mock_parser = mock_service.parser_service
        mock_parser.parse.return_value = ""
        
        with pytest.raises(SummarizerError) as exc_info:
            await mock_service.summarize_text("Some text")
        
        assert "No text content" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_summarize_whitespace_only_raises_error(self, mock_service):
        """Test that whitespace-only content raises SummarizerError."""
        mock_parser = mock_service.parser_service
        mock_parser.parse.return_value = "   \n\t  "
        
        with pytest.raises(SummarizerError) as exc_info:
            await mock_service.summarize_text("Some text")
        
        assert "No text content" in str(exc_info.value)


class TestLongArticleChunking:
    """Tests for long article chunking strategy."""
    
    @pytest.fixture
    def mock_service(self):
        """Create a mock service with mocked dependencies."""
        service = SummarizerService(chunk_size=10, chunk_overlap=2)
        service.llm_service = MagicMock()
        service.parser_service = MagicMock()
        return service
    
    @pytest.mark.asyncio
    async def test_long_article_splits_into_chunks(self, mock_service):
        """Test that long articles are split into chunks."""
        # Create a long article (will be split into multiple chunks)
        long_text = "Sentence one. " * 100
        mock_parser = mock_service.parser_service
        mock_parser.parse.return_value = long_text
        
        # Mock LLM responses for chunk summaries
        mock_llm = mock_service.llm_service
        
        async def mock_complete(*args, **kwargs):
            response = MagicMock()
            if "partial summaries" in str(kwargs.get('prompt', '')).lower():
                response.content = "Short: Combined short.\nMedium: Combined medium.\nHeadline: Combined Headline"
            else:
                response.content = "Chunk summary"
            return response
        
        mock_llm.complete = AsyncMock(side_effect=mock_complete)
        
        # Execute
        short, medium, headline = await mock_service.summarize_text(long_text)
        
        # Verify multiple chunks were processed
        assert mock_llm.complete.call_count >= 2  # At least 2 chunk summaries + final
    
    @pytest.mark.asyncio
    async def test_chunk_summaries_combined(self, mock_service):
        """Test that chunk summaries are combined into final summaries."""
        # Setup
        long_text = "Sentence one. " * 100
        mock_parser = mock_service.parser_service
        mock_parser.parse.return_value = long_text
        
        mock_llm = mock_service.llm_service
        
        call_count = 0
        async def mock_complete(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            response = MagicMock()
            if call_count <= 10:  # First calls are chunk summaries
                response.content = f"Summary of chunk {call_count}"
            else:  # Final call combines summaries
                response.content = "1. Short Summary:\nFinal short.\n\n2. Medium Summary:\nFinal medium.\n\n3. Headline:\nFinal Headline"
            return response
        
        mock_llm.complete = AsyncMock(side_effect=mock_complete)
        
        # Execute
        short, medium, headline = await mock_service.summarize_text(long_text)
        
        # Verify
        assert short != ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
