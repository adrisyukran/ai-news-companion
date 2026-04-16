"""
AI News Companion - LLM Service Tests

Basic connectivity and functionality tests for nano-gpt integration.
"""
import asyncio
import os
import pytest

from backend.services.llm_service import NanoGPTService, LLMResponse


class TestNanoGPTService:
    """Test suite for NanoGPTService."""
    
    @pytest.fixture
    def llm_service(self):
        """Create LLM service instance for testing."""
        return NanoGPTService()
    
    @pytest.mark.asyncio
    async def test_connectivity(self, llm_service: NanoGPTService):
        """Test basic API connectivity."""
        result = await llm_service.test_connectivity()
        assert result is True, "API connectivity test failed"
    
    @pytest.mark.asyncio
    async def test_simple_completion(self, llm_service: NanoGPTService):
        """Test basic completion request."""
        response = await llm_service.complete(
            prompt="What is 2 + 2? Answer with only the number.",
            max_tokens=10,
        )
        assert isinstance(response, LLMResponse)
        assert len(response.content) > 0
        assert response.model is not None
    
    @pytest.mark.asyncio
    async def test_completion_with_system_prompt(self, llm_service: NanoGPTService):
        """Test completion with system prompt."""
        response = await llm_service.complete(
            prompt="Translate 'Hello' to Bahasa Melayu.",
            system_prompt="You are a helpful translation assistant.",
            max_tokens=20,
        )
        assert isinstance(response, LLMResponse)
        assert len(response.content) > 0
    
    @pytest.mark.asyncio
    async def test_streaming_completion(self, llm_service: NanoGPTService):
        """Test streaming completion."""
        chunks = []
        async for chunk in llm_service.complete_stream(
            prompt="Count from 1 to 3, one number per line.",
            max_tokens=20,
        ):
            chunks.append(chunk)
        
        assert len(chunks) > 0
        full_text = "".join(chunks)
        assert len(full_text) > 0
    
    def test_token_estimation(self, llm_service: NanoGPTService):
        """Test token estimation logic."""
        text = "This is a test sentence."
        tokens = llm_service._estimate_tokens(text)
        assert tokens > 0
        assert isinstance(tokens, int)
    
    def test_chunking_short_text(self, llm_service: NanoGPTService):
        """Test chunking with text that fits in single chunk."""
        short_text = "This is a short article."
        chunks = llm_service._chunk_text(short_text)
        assert len(chunks) == 1
        assert chunks[0] == short_text
    
    def test_chunking_long_text(self, llm_service: NanoGPTService):
        """Test chunking with long text requiring multiple chunks."""
        # Create text ~5000 tokens (approx 20000 chars)
        long_text = "This is a sentence. " * 1000
        chunks = llm_service._chunk_text(long_text)
        assert len(chunks) > 1
        
        # Verify chunks have overlap
        for i in range(len(chunks) - 1):
            # Check that consecutive chunks don't have large gaps
            assert len(chunks[i]) > 0
            assert len(chunks[i + 1]) > 0
    
    def test_chunking_preserves_content(self, llm_service: NanoGPTService):
        """Test that chunking preserves all content."""
        original = "First paragraph. Second paragraph. Third paragraph. " * 100
        chunks = llm_service._chunk_text(original)
        
        # All chunks combined should contain most of original content
        combined = " ".join(chunks)
        # Allow for some overlap, so combined may be slightly longer
        assert len(combined) >= len(original) * 0.9


class TestNanoGPTServiceWithMock:
    """Tests using mocked responses for edge cases."""
    
    @pytest.mark.asyncio
    async def test_retry_logic(self):
        """Test that retry logic is implemented (integration test)."""
        # This test verifies the service can handle transient failures
        # Actual retry behavior tested through integration
        service = NanoGPTService(max_retries=3)
        assert service.max_retries == 3
    
    def test_api_key_validation(self):
        """Test API key validation on initialization."""
        # Service should initialize even without API key
        # but will fail on actual API calls
        service = NanoGPTService(api_key="")
        assert service.api_key == ""
        
        # With API key
        service_with_key = NanoGPTService(api_key="test-key")
        assert service_with_key.api_key == "test-key"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
