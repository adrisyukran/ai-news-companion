"""
AI News Companion - Translator Service Tests

Tests cover:
- Language code validation
- Same-language passthrough handling
- Prompt building for en↔bm translation
- News style preservation
- TranslatorService async methods
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic import ValidationError

from backend.models.schemas import TranslateRequest, TranslateResponse
from backend.services.translator import (
    TranslatorService,
    LANGUAGE_NAMES,
    TRANSLATION_SYSTEM_PROMPT,
)


class TestLanguageNames:
    """Tests for language name mappings."""
    
    def test_english_name(self):
        """Test English language name mapping."""
        assert LANGUAGE_NAMES["en"] == "English"
    
    def test_malay_name(self):
        """Test Bahasa Melayu language name mapping."""
        assert LANGUAGE_NAMES["bm"] == "Bahasa Melayu"
    
    def test_all_keys_present(self):
        """Test that both en and bm keys exist."""
        assert set(LANGUAGE_NAMES.keys()) == {"en", "bm"}


class TestTranslationSystemPrompt:
    """Tests for the translation system prompt."""
    
    def test_system_prompt_exists(self):
        """Test that system prompt is defined."""
        assert len(TRANSLATION_SYSTEM_PROMPT) > 0
    
    def test_system_prompt_mentions_news_style(self):
        """Test that system prompt emphasizes news style."""
        assert "NEWS" in TRANSLATION_SYSTEM_PROMPT
        assert "formal" in TRANSLATION_SYSTEM_PROMPT.lower()
    
    def test_system_prompt_mentions_accuracy(self):
        """Test that system prompt emphasizes accuracy."""
        assert "ACCURACY" in TRANSLATION_SYSTEM_PROMPT
    
    def test_system_prompt_mentions_malaysian_context(self):
        """Test that system prompt mentions Malaysian context."""
        assert "Malaysian" in TRANSLATION_SYSTEM_PROMPT


class TestTranslatorService:
    """Tests for TranslatorService class."""
    
    def test_init_default(self):
        """Test service initialization with defaults."""
        service = TranslatorService()
        assert service.llm_service is not None
    
    def test_init_with_llm_service(self):
        """Test service initialization with custom LLM service."""
        mock_llm = MagicMock()
        service = TranslatorService(llm_service=mock_llm)
        assert service.llm_service == mock_llm


class TestBuildTranslationPrompt:
    """Tests for prompt building logic."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = TranslatorService()
    
    def test_en_to_bm_prompt(self):
        """Test building English to BM translation prompt."""
        text = "The Prime Minister announced new policies."
        prompt = self.service._build_translation_prompt(text, "en", "bm")
        
        assert "English" in prompt
        assert "Bahasa Melayu" in prompt
        assert text in prompt
    
    def test_bm_to_en_prompt(self):
        """Test building BM to English translation prompt."""
        text = "Perdana Menteri mengumumkan dasar baharu."
        prompt = self.service._build_translation_prompt(text, "bm", "en")
        
        assert "Bahasa Melayu" in prompt
        assert "English" in prompt
        assert text in prompt
    
    def test_prompt_contains_translation_instruction(self):
        """Test that prompt contains translation instruction."""
        prompt = self.service._build_translation_prompt("Test text", "en", "bm")
        assert "Translate" in prompt
    
    def test_prompt_contains_news_style_instruction(self):
        """Test that prompt contains news style instruction."""
        prompt = self.service._build_translation_prompt("Test text", "en", "bm")
        assert "news" in prompt.lower()


class TestTranslateMethod:
    """Tests for the main translate method."""
    
    @pytest.fixture
    def mock_service(self):
        """Create a mock service with mocked LLM."""
        mock_llm = MagicMock()
        service = TranslatorService(llm_service=mock_llm)
        return service
    
    @pytest.mark.asyncio
    async def test_translate_en_to_bm(self, mock_service):
        """Test translating English to Bahasa Melayu."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.content = "Perdana Menteri mengumumkan dasar baharu."
        mock_service.llm_service.complete = AsyncMock(return_value=mock_response)
        
        # Execute
        request = TranslateRequest(
            text="The Prime Minister announced new policies.",
            source_lang="en",
            target_lang="bm",
        )
        response = await mock_service.translate(request)
        
        # Verify
        assert isinstance(response, TranslateResponse)
        assert response.translated_text == "Perdana Menteri mengumumkan dasar baharu."
        assert response.maintained_tone == "news"
        mock_service.llm_service.complete.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_translate_bm_to_en(self, mock_service):
        """Test translating Bahasa Melayu to English."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.content = "The Prime Minister announced new policies."
        mock_service.llm_service.complete = AsyncMock(return_value=mock_response)
        
        # Execute
        request = TranslateRequest(
            text="Perdana Menteri mengumumkan dasar baharu.",
            source_lang="bm",
            target_lang="en",
        )
        response = await mock_service.translate(request)
        
        # Verify
        assert isinstance(response, TranslateResponse)
        assert response.translated_text == "The Prime Minister announced new policies."
        assert response.maintained_tone == "news"
    
    @pytest.mark.asyncio
    async def test_translate_same_language_passthrough(self, mock_service):
        """Test that same language returns original text."""
        # Execute - no LLM call should be made
        request = TranslateRequest(
            text="Same language text.",
            source_lang="en",
            target_lang="en",
        )
        response = await mock_service.translate(request)
        
        # Verify
        assert response.translated_text == "Same language text."
        assert response.maintained_tone == "news"
        mock_service.llm_service.complete.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_translate_invalid_source_lang(self, mock_service):
        """Test translation with invalid source language raises error."""
        # Pydantic validates the pattern before the service gets the request
        with pytest.raises(ValidationError) as exc_info:
            TranslateRequest(
                text="Test text",
                source_lang="invalid",
                target_lang="bm",
            )
        
        assert "source_lang" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_translate_invalid_target_lang(self, mock_service):
        """Test translation with invalid target language raises error."""
        # Pydantic validates the pattern before the service gets the request
        with pytest.raises(ValidationError) as exc_info:
            TranslateRequest(
                text="Test text",
                source_lang="en",
                target_lang="invalid",
            )
        
        assert "target_lang" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_translate_calls_llm_with_correct_params(self, mock_service):
        """Test that LLM is called with correct parameters."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.content = "Translated text"
        mock_service.llm_service.complete = AsyncMock(return_value=mock_response)
        
        # Execute
        request = TranslateRequest(
            text="Test text",
            source_lang="en",
            target_lang="bm",
        )
        await mock_service.translate(request)
        
        # Verify LLM call parameters
        call_kwargs = mock_service.llm_service.complete.call_args.kwargs
        assert "prompt" in call_kwargs
        assert "system_prompt" in call_kwargs
        assert call_kwargs["temperature"] == 0.3
        assert call_kwargs["max_tokens"] == 4096


class TestTranslateConvenienceMethods:
    """Tests for translate_en_to_bm and translate_bm_to_en convenience methods."""
    
    @pytest.fixture
    def mock_service(self):
        """Create a mock service with mocked LLM."""
        mock_llm = MagicMock()
        service = TranslatorService(llm_service=mock_llm)
        return service
    
    @pytest.mark.asyncio
    async def test_translate_en_to_bm_convenience(self, mock_service):
        """Test translate_en_to_bm convenience method."""
        mock_response = MagicMock()
        mock_response.content = "BM translation"
        mock_service.llm_service.complete = AsyncMock(return_value=mock_response)
        
        response = await mock_service.translate_en_to_bm("EN text")
        
        assert response.translated_text == "BM translation"
        mock_service.llm_service.complete.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_translate_bm_to_en_convenience(self, mock_service):
        """Test translate_bm_to_en convenience method."""
        mock_response = MagicMock()
        mock_response.content = "EN translation"
        mock_service.llm_service.complete = AsyncMock(return_value=mock_response)
        
        response = await mock_service.translate_bm_to_en("BM text")
        
        assert response.translated_text == "EN translation"
        mock_service.llm_service.complete.assert_called_once()


class TestTranslateRequest:
    """Tests for TranslateRequest schema validation."""
    
    def test_valid_request_en_to_bm(self):
        """Test valid English to BM request."""
        request = TranslateRequest(
            text="The Prime Minister announced new policies.",
            source_lang="en",
            target_lang="bm",
        )
        assert request.source_lang == "en"
        assert request.target_lang == "bm"
    
    def test_valid_request_bm_to_en(self):
        """Test valid BM to English request."""
        request = TranslateRequest(
            text="Perdana Menteri mengumumkan dasar baharu.",
            source_lang="bm",
            target_lang="en",
        )
        assert request.source_lang == "bm"
        assert request.target_lang == "en"
    
    def test_language_code_normalization(self):
        """Test that language codes are normalized to lowercase."""
        request = TranslateRequest(
            text="Test text",
            source_lang="EN",
            target_lang="BM",
        )
        assert request.source_lang == "en"
        assert request.target_lang == "bm"
    
    def test_whitespace_stripping(self):
        """Test that language codes have whitespace stripped."""
        request = TranslateRequest(
            text="Test text",
            source_lang=" en ",
            target_lang=" bm ",
        )
        assert request.source_lang == "en"
        assert request.target_lang == "bm"


class TestTranslateResponse:
    """Tests for TranslateResponse schema."""
    
    def test_response_creation(self):
        """Test TranslateResponse creation."""
        response = TranslateResponse(
            translated_text="Translated text here.",
            maintained_tone="news",
        )
        assert response.translated_text == "Translated text here."
        assert response.maintained_tone == "news"
    
    def test_response_default_maintained_tone(self):
        """Test that maintained_tone defaults to news."""
        response = TranslateResponse(
            translated_text="Translated text here.",
        )
        assert response.maintained_tone == "news"


class TestTranslateRequestValidation:
    """Tests for TranslateRequest validation."""
    
    def test_text_min_length(self):
        """Test that text has minimum length of 1."""
        with pytest.raises(ValueError):
            TranslateRequest(
                text="",
                source_lang="en",
                target_lang="bm",
            )
    
    def test_text_max_length(self):
        """Test that text has maximum length of 10000."""
        long_text = "x" * 10001
        with pytest.raises(ValueError):
            TranslateRequest(
                text=long_text,
                source_lang="en",
                target_lang="bm",
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
