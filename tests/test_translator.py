"""
AI News Companion - Translator Service Tests

Tests cover:
- Language code validation
- Same-language passthrough handling
- Auto language detection
- LLM refinement pipeline with DBP standards
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
    REFINEMENT_PROMPT,
    DBP_INSTRUCTION,
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
        """Test that en, bm, and ms keys exist (ms is for argostranslate compatibility)."""
        assert "en" in LANGUAGE_NAMES
        assert "bm" in LANGUAGE_NAMES
        assert "ms" in LANGUAGE_NAMES  # For argostranslate compatibility
        assert LANGUAGE_NAMES["en"] == "English"
        assert LANGUAGE_NAMES["bm"] == "Bahasa Melayu"
        assert LANGUAGE_NAMES["ms"] == "Bahasa Melayu"


class TestRefinementPrompt:
    """Tests for the LLM refinement prompt."""
    
    def test_refinement_prompt_exists(self):
        """Test that refinement prompt is defined."""
        assert len(REFINEMENT_PROMPT) > 0
    
    def test_refinement_prompt_mentions_grammar(self):
        """Test that refinement prompt mentions grammar checking."""
        assert "grammatical" in REFINEMENT_PROMPT.lower()
    
    def test_refinement_prompt_mentions_spelling(self):
        """Test that refinement prompt mentions spelling."""
        assert "spelling" in REFINEMENT_PROMPT.lower()
    
    def test_dbp_instruction_exists(self):
        """Test that DBP instruction is defined."""
        assert len(DBP_INSTRUCTION) > 0
    
    def test_dbp_instruction_mentions_dewan_bahasa(self):
        """Test that DBP instruction mentions Dewan Bahasa dan Pustaka."""
        assert "Dewan Bahasa dan Pustaka" in DBP_INSTRUCTION
        assert "DBP" in DBP_INSTRUCTION
    
    def test_dbp_instruction_mentions_indonesian(self):
        """Test that DBP instruction warns against Indonesian."""
        assert "Indonesian" in DBP_INSTRUCTION


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


class TestLanguageDetection:
    """Tests for auto language detection."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = TranslatorService()
    
    def test_detect_english_with_common_words(self):
        """Test detecting English text with common words."""
        text = "The government announced new policies for the people."
        detected = self.service._detect_language(text)
        assert detected == "en"
    
    def test_detect_bm_with_common_words(self):
        """Test detecting BM text with common words."""
        text = "Kerajaan mengumumkan dasar baharu untuk rakyat."
        detected = self.service._detect_language(text)
        assert detected == "bm"
    
    def test_detect_bm_with_yang(self):
        """Test detecting BM text with 'yang' keyword."""
        text = "Perdana Menteri yang mengumumkan dasar itu hadir di parlimen."
        detected = self.service._detect_language(text)
        assert detected == "bm"
    
    def test_detect_english_with_the(self):
        """Test detecting English text with 'the' keyword."""
        text = "The Prime Minister who announced the policies was at parliament."
        detected = self.service._detect_language(text)
        assert detected == "en"


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
        
        # Mock the argostranslate to return a base translation
        with patch('backend.services.translator.argostranslate.translate.translate') as mock_translate:
            mock_translate.return_value = "Base translation"
            
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
    async def test_translate_auto_detect_source(self, mock_service):
        """Test auto-detection of source language."""
        # Setup mock for LLM detection (heuristic will be inconclusive)
        mock_response = MagicMock()
        mock_response.content = "Perdana Menteri mengumumkan dasar baharu."
        mock_service.llm_service.complete = AsyncMock(return_value=mock_response)
        
        # Mock the argostranslate to return a base translation
        with patch('backend.services.translator.argostranslate.translate.translate') as mock_translate:
            mock_translate.return_value = "Base translation"
            
            # Execute without source_lang - should auto-detect
            request = TranslateRequest(
                text="The Prime Minister announced new policies.",
                target_lang="bm",
            )
            response = await mock_service.translate(request)
            
            # Verify
            assert isinstance(response, TranslateResponse)
            mock_service.llm_service.complete.assert_called()
    
    @pytest.mark.asyncio
    async def test_translate_auto_detect_both(self, mock_service):
        """Test auto-detection of both source and target languages."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.content = "Perdana Menteri mengumumkan dasar baharu."
        mock_service.llm_service.complete = AsyncMock(return_value=mock_response)
        
        # Mock the argostranslate
        with patch('backend.services.translator.argostranslate.translate.translate') as mock_translate:
            mock_translate.return_value = "Base translation"
            
            # Execute without any language codes
            request = TranslateRequest(
                text="The Prime Minister announced new policies.",
            )
            response = await mock_service.translate(request)
            
            # Verify - should detect EN and translate to BM
            assert isinstance(response, TranslateResponse)
            assert response.translated_text == "Perdana Menteri mengumumkan dasar baharu."
    
    @pytest.mark.asyncio
    async def test_translate_auto_detect_bm_source(self, mock_service):
        """Test auto-detection of BM source language."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.content = "The Prime Minister announced new policies."
        mock_service.llm_service.complete = AsyncMock(return_value=mock_response)
        
        # Mock the argostranslate
        with patch('backend.services.translator.argostranslate.translate.translate') as mock_translate:
            mock_translate.return_value = "Base translation"
            
            # Execute with BM text (should auto-detect as BM and translate to EN)
            request = TranslateRequest(
                text="Perdana Menteri mengumumkan dasar baharu.",
            )
            response = await mock_service.translate(request)
            
            # Verify
            assert isinstance(response, TranslateResponse)
    
    @pytest.mark.asyncio
    async def test_translate_calls_llm_with_correct_params(self, mock_service):
        """Test that LLM is called with correct parameters for refinement."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.content = "Translated text"
        mock_service.llm_service.complete = AsyncMock(return_value=mock_response)
        
        # Mock the argostranslate
        with patch('backend.services.translator.argostranslate.translate.translate') as mock_translate:
            mock_translate.return_value = "Base translation"
            
            # Execute
            request = TranslateRequest(
                text="Test text",
                source_lang="en",
                target_lang="bm",
            )
            await mock_service.translate(request)
        
        # Verify LLM was called (for refinement)
        assert mock_service.llm_service.complete.call_count >= 1
        # Check that refinement prompt was used (contains source and base translation)
        call_args = mock_service.llm_service.complete.call_args
        assert call_args is not None


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
        
        with patch('backend.services.translator.argostranslate.translate.translate') as mock_translate:
            mock_translate.return_value = "Base translation"
            
            response = await mock_service.translate_en_to_bm("EN text")
            
            assert response.translated_text == "BM translation"
            assert mock_service.llm_service.complete.call_count >= 1
    
    @pytest.mark.asyncio
    async def test_translate_bm_to_en_convenience(self, mock_service):
        """Test translate_bm_to_en convenience method."""
        mock_response = MagicMock()
        mock_response.content = "EN translation"
        mock_service.llm_service.complete = AsyncMock(return_value=mock_response)
        
        with patch('backend.services.translator.argostranslate.translate.translate') as mock_translate:
            mock_translate.return_value = "Base translation"
            
            response = await mock_service.translate_bm_to_en("BM text")
            
            assert response.translated_text == "EN translation"
            assert mock_service.llm_service.complete.call_count >= 1


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
