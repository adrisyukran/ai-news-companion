"""
AI News Companion - Translate API Tests

Tests cover:
- POST /api/translate endpoint
- English to Bahasa Melayu translation
- Bahasa Melayu to English translation
- Same language passthrough
- Invalid language code validation
- Error handling
- Request/response schema validation
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from backend.main import app
from backend.models.schemas import TranslateRequest, TranslateResponse
from backend.services.translator import TranslatorService


class TestTranslateRequestSchema:
    """Tests for TranslateRequest schema."""
    
    def test_create_en_to_bm_request(self):
        """Test creating English to BM request."""
        request = TranslateRequest(
            text="The Prime Minister announced new policies.",
            source_lang="en",
            target_lang="bm",
        )
        assert request.text == "The Prime Minister announced new policies."
        assert request.source_lang == "en"
        assert request.target_lang == "bm"
    
    def test_create_bm_to_en_request(self):
        """Test creating BM to English request."""
        request = TranslateRequest(
            text="Perdana Menteri mengumumkan dasar baharu.",
            source_lang="bm",
            target_lang="en",
        )
        assert request.source_lang == "bm"
        assert request.target_lang == "en"
    
    def test_language_code_case_insensitive(self):
        """Test that language codes are case insensitive."""
        request = TranslateRequest(
            text="Test text",
            source_lang="EN",
            target_lang="BM",
        )
        assert request.source_lang == "en"
        assert request.target_lang == "bm"
    
    def test_whitespace_stripped_from_codes(self):
        """Test that whitespace is stripped from language codes."""
        request = TranslateRequest(
            text="Test text",
            source_lang=" en ",
            target_lang=" bm ",
        )
        assert request.source_lang == "en"
        assert request.target_lang == "bm"


class TestTranslateResponseSchema:
    """Tests for TranslateResponse schema."""
    
    def test_create_response(self):
        """Test creating a response."""
        response = TranslateResponse(
            translated_text="Teks terjemahan.",
            maintained_tone="news",
        )
        assert response.translated_text == "Teks terjemahan."
        assert response.maintained_tone == "news"
    
    def test_response_default_tone(self):
        """Test that maintained_tone defaults to news."""
        response = TranslateResponse(
            translated_text="Translated text.",
        )
        assert response.maintained_tone == "news"


class TestTranslateEndpoint:
    """Tests for /api/translate endpoint."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_translator(self):
        """Create mock translator service."""
        with patch('backend.routers.translate.get_translator_service') as mock:
            service = MagicMock(spec=TranslatorService)
            service.translate = AsyncMock(return_value=TranslateResponse(
                translated_text="Teks terjemahan.",
                maintained_tone="news",
            ))
            mock.return_value = service
            yield service
    
    def test_translate_en_to_bm_success(self, client, mock_translator):
        """Test successful English to Bahasa Melayu translation."""
        response = client.post(
            "/api/translate",
            json={
                "text": "The Prime Minister announced new policies.",
                "source_lang": "en",
                "target_lang": "bm",
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "translated_text" in data
        assert data["maintained_tone"] == "news"
        mock_translator.translate.assert_called_once()
    
    def test_translate_bm_to_en_success(self, client, mock_translator):
        """Test successful Bahasa Melayu to English translation."""
        response = client.post(
            "/api/translate",
            json={
                "text": "Perdana Menteri mengumumkan dasar baharu.",
                "source_lang": "bm",
                "target_lang": "en",
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "translated_text" in data
        assert data["maintained_tone"] == "news"
        mock_translator.translate.assert_called_once()
    
    def test_translate_same_language_passthrough(self, client, mock_translator):
        """Test same language returns original text."""
        response = client.post(
            "/api/translate",
            json={
                "text": "Same language text.",
                "source_lang": "en",
                "target_lang": "en",
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["translated_text"] == "Teks terjemahan."
    
    def test_translate_invalid_source_lang(self, client, mock_translator):
        """Test that invalid source_lang returns 422 (Pydantic validation error)."""
        response = client.post(
            "/api/translate",
            json={
                "text": "Test text",
                "source_lang": "french",
                "target_lang": "bm",
            },
        )
        
        # FastAPI returns 422 for Pydantic validation errors (pattern mismatch)
        assert response.status_code == 422
        mock_translator.translate.assert_not_called()
    
    def test_translate_invalid_target_lang(self, client, mock_translator):
        """Test that invalid target_lang returns 422 (Pydantic validation error)."""
        response = client.post(
            "/api/translate",
            json={
                "text": "Test text",
                "source_lang": "en",
                "target_lang": "invalid",
            },
        )
        
        # FastAPI returns 422 for Pydantic validation errors (pattern mismatch)
        assert response.status_code == 422
        mock_translator.translate.assert_not_called()
    
    def test_translate_missing_text(self, client, mock_translator):
        """Test that missing text returns 422 validation error."""
        response = client.post(
            "/api/translate",
            json={
                "source_lang": "en",
                "target_lang": "bm",
            },
        )
        
        assert response.status_code == 422  # FastAPI validation error
        mock_translator.translate.assert_not_called()
    
    def test_translate_missing_source_lang(self, client, mock_translator):
        """Test that missing source_lang returns 422."""
        response = client.post(
            "/api/translate",
            json={
                "text": "Test text",
                "target_lang": "bm",
            },
        )
        
        assert response.status_code == 422
        mock_translator.translate.assert_not_called()
    
    def test_translate_missing_target_lang(self, client, mock_translator):
        """Test that missing target_lang returns 422."""
        response = client.post(
            "/api/translate",
            json={
                "text": "Test text",
                "source_lang": "en",
            },
        )
        
        assert response.status_code == 422
        mock_translator.translate.assert_not_called()
    
    def test_translate_empty_text(self, client, mock_translator):
        """Test that empty text returns 422."""
        response = client.post(
            "/api/translate",
            json={
                "text": "",
                "source_lang": "en",
                "target_lang": "bm",
            },
        )
        
        assert response.status_code == 422  # min_length validation
        mock_translator.translate.assert_not_called()
    
    def test_translate_service_error(self, client, mock_translator):
        """Test handling of service error."""
        mock_translator.translate = AsyncMock(
            side_effect=Exception("Translation service unavailable")
        )
        
        response = client.post(
            "/api/translate",
            json={
                "text": "Test text",
                "source_lang": "en",
                "target_lang": "bm",
            },
        )
        
        assert response.status_code == 500
        assert "failed" in response.json()["detail"].lower()
    
    def test_translate_value_error(self, client, mock_translator):
        """Test handling of ValueError from translator."""
        mock_translator.translate = AsyncMock(
            side_effect=ValueError("Invalid language code")
        )
        
        response = client.post(
            "/api/translate",
            json={
                "text": "Test text",
                "source_lang": "en",
                "target_lang": "bm",
            },
        )
        
        assert response.status_code == 400
        assert "Invalid language code" in response.json()["detail"]
    
    def test_translate_case_insensitive_lang_codes(self, client, mock_translator):
        """Test that language codes are case insensitive."""
        response = client.post(
            "/api/translate",
            json={
                "text": "Test text",
                "source_lang": "EN",
                "target_lang": "BM",
            },
        )
        
        assert response.status_code == 200
        mock_translator.translate.assert_called_once()


class TestTranslateEndpointJSONContent:
    """Tests for translate endpoint JSON content handling."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_translate_with_news_style_response(self, client):
        """Test translation response maintains news tone."""
        with patch('backend.routers.translate.get_translator_service') as mock_get:
            service = MagicMock()
            service.translate = AsyncMock(return_value=TranslateResponse(
                translated_text="Kerajaan akan melancarkan program baharu untuk membantu rakyat.",
                maintained_tone="news",
            ))
            mock_get.return_value = service
            
            response = client.post(
                "/api/translate",
                json={
                    "text": "The government will launch a new program to help the people.",
                    "source_lang": "en",
                    "target_lang": "bm",
                },
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["maintained_tone"] == "news"
            assert "news" in data["maintained_tone"]
    
    def test_translate_long_text(self, client):
        """Test translation of long text content."""
        long_text = "This is a very long article about various topics. " * 50
        
        with patch('backend.routers.translate.get_translator_service') as mock_get:
            service = MagicMock()
            service.translate = AsyncMock(return_value=TranslateResponse(
                translated_text="Ini adalah artikel yang sangat panjang.",
                maintained_tone="news",
            ))
            mock_get.return_value = service
            
            response = client.post(
                "/api/translate",
                json={
                    "text": long_text,
                    "source_lang": "en",
                    "target_lang": "bm",
                },
            )
            
            assert response.status_code == 200
            service.translate.assert_called_once()


class TestTranslateEndpointResponseFormat:
    """Tests for translate endpoint response format."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_response_has_translated_text_field(self, client):
        """Test that response contains translated_text field."""
        with patch('backend.routers.translate.get_translator_service') as mock_get:
            service = MagicMock()
            service.translate = AsyncMock(return_value=TranslateResponse(
                translated_text="Teks diterjemahkan.",
                maintained_tone="news",
            ))
            mock_get.return_value = service
            
            response = client.post(
                "/api/translate",
                json={
                    "text": "Translated text.",
                    "source_lang": "en",
                    "target_lang": "bm",
                },
            )
            
            data = response.json()
            assert "translated_text" in data
            assert isinstance(data["translated_text"], str)
    
    def test_response_has_maintained_tone_field(self, client):
        """Test that response contains maintained_tone field."""
        with patch('backend.routers.translate.get_translator_service') as mock_get:
            service = MagicMock()
            service.translate = AsyncMock(return_value=TranslateResponse(
                translated_text="Teks.",
                maintained_tone="news",
            ))
            mock_get.return_value = service
            
            response = client.post(
                "/api/translate",
                json={
                    "text": "Text.",
                    "source_lang": "en",
                    "target_lang": "bm",
                },
            )
            
            data = response.json()
            assert "maintained_tone" in data
            assert data["maintained_tone"] == "news"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
