"""
AI News Companion - Summarize API Tests

Tests cover:
- POST /api/summarize endpoint
- URL-based summarization
- File upload handling
- Plain text summarization
- Error handling
- Request validation
"""
import pytest
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from fastapi.testclient import TestClient

from backend.main import app
from backend.models.schemas import SummarizeRequest, SummarizeResponse
from backend.services.summarizer import SummarizerService, SummarizerError


class TestSummarizeRequest:
    """Tests for SummarizeRequest schema."""
    
    def test_create_with_url(self):
        """Test creating request with URL."""
        request = SummarizeRequest(url="https://example.com/article")
        assert request.url == "https://example.com/article"
    
    def test_create_with_text(self):
        """Test creating request with text."""
        request = SummarizeRequest(text="Article content here")
        assert request.text == "Article content here"
    
    def test_create_with_file_path(self):
        """Test creating request with file path."""
        request = SummarizeRequest(file_path="/path/to/document.pdf")
        assert request.file_path == "/path/to/document.pdf"
    
    def test_create_with_whitespace_stripped(self):
        """Test that whitespace is stripped from inputs."""
        request = SummarizeRequest(text="  Article content  ")
        assert request.text == "Article content"
    
    def test_create_with_multiple_inputs_allowed(self):
        """Test that providing multiple inputs is allowed (validation at API level)."""
        # Pydantic allows multiple fields, API endpoint enforces mutual exclusivity
        request = SummarizeRequest(url="https://example.com", text="Article")
        assert request.url == "https://example.com"
        assert request.text == "Article"
    
    def test_create_with_no_inputs_raises_error(self):
        """Test that providing no inputs raises error."""
        with pytest.raises(ValueError) as exc_info:
            SummarizeRequest()
        assert "At least one" in str(exc_info.value)
    
    def test_get_input_source_url(self):
        """Test getting input source for URL."""
        request = SummarizeRequest(url="https://example.com")
        source_type, source_value = request.get_input_source()
        assert source_type == "url"
        assert source_value == "https://example.com"
    
    def test_get_input_source_text(self):
        """Test getting input source for text."""
        request = SummarizeRequest(text="Article text")
        source_type, source_value = request.get_input_source()
        assert source_type == "text"
        assert source_value == "Article text"


class TestSummarizeResponse:
    """Tests for SummarizeResponse schema."""
    
    def test_create_response(self):
        """Test creating a response."""
        response = SummarizeResponse(
            short_summary="Short summary",
            medium_summary="Medium summary",
            headline="Headline",
        )
        assert response.short_summary == "Short summary"
        assert response.medium_summary == "Medium summary"
        assert response.headline == "Headline"
    
    def test_response_with_long_summary(self):
        """Test response with longer summaries."""
        response = SummarizeResponse(
            short_summary="A" * 100,
            medium_summary="B" * 500,
            headline="C" * 50,
        )
        assert len(response.short_summary) == 100
        assert len(response.medium_summary) == 500
        assert len(response.headline) == 50


class TestSummarizeEndpoint:
    """Tests for /api/summarize endpoint."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_summarizer(self):
        """Create mock summarizer service."""
        with patch('backend.routers.summarize.get_summarizer_service') as mock:
            service = MagicMock(spec=SummarizerService)
            service.summarize_url = AsyncMock(return_value=(
                "Short summary",
                "Medium summary",
                "Headline",
            ))
            service.summarize_text = AsyncMock(return_value=(
                "Short summary",
                "Medium summary",
                "Headline",
            ))
            service.summarize_file = AsyncMock(return_value=(
                "Short summary",
                "Medium summary",
                "Headline",
            ))
            mock.return_value = service
            yield service
    
    def test_summarize_url_success(self, client, mock_summarizer):
        """Test successful URL-based summarization."""
        response = client.post(
            "/api/summarize",
            data={"url": "https://example.com/article"},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "short_summary" in data
        assert "medium_summary" in data
        assert "headline" in data
        mock_summarizer.summarize_url.assert_called_once()
    
    def test_summarize_text_success(self, client, mock_summarizer):
        """Test successful text-based summarization."""
        response = client.post(
            "/api/summarize",
            data={"text": "This is an article to summarize."},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["short_summary"] == "Short summary"
        assert data["medium_summary"] == "Medium summary"
        assert data["headline"] == "Headline"
        mock_summarizer.summarize_text.assert_called_once()
    
    def test_summarize_no_input_raises_error(self, client):
        """Test that no input raises 400 error."""
        response = client.post("/api/summarize", data={})
        
        assert response.status_code == 400
        assert "at least one" in response.json()["detail"].lower()
    
    def test_summarize_multiple_inputs_raises_error(self, client):
        """Test that multiple inputs raises 400 error."""
        response = client.post(
            "/api/summarize",
            data={
                "url": "https://example.com",
                "text": "Article text",
            },
        )
        
        assert response.status_code == 400
        assert "only one" in response.json()["detail"].lower()
    
    def test_summarize_file_upload_pdf(self, client):
        """Test successful PDF file upload with mocked service."""
        pdf_content = b"%PDF-1.4 fake pdf content"
        files = {"file": ("test.pdf", BytesIO(pdf_content), "application/pdf")}
        
        # Mock the file path to pass through without actual processing
        with patch('backend.routers.summarize.get_summarizer_service') as mock_get:
            service = MagicMock()
            service.summarize_file = AsyncMock(return_value=(
                "Short summary",
                "Medium summary",
                "Headline",
            ))
            mock_get.return_value = service
            
            response = client.post("/api/summarize", files=files)
            
            assert response.status_code == 200
            service.summarize_file.assert_called_once()
    
    def test_summarize_file_upload_txt(self, client):
        """Test successful TXT file upload with mocked service."""
        text_content = b"This is a test article."
        files = {"file": ("test.txt", BytesIO(text_content), "text/plain")}
        
        with patch('backend.routers.summarize.get_summarizer_service') as mock_get:
            service = MagicMock()
            service.summarize_file = AsyncMock(return_value=(
                "Short summary",
                "Medium summary",
                "Headline",
            ))
            mock_get.return_value = service
            
            response = client.post("/api/summarize", files=files)
            
            assert response.status_code == 200
            service.summarize_file.assert_called_once()
    
    def test_summarize_file_upload_docx(self, client):
        """Test successful DOCX file upload with mocked service."""
        docx_content = b"PK fake docx content"  # DOCX is ZIP-based
        files = {"file": ("test.docx", BytesIO(docx_content), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
        
        with patch('backend.routers.summarize.get_summarizer_service') as mock_get:
            service = MagicMock()
            service.summarize_file = AsyncMock(return_value=(
                "Short summary",
                "Medium summary",
                "Headline",
            ))
            mock_get.return_value = service
            
            response = client.post("/api/summarize", files=files)
            
            assert response.status_code == 200
            service.summarize_file.assert_called_once()
    
    def test_summarize_file_unsupported_type(self, client):
        """Test that unsupported file types are rejected."""
        html_content = b"<html>Not a supported file</html>"
        files = {"file": ("test.html", BytesIO(html_content), "text/html")}
        
        response = client.post("/api/summarize", files=files)
        
        assert response.status_code == 400
        assert "unsupported" in response.json()["detail"].lower()
    
    def test_summarize_summarizer_error(self, client, mock_summarizer):
        """Test handling of SummarizerError."""
        mock_summarizer.summarize_url = AsyncMock(
            side_effect=SummarizerError("Test error")
        )
        
        response = client.post(
            "/api/summarize",
            data={"url": "https://example.com"},
        )
        
        assert response.status_code == 500
        assert "failed" in response.json()["detail"].lower()
    
    def test_summarize_parser_error(self, client, mock_summarizer):
        """Test handling of ParserError in summarizer."""
        mock_summarizer.summarize_url = AsyncMock(
            side_effect=SummarizerError("Failed to parse URL")
        )
        
        response = client.post(
            "/api/summarize",
            data={"url": "https://invalid-domain.xyz"},
        )
        
        assert response.status_code == 500
        assert "failed" in response.json()["detail"].lower()


class TestHealthEndpoint:
    """Tests for health check endpoint."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/api/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "summarize"


class TestRootEndpoint:
    """Tests for root endpoint."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_root(self, client):
        """Test root endpoint."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data


class TestMainHealthEndpoint:
    """Tests for main /health endpoint."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_main_health(self, client):
        """Test main health endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
