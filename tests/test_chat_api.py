"""
AI News Companion - Chat API Tests

Tests for /api/chat endpoint including:
- Chat endpoint
- Load article endpoint
- Session management endpoints
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
import uuid

# Import the app for testing
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.main import app
from backend.models.schemas import ChatRequest, ChatResponse


class TestChatEndpoint:
    """Tests for POST /api/chat endpoint."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_rag_service(self):
        """Create a mock RAG service."""
        mock = MagicMock()
        mock.session_exists = MagicMock(return_value=True)
        mock.chat = AsyncMock(return_value={
            "answer": "Based on the article, AI refers to intelligence...",
            "session_id": "550e8400-e29b-41d4-a716-446655440000",
        })
        return mock
    
    def test_chat_invalid_uuid(self, client):
        """Test that invalid UUID format is rejected."""
        response = client.post(
            "/api/chat",
            json={
                "session_id": "not-a-uuid",
                "question": "What is AI?",
            }
        )
        assert response.status_code == 422  # Validation error
    
    def test_chat_missing_question(self, client):
        """Test that missing question is rejected."""
        response = client.post(
            "/api/chat",
            json={
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
            }
        )
        assert response.status_code == 422
    
    def test_chat_empty_question(self, client):
        """Test that empty question is rejected."""
        response = client.post(
            "/api/chat",
            json={
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "question": "",
            }
        )
        assert response.status_code == 422
    
    def test_chat_session_not_found(self, client):
        """Test chat with non-existent session."""
        with patch('backend.routers.chat.get_rag_service') as mock_get:
            mock_service = MagicMock()
            mock_service.session_exists = MagicMock(return_value=False)
            mock_get.return_value = mock_service
            
            response = client.post(
                "/api/chat",
                json={
                    "session_id": "550e8400-e29b-41d4-a716-446655440000",
                    "question": "What is AI?",
                }
            )
            assert response.status_code == 404
            assert "not found" in response.json()["detail"].lower()


class TestLoadArticleEndpoint:
    """Tests for POST /api/chat/load endpoint."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_load_article_basic(self, client):
        """Test basic article loading."""
        with patch('backend.routers.chat.get_rag_service') as mock_get:
            mock_service = MagicMock()
            mock_service.create_session = MagicMock(return_value="550e8400-e29b-41d4-a716-446655440000")
            mock_get.return_value = mock_service
            
            response = client.post(
                "/api/chat/load",
                json={
                    "text": "This is a test article about AI.",
                }
            )
            assert response.status_code == 200
            data = response.json()
            assert "session_id" in data
            assert data["status"] == "loaded"
    
    def test_load_article_with_source(self, client):
        """Test article loading with source metadata."""
        with patch('backend.routers.chat.get_rag_service') as mock_get:
            mock_service = MagicMock()
            mock_service.create_session = MagicMock(return_value="550e8400-e29b-41d4-a716-446655440000")
            mock_get.return_value = mock_service
            
            response = client.post(
                "/api/chat/load",
                json={
                    "text": "Article content",
                    "source_type": "url",
                    "source_value": "https://example.com/news",
                }
            )
            assert response.status_code == 200
    
    def test_load_article_missing_text(self, client):
        """Test that missing text is rejected."""
        response = client.post(
            "/api/chat/load",
            json={}
        )
        assert response.status_code == 422
    
    def test_load_article_empty_text(self, client):
        """Test that empty text is rejected."""
        response = client.post(
            "/api/chat/load",
            json={
                "text": "",
            }
        )
        # Should be rejected - either 422 (validation) or 500 (error handling)
        assert response.status_code in (422, 500)


class TestDeleteSessionEndpoint:
    """Tests for DELETE /api/chat/session/{session_id} endpoint."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_delete_session_success(self, client):
        """Test successful session deletion."""
        with patch('backend.routers.chat.get_rag_service') as mock_get:
            mock_service = MagicMock()
            mock_service.delete_session = MagicMock(return_value=True)
            mock_get.return_value = mock_service
            
            response = client.delete(
                "/api/chat/session/550e8400-e29b-41d4-a716-446655440000"
            )
            assert response.status_code == 200
            assert response.json()["status"] == "deleted"
    
    def test_delete_session_not_found(self, client):
        """Test deleting non-existent session."""
        with patch('backend.routers.chat.get_rag_service') as mock_get:
            mock_service = MagicMock()
            mock_service.delete_session = MagicMock(return_value=False)
            mock_get.return_value = mock_service
            
            response = client.delete(
                "/api/chat/session/550e8400-e29b-41d4-a716-446655440000"
            )
            assert response.status_code == 404


class TestCheckSessionEndpoint:
    """Tests for GET /api/chat/session/{session_id}/exists endpoint."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_check_session_exists(self, client):
        """Test checking existing session."""
        with patch('backend.routers.chat.get_rag_service') as mock_get:
            mock_service = MagicMock()
            mock_service.session_exists = MagicMock(return_value=True)
            mock_get.return_value = mock_service
            
            response = client.get(
                "/api/chat/session/550e8400-e29b-41d4-a716-446655440000/exists"
            )
            assert response.status_code == 200
            assert response.json()["exists"] is True
    
    def test_check_session_not_exists(self, client):
        """Test checking non-existing session."""
        with patch('backend.routers.chat.get_rag_service') as mock_get:
            mock_service = MagicMock()
            mock_service.session_exists = MagicMock(return_value=False)
            mock_get.return_value = mock_service
            
            response = client.get(
                "/api/chat/session/550e8400-e29b-41d4-a716-446655440000/exists"
            )
            assert response.status_code == 200
            assert response.json()["exists"] is False


class TestChatRequestSchema:
    """Tests for ChatRequest schema validation."""
    
    def test_valid_uuid_format(self):
        """Test that valid UUIDs are accepted."""
        request = ChatRequest(
            session_id="550e8400-e29b-41d4-a716-446655440000",
            question="What is AI?"
        )
        assert request.session_id == "550e8400-e29b-41d4-a716-446655440000"
    
    def test_invalid_uuid_format(self):
        """Test that invalid UUIDs are rejected."""
        with pytest.raises(ValueError):
            ChatRequest(
                session_id="not-a-valid-uuid",
                question="What is AI?"
            )
    
    def test_empty_question(self):
        """Test that empty question is rejected."""
        with pytest.raises(ValueError):
            ChatRequest(
                session_id="550e8400-e29b-41d4-a716-446655440000",
                question=""
            )


class TestChatResponseSchema:
    """Tests for ChatResponse schema."""
    
    def test_valid_response(self):
        """Test creating a valid response."""
        response = ChatResponse(
            answer="Based on the article...",
            session_id="550e8400-e29b-41d4-a716-446655440000"
        )
        assert response.answer == "Based on the article..."
        assert response.session_id == "550e8400-e29b-41d4-a716-446655440000"
