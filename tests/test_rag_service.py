"""
AI News Companion - RAG Service Tests

Tests for RAG pipeline including:
- Text chunking
- Embeddings and vector store
- Retrieval
- Session management
"""
import pytest
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from backend.services.llm_service import LLMResponse


class TestRAGServiceSessionManagement:
    """Tests for session management functionality."""
    
    @pytest.fixture
    def rag_service(self):
        """Create a RAGService instance with mocked dependencies."""
        with patch('backend.services.rag_service.NanoGPTService'):
            with patch('backend.services.rag_service.SentenceTransformerEmbeddings'):
                from backend.services.rag_service import RAGService
                service = RAGService()
                return service
    
    def test_create_session_id(self, rag_service):
        """Test that created session IDs are valid UUIDs."""
        session_id = rag_service._create_session_id()
        assert uuid.UUID(session_id)  # Should not raise
        assert len(session_id) == 36  # Standard UUID format
    
    def test_get_nonexistent_session(self, rag_service):
        """Test getting a session that doesn't exist."""
        result = rag_service.get_session("nonexistent-id")
        assert result is None
    
    def test_session_exists(self, rag_service):
        """Test checking if a session exists."""
        assert not rag_service.session_exists("nonexistent")
    
    def test_delete_nonexistent_session(self, rag_service):
        """Test deleting a session that doesn't exist."""
        result = rag_service.delete_session("nonexistent")
        assert result is False


class TestRAGServiceChunking:
    """Tests for text chunking functionality."""
    
    @pytest.fixture
    def rag_service(self):
        """Create a RAGService instance with mocked dependencies."""
        with patch('backend.services.rag_service.NanoGPTService'):
            with patch('backend.services.rag_service.SentenceTransformerEmbeddings'):
                from backend.services.rag_service import RAGService
                service = RAGService()
                return service
    
    def test_create_documents(self, rag_service):
        """Test creating documents from text."""
        text = "This is the first paragraph.\n\nThis is the second paragraph."
        session_id = str(uuid.uuid4())
        
        documents = rag_service._create_documents(
            text=text,
            session_id=session_id,
            source_type="text",
            source_value="inline",
        )
        
        assert len(documents) > 0
        for doc in documents:
            assert doc.metadata["session_id"] == session_id
            assert doc.metadata["source_type"] == "text"
    
    def test_create_documents_short_text(self, rag_service):
        """Test that short text still creates at least one document."""
        text = "Short text."
        session_id = str(uuid.uuid4())
        
        documents = rag_service._create_documents(
            text=text,
            session_id=session_id,
            source_type="text",
            source_value="inline",
        )
        
        assert len(documents) >= 1
    
    def test_create_documents_metadata(self, rag_service):
        """Test that document metadata is correctly set."""
        text = "This is a longer piece of text that should be split into multiple chunks for testing purposes."
        session_id = str(uuid.uuid4())
        source_type = "url"
        source_value = "https://example.com/article"
        
        documents = rag_service._create_documents(
            text=text,
            session_id=session_id,
            source_type=source_type,
            source_value=source_value,
        )
        
        for i, doc in enumerate(documents):
            assert doc.metadata["chunk_index"] == i
            assert doc.metadata["total_chunks"] == len(documents)
            assert doc.metadata["session_id"] == session_id
            assert doc.metadata["source_type"] == source_type
            assert doc.metadata["source_value"] == source_value


class TestRAGServiceRetrieval:
    """Tests for retrieval functionality."""
    
    @pytest.fixture
    def rag_service(self):
        """Create a RAGService instance with mocked dependencies."""
        with patch('backend.services.rag_service.NanoGPTService'):
            with patch('backend.services.rag_service.SentenceTransformerEmbeddings'):
                from backend.services.rag_service import RAGService
                service = RAGService()
                return service
    
    def test_retrieve_relevant_chunks_no_session(self, rag_service):
        """Test retrieval with non-existent session."""
        chunks = rag_service.retrieve_relevant_chunks(
            session_id="nonexistent",
            query="What is AI?",
        )
        assert chunks == []


class TestRAGServiceGeneration:
    """Tests for answer generation functionality."""
    
    @pytest.fixture
    def mock_llm_service(self):
        """Create a mock LLM service."""
        mock = MagicMock()
        mock.complete = AsyncMock(return_value=LLMResponse(
            content="Based on the article, AI refers to...",
            model="nano-gpt",
        ))
        mock.model = "nano-gpt"
        return mock
    
    @pytest.fixture
    def rag_service_with_mock_llm(self, mock_llm_service):
        """Create a RAGService with mocked LLM."""
        with patch('backend.services.rag_service.SentenceTransformerEmbeddings'):
            from backend.services.rag_service import RAGService
            service = RAGService(llm_service=mock_llm_service)
            return service
    
    def test_build_context_prompt(self, rag_service_with_mock_llm):
        """Test that context prompt is properly formatted."""
        from langchain_core.documents import Document
        
        chunks = [
            Document(
                page_content="This is the first chunk about AI.",
                metadata={"chunk_index": 0},
            ),
            Document(
                page_content="This is the second chunk about ML.",
                metadata={"chunk_index": 1},
            ),
        ]
        
        prompt = rag_service_with_mock_llm._build_context_prompt(
            query="What is AI?",
            chunks=chunks,
        )
        
        assert "CONTEXT FROM ARTICLE" in prompt
        assert "[Chunk 1]" in prompt
        assert "[Chunk 2]" in prompt
        assert "This is the first chunk" in prompt
        assert "USER QUESTION" in prompt
        assert "What is AI?" in prompt
    
    @pytest.mark.asyncio
    async def test_generate_answer_no_session(self, rag_service_with_mock_llm):
        """Test generating answer for non-existent session."""
        response = await rag_service_with_mock_llm.generate_answer(
            session_id="nonexistent",
            question="What is AI?",
        )
        
        # Should return a response indicating no context
        assert "I don't have enough information" in response.content


class TestRAGServiceSingleton:
    """Tests for singleton pattern."""
    
    def test_get_rag_service_creates_instance(self):
        """Test that get_rag_service creates an instance."""
        with patch('backend.services.rag_service.NanoGPTService'):
            with patch('backend.services.rag_service.SentenceTransformerEmbeddings'):
                # Reset singleton
                import backend.services.rag_service
                backend.services.rag_service._rag_service = None
                
                from backend.services.rag_service import get_rag_service
                service = get_rag_service()
                assert service is not None
    
    def test_get_rag_service_returns_same_instance(self):
        """Test that get_rag_service returns the same instance."""
        with patch('backend.services.rag_service.NanoGPTService'):
            with patch('backend.services.rag_service.SentenceTransformerEmbeddings'):
                from backend.services.rag_service import get_rag_service
                service1 = get_rag_service()
                service2 = get_rag_service()
                assert service1 is service2
