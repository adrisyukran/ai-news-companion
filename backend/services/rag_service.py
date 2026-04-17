"""
AI News Companion - RAG Service

Implements Retrieval-Augmented Generation pipeline for chat about articles.
Includes text chunking, embeddings, vector store, and generation chain.
"""
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from backend.config import (
    CHUNK_SIZE_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    EMBEDDING_MODEL,
)
from backend.services.llm_service import NanoGPTService, LLMResponse

logger = logging.getLogger(__name__)


@dataclass
class ChunkMetadata:
    """Metadata for a text chunk."""
    chunk_index: int
    total_chunks: int
    session_id: str
    source_type: str  # 'url', 'file', 'text'
    source_value: str


@dataclass
class SessionContext:
    """Context for a chat session containing article data."""
    session_id: str
    source_type: str
    source_value: str
    original_text: str
    chunks: List[Document] = field(default_factory=list)
    vector_store: Optional[Chroma] = None
    created_at: float = field(default_factory=lambda: 0.0)


class RAGError(Exception):
    """Exception raised when RAG operations fail."""
    pass


class RAGService:
    """
    RAG Service for article-based chat.
    
    Features:
    - Text chunking with overlap
    - Embeddings via sentence-transformers
    - In-memory Chroma vector store
    - Retrieval-augmented generation
    - Session management
    """
    
    # System prompt for hallucination prevention
    SYSTEM_PROMPT = """You are a helpful assistant that answers questions based ONLY on the provided article context.
You must follow these rules STRICTLY:
1. ONLY use information from the provided context to answer questions
2. If the answer is not in the context, say "I don't have enough information in the article to answer this question."
3. NEVER make up or invent information not present in the context
4. Quote relevant parts from the context when possible
5. Be concise but comprehensive in your answers

Context from the article will be provided below. Answer the question based ONLY on this context."""

    
    def __init__(
        self,
        llm_service: Optional[NanoGPTService] = None,
        embedding_model: str = EMBEDDING_MODEL,
        chunk_size: int = CHUNK_SIZE_TOKENS,
        chunk_overlap: int = CHUNK_OVERLAP_TOKENS,
        top_k: int = 5,
    ):
        """
        Initialize RAG Service.
        
        Args:
            llm_service: LLM service for generation (NanoGPTService)
            embedding_model: Model name for embeddings
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            top_k: Number of top chunks to retrieve
        """
        self.llm_service = llm_service or NanoGPTService()
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        
        # Initialize text splitter
        # Use characters as proxy for tokens (roughly 4 chars per token)
        chars_per_token = 4
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size * chars_per_token,
            chunk_overlap=chunk_overlap * chars_per_token,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        
        # Initialize embeddings
        self._embeddings = SentenceTransformerEmbeddings(
            model_name=embedding_model,
        )
        
        # Session storage (in-memory)
        # Maps session_id -> SessionContext
        self._sessions: Dict[str, SessionContext] = {}
        
        # Track collection versions for session isolation
        # Maps session_id -> collection_version (int)
        self._collection_versions: Dict[str, int] = {}
        
        logger.info(f"Initialized RAGService with embedding model: {embedding_model}")
    
    def _create_session_id(self) -> str:
        """Generate a new session ID."""
        return str(uuid.uuid4())
    
    def _create_documents(
        self,
        text: str,
        session_id: str,
        source_type: str,
        source_value: str,
    ) -> List[Document]:
        """
        Split text into LangChain Documents with metadata.
        
        Args:
            text: Text to split
            session_id: Session identifier
            source_type: Type of source ('url', 'file', 'text')
            source_value: Value of source (URL, path, or inline text marker)
            
        Returns:
            List of LangChain Document objects
        """
        # Split text into chunks
        raw_chunks = self._text_splitter.split_text(text)
        
        documents = []
        for i, chunk in enumerate(raw_chunks):
            metadata = {
                "chunk_index": i,
                "total_chunks": len(raw_chunks),
                "session_id": session_id,
                "source_type": source_type,
                "source_value": source_value,
            }
            doc = Document(page_content=chunk, metadata=metadata)
            documents.append(doc)
        
        logger.info(f"Split text into {len(documents)} chunks for session {session_id}")
        return documents
    
    def create_session(
        self,
        text: str,
        source_type: str = "text",
        source_value: str = "inline",
        session_id: Optional[str] = None,
    ) -> str:
        """
        Create a new session with article content.
        
        When reusing a session ID, uses a versioned collection name to prevent
        any context leakage between articles. Each time a session is recreated
        with the same ID, a new unique collection is created.
        
        Args:
            text: Article text content
            source_type: Type of source ('url', 'file', 'text')
            source_value: Value of source
            session_id: Optional existing session ID
            
        Returns:
            Session ID
        """
        session_id = session_id or self._create_session_id()
        
        # Increment collection version to ensure unique collection name
        # This prevents context leakage when reusing session IDs
        if session_id in self._collection_versions:
            self._collection_versions[session_id] += 1
            logger.info(f"Session {session_id} reused, incrementing collection version to {self._collection_versions[session_id]}")
        else:
            self._collection_versions[session_id] = 0
        
        # Create documents from text
        documents = self._create_documents(
            text=text,
            session_id=session_id,
            source_type=source_type,
            source_value=source_value,
        )
        
        try:
            # Create vector store (in-memory) with a versioned collection name
            # Format: session_{session_id}_v{version}
            # This ensures complete isolation between session recreations
            collection_version = self._collection_versions[session_id]
            collection_name = f"session_{session_id}_v{collection_version}"
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self._embeddings,
                collection_name=collection_name,
            )
            
            # Verify vector store was created properly
            if vector_store is None:
                raise RAGError("Failed to create vector store: Chroma returned None")
            
        except TypeError as e:
            # Handle cases where Chroma or embeddings return unexpected types
            logger.error(f"TypeError creating vector store: {e}")
            raise RAGError(f"Vector store creation failed: {str(e)}")
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise RAGError(f"Failed to create vector store: {str(e)}")
        
        # Store session context
        self._sessions[session_id] = SessionContext(
            session_id=session_id,
            source_type=source_type,
            source_value=source_value,
            original_text=text,
            chunks=documents,
            vector_store=vector_store,
        )
        
        logger.info(f"Created session {session_id} with {len(documents)} chunks")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[SessionContext]:
        """
        Get session context by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            SessionContext or None if not found
        """
        return self._sessions.get(session_id)
    
    def session_exists(self, session_id: str) -> bool:
        """
        Check if a session exists.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session exists
        """
        return session_id in self._sessions
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and its associated data.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session was deleted
        """
        if session_id in self._sessions:
            # Clean up vector store collection if it exists
            session = self._sessions[session_id]
            if session.vector_store:
                try:
                    collection_name = session.vector_store._collection.name
                    session.vector_store._client.delete_collection(name=collection_name)
                    logger.info(f"Deleted collection {collection_name} for session {session_id}")
                except Exception as e:
                    logger.error(f"Error deleting collection for session {session_id}: {e}")
            
            del self._sessions[session_id]
            # Also clean up version tracking
            if session_id in self._collection_versions:
                del self._collection_versions[session_id]
            logger.info(f"Deleted session {session_id}")
            return True
        return False
    
    def clear_session(self, session_id: str) -> bool:
        """
        Clear all documents from the vector store for a given session.
        This ensures no context leakage when reusing a session ID or loading new content.
        
        Since each session has its own isolated collection named 'session_{session_id}',
        we delete the entire collection to ensure complete isolation.
        
        Args:
            session_id: Session identifier to clear
            
        Returns:
            True if session was cleared successfully
        """
        if session_id not in self._sessions:
            logger.warning(f"Session {session_id} not found, nothing to clear")
            return False
        
        session = self._sessions[session_id]
        
        # Delete the entire collection for this session
        if session.vector_store:
            try:
                collection_name = session.vector_store._collection.name
                # Delete the entire collection
                session.vector_store._client.delete_collection(name=collection_name)
                logger.info(f"Deleted collection {collection_name} for session {session_id}")
            except Exception as e:
                logger.error(f"Error deleting collection for session {session_id}: {e}")
                return False
        
        # Clear the chunks list
        session.chunks = []
        
        logger.info(f"Cleared session {session_id} documents")
        return True
    
    def retrieve_relevant_chunks(
        self,
        session_id: str,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Document]:
        """
        Retrieve top-k most relevant chunks for a query.
        
        Args:
            session_id: Session identifier
            query: Question/query to find relevant chunks for
            top_k: Number of chunks to retrieve (default: self.top_k)
            
        Returns:
            List of relevant Documents
        """
        session = self.get_session(session_id)
        if not session or not session.vector_store:
            logger.warning(f"Session {session_id} not found or has no vector store")
            return []
        
        top_k = top_k or self.top_k
        
        # Similarity search
        docs = session.vector_store.similarity_search(query, k=top_k)
        
        logger.debug(f"Retrieved {len(docs)} relevant chunks for session {session_id}")
        return docs
    
    def _build_context_prompt(self, query: str, chunks: List[Document]) -> str:
        """
        Build context prompt with retrieved chunks.
        
        Args:
            query: User's question
            chunks: Relevant document chunks
            
        Returns:
            Formatted prompt with context
        """
        # Format chunks into context string
        context_parts = []
        for i, chunk in enumerate(chunks):
            context_parts.append(chunk.page_content)
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""CONTEXT FROM ARTICLE:
{context}

---

USER QUESTION: {query}

---

Based ONLY on the context provided above, answer the user's question. Follow the rules in your system prompt."""
        
        return prompt
    
    async def generate_answer(
        self,
        session_id: str,
        question: str,
    ) -> LLMResponse:
        """
        Generate answer using RAG pipeline.
        
        Args:
            session_id: Session identifier
            question: User's question
            
        Returns:
            LLMResponse with generated answer
        """
        # Retrieve relevant chunks
        chunks = self.retrieve_relevant_chunks(session_id, question)
        
        if not chunks:
            # Return a response indicating no context found
            return LLMResponse(
                content="I don't have enough information in the article to answer this question. Please ensure the article content has been loaded into this session.",
                model=self.llm_service.model,
            )
        
        # Build prompt with context
        prompt = self._build_context_prompt(question, chunks)
        
        # Generate answer
        response = await self.llm_service.complete(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.3,  # Lower temperature for factual answers
            max_tokens=1000,
        )
        
        logger.info(f"Generated answer for session {session_id}")
        return response
    
    async def chat(
        self,
        session_id: str,
        question: str,
    ) -> Dict[str, Any]:
        """
        Main chat method - retrieve and generate in one call.
        
        Args:
            session_id: Session identifier
            question: User's question
            
        Returns:
            Dict with 'answer' and 'session_id'
        """
        # Check if session exists
        if not self.session_exists(session_id):
            return {
                "answer": f"Session {session_id} not found. Please load an article first.",
                "session_id": session_id,
            }
        
        # Generate answer
        response = await self.generate_answer(session_id, question)
        
        return {
            "answer": response.content,
            "session_id": session_id,
        }


# Singleton instance for shared use
_rag_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """
    Get or create singleton RAGService instance.
    
    Returns:
        RAGService instance
    """
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service
