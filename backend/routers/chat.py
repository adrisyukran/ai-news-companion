"""
AI News Companion - Chat Router

RAG-based chat endpoint for asking questions about articles.
"""
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Body

from backend.models.schemas import ChatRequest, ChatResponse
from backend.services.rag_service import RAGService, get_rag_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Chat endpoint for RAG-based question answering about articles.
    
    The session must have an associated article loaded first.
    Use this endpoint to ask questions about the article content.
    
    Args:
        request: ChatRequest with session_id and question
        
    Returns:
        ChatResponse with answer based on retrieved context
        
    Raises:
        HTTPException: If session not found or other errors occur
    """
    try:
        rag_service = get_rag_service()
        
        # Check if session exists
        if not rag_service.session_exists(request.session_id):
            raise HTTPException(
                status_code=404,
                detail=f"Session {request.session_id} not found. Please load an article first."
            )
        
        # Generate answer
        result = await rag_service.chat(
            session_id=request.session_id,
            question=request.question,
        )
        
        return ChatResponse(
            answer=result["answer"],
            session_id=request.session_id,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error generating answer: {str(e)}"
        )


@router.post("/chat/load")
async def load_article(
    session_id: Optional[str] = Body(default=None),
    text: str = Body(..., description="Article text content"),
    source_type: str = Body(default="text", description="Source type: url, file, or text"),
    source_value: str = Body(default="inline", description="Source value or URL/path marker"),
) -> dict:
    """
    Load article content into a chat session.
    
    This endpoint creates or reuses a session and stores the article
    for RAG-based question answering.
    
    Args:
        session_id: Optional existing session ID to reuse
        text: Article text content
        source_type: Type of source ('url', 'file', 'text')
        source_value: Source URL, path, or 'inline'
        
    Returns:
        Dict with session_id
    """
    try:
        rag_service = get_rag_service()
        
        # Create session with article
        new_session_id = rag_service.create_session(
            text=text,
            source_type=source_type,
            source_value=source_value,
            session_id=session_id,
        )
        
        logger.info(f"Loaded article into session {new_session_id}")
        
        return {
            "session_id": new_session_id,
            "status": "loaded",
        }
        
    except Exception as e:
        logger.error(f"Error loading article: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error loading article: {str(e)}"
        )


@router.delete("/chat/session/{session_id}")
async def delete_session(session_id: str) -> dict:
    """
    Delete a chat session and its associated data.
    
    Args:
        session_id: Session identifier to delete
        
    Returns:
        Dict with deletion status
    """
    try:
        rag_service = get_rag_service()
        
        if rag_service.delete_session(session_id):
            return {
                "session_id": session_id,
                "status": "deleted",
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting session: {str(e)}"
        )


@router.post("/chat/session/{session_id}/clear")
async def clear_session(session_id: str) -> dict:
    """
    Clear all documents from a session's vector store without deleting the session.
    This is useful for preventing context leakage when loading new content.
    
    Args:
        session_id: Session identifier to clear
        
    Returns:
        Dict with clear status
    """
    try:
        rag_service = get_rag_service()
        
        if rag_service.clear_session(session_id):
            return {
                "session_id": session_id,
                "status": "cleared",
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing session: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing session: {str(e)}"
        )


@router.get("/chat/session/{session_id}/exists")
async def check_session(session_id: str) -> dict:
    """
    Check if a session exists.
    
    Args:
        session_id: Session identifier to check
        
    Returns:
        Dict with exists status
    """
    rag_service = get_rag_service()
    return {
        "session_id": session_id,
        "exists": rag_service.session_exists(session_id),
    }
