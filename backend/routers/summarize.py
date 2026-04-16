"""
AI News Companion - Summarize Router

FastAPI router for the /api/summarize endpoint.
Handles article summarization requests via URL, file upload, or plain text.
"""
import logging
from typing import Optional
from pathlib import Path
import tempfile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from backend.models.schemas import SummarizeRequest, SummarizeResponse, ErrorResponse
from backend.services.summarizer import SummarizerService, SummarizerError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["summarize"])

# Singleton summarizer service instance
_summarizer_service: Optional[SummarizerService] = None


def get_summarizer_service() -> SummarizerService:
    """
    Get or create the SummarizerService singleton.
    
    Returns:
        SummarizerService instance
    """
    global _summarizer_service
    if _summarizer_service is None:
        _summarizer_service = SummarizerService()
    return _summarizer_service


@router.post(
    "/summarize",
    response_model=SummarizeResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Summarization failed"},
    },
    summary="Summarize an article",
    description="Summarize an article from URL, file upload, or plain text. Returns three summary types.",
)
async def summarize_article(
    url: Optional[str] = Form(
        default=None,
        description="URL of the article to summarize",
        examples=["https://example.com/news/article"],
    ),
    file: Optional[UploadFile] = Form(
        default=None,
        description="File to summarize (PDF, DOCX, or TXT)",
    ),
    text: Optional[str] = Form(
        default=None,
        description="Plain text content to summarize",
    ),
) -> SummarizeResponse:
    """
    Summarize an article and return three types of summaries.
    
    Accepts input via:
    - URL: Provide a web article URL
    - File: Upload a PDF, DOCX, or TXT file
    - Text: Provide plain text directly
    
    Returns:
        SummarizeResponse with short_summary, medium_summary, and headline
    """
    logger.info("Received summarization request")
    
    # Validate that at least one input is provided
    if not any([url, file, text]):
        raise HTTPException(
            status_code=400,
            detail="At least one of 'url', 'file', or 'text' must be provided",
        )
    
    # Validate mutual exclusivity (at most one input should be provided)
    provided_inputs = sum(1 for x in [url, file, text] if x)
    if provided_inputs > 1:
        raise HTTPException(
            status_code=400,
            detail="Only one of 'url', 'file', or 'text' should be provided",
        )
    
    try:
        summarizer = get_summarizer_service()
        
        # Process based on input type
        if url:
            logger.info(f"Summarizing URL: {url}")
            short_summary, medium_summary, headline = await summarizer.summarize_url(url)
        elif file:
            # Handle file upload
            logger.info(f"Summarizing uploaded file: {file.filename}")
            short_summary, medium_summary, headline = await _summarize_file(summarizer, file)
        elif text:
            logger.info(f"Summarizing text input ({len(text)} characters)")
            short_summary, medium_summary, headline = await summarizer.summarize_text(text)
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid input type",
            )
        
        return SummarizeResponse(
            short_summary=short_summary,
            medium_summary=medium_summary,
            headline=headline,
        )
        
    except SummarizerError as e:
        logger.error(f"Summarization error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Summarization failed: {str(e)}",
        )
    except HTTPException:
        # Re-raise HTTPException (validation errors, etc.)
        raise
    except Exception as e:
        logger.exception(f"Unexpected error during summarization: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}",
        )


async def _summarize_file(
    summarizer: SummarizerService,
    file: UploadFile,
) -> tuple:
    """
    Process an uploaded file and generate summaries.
    
    Args:
        summarizer: SummarizerService instance
        file: Uploaded file
        
    Returns:
        Tuple of (short_summary, medium_summary, headline)
        
    Raises:
        HTTPException: If file processing fails
    """
    # Validate file extension
    allowed_extensions = {'.pdf', '.docx', '.doc', '.txt'}
    file_ext = Path(file.filename).suffix.lower() if file.filename else ''
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Supported types: PDF, DOCX, DOC, TXT",
        )
    
    # Create a temporary file to store the uploaded content
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=file_ext,
            delete=False,
        ) as temp_file:
            temp_path = temp_file.name
            
            # Write uploaded content to temp file
            content = await file.read()
            temp_file.write(content)
        # File is now closed after exiting the 'with' block
        
        # Generate summaries
        return await summarizer.summarize_file(temp_path)
        
    finally:
        # Clean up temp file
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except PermissionError:
                # On Windows, file might still be held - ignore
                logger.warning(f"Could not delete temp file: {temp_path}")


@router.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint for the summarization service."""
    return {"status": "ok", "service": "summarize"}
