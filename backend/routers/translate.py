"""
AI News Companion - Translate Router

FastAPI router for the /api/translate endpoint.
Handles BM↔English translation with formal news style preservation.
"""
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException

from backend.models.schemas import TranslateRequest, TranslateResponse, ErrorResponse
from backend.services.translator import TranslatorService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["translate"])

# Singleton translator service instance
_translator_service: Optional[TranslatorService] = None


def get_translator_service() -> TranslatorService:
    """
    Get or create the TranslatorService singleton.
    
    Returns:
        TranslatorService instance
    """
    global _translator_service
    if _translator_service is None:
        _translator_service = TranslatorService()
    return _translator_service


@router.post(
    "/translate",
    response_model=TranslateResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Translation failed"},
    },
    summary="Translate text between English and Bahasa Melayu",
    description="Translate text with formal news style preservation. Supports en↔bm translation.",
)
async def translate_text(request: TranslateRequest) -> TranslateResponse:
    """
    Translate text between English and Bahasa Melayu.
    
    The translation maintains a formal news style appropriate for
    Malaysian news media. Named entities, technical terms, and
    journalistic tone are preserved.
    
    Args:
        request: TranslateRequest with:
            - text: Text to translate
            - source_lang: Source language ('en' or 'bm')
            - target_lang: Target language ('en' or 'bm')
    
    Returns:
        TranslateResponse with:
            - translated_text: The translated text
            - maintained_tone: Always 'news' indicating news style was preserved
    
    Raises:
        HTTPException: 400 if language codes are invalid
        HTTPException: 500 if translation fails
    """
    logger.info(
        f"Received translation request: {request.source_lang} -> {request.target_lang} "
        f"({len(request.text)} characters)"
    )
    
    # Validate language codes
    valid_langs = {"en", "bm"}
    if request.source_lang not in valid_langs:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid source_lang: '{request.source_lang}'. Must be 'en' or 'bm'.",
        )
    if request.target_lang not in valid_langs:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid target_lang: '{request.target_lang}'. Must be 'en' or 'bm'.",
        )
    
    try:
        translator = get_translator_service()
        response = await translator.translate(request)
        
        logger.info(
            f"Translation successful: {request.source_lang} -> {request.target_lang} "
            f"({len(request.text)} -> {len(response.translated_text)} characters)"
        )
        
        return response
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )
    except Exception as e:
        logger.exception(f"Translation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Translation failed: {str(e)}",
        )
