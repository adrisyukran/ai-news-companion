"""
AI News Companion - Translator Service

BM↔English translation with formal news style preservation.
"""
import logging
from typing import Optional

from backend.services.llm_service import NanoGPTService, LLMResponse
from backend.models.schemas import TranslateRequest, TranslateResponse

logger = logging.getLogger(__name__)


# Language configuration
LANGUAGE_NAMES = {
    "en": "English",
    "bm": "Bahasa Melayu",
}

# System prompt for translation with news style preservation
TRANSLATION_SYSTEM_PROMPT = """You are a professional translator specializing in Malaysian news media translation.

Your task is to translate text between English and Bahasa Melayu (BM) with EXACT preservation of:
1. FORMAL NEWS STYLE: Maintain the professional, objective, and authoritative tone of news articles
2. ACCURACY: Preserve the original meaning without omission or addition
3. NAMED ENTITIES: Keep names, locations, organizations exactly as provided
4. TECHNICAL TERMS: Use standard Malaysian news terminology
5. DATE/NUMBER FORMATS: Keep original formats unless BM convention requires otherwise

For BM translation specifically:
- Use formal Bahasa Melayu appropriate for Malaysian news broadcasts/publications
- Avoid colloquialisms, slang, or Manglish
- Use proper Malaysian news conventions (e.g., "Yang di-Pertuan Agong", "Parlimen")
- Maintain journalistic objectivity and neutrality

For English translation:
- Use formal English appropriate for international news outlets
- Avoid British/American-specific idioms if they don't translate naturally
- Keep translation natural and readable

Output ONLY the translated text, nothing else."""


class TranslatorService:
    """
    Service for BM↔English translation with news style preservation.
    
    Features:
    - Bidirectional translation (en↔bm)
    - Formal news style preservation
    - Prompt engineering for Malaysian news context
    - Same-language passthrough handling
    """
    
    def __init__(self, llm_service: Optional[NanoGPTService] = None):
        """
        Initialize the translator service.
        
        Args:
            llm_service: Optional NanoGPTService instance. If not provided,
                        a new instance will be created.
        """
        self._llm_service = llm_service or NanoGPTService()
    
    @property
    def llm_service(self) -> NanoGPTService:
        """Get the LLM service instance."""
        return self._llm_service
    
    def _build_translation_prompt(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        """
        Build the translation prompt with proper context.
        
        Args:
            text: Text to translate
            source_lang: Source language code (en or bm)
            target_lang: Target language code (en or bm)
            
        Returns:
            Formatted translation prompt
        """
        source_name = LANGUAGE_NAMES.get(source_lang, source_lang)
        target_name = LANGUAGE_NAMES.get(target_lang, target_lang)
        
        prompt = f"""Translate the following {source_name} text to {target_name}.

IMPORTANT: This is news article content. Preserve the formal, professional news style.

Source Text ({source_name}):
---
{text}
---

Translation ({target_name}):
"""
        return prompt
    
    async def translate(self, request: TranslateRequest) -> TranslateResponse:
        """
        Translate text between English and Bahasa Melayu.
        
        Args:
            request: TranslateRequest with text and language codes
            
        Returns:
            TranslateResponse with translated text
            
        Raises:
            ValueError: If invalid language codes provided
        """
        # Validate language codes
        valid_langs = {"en", "bm"}
        if request.source_lang not in valid_langs:
            raise ValueError(f"Invalid source_lang: {request.source_lang}. Must be 'en' or 'bm'.")
        if request.target_lang not in valid_langs:
            raise ValueError(f"Invalid target_lang: {request.target_lang}. Must be 'en' or 'bm'.")
        
        # Handle same language translation (passthrough)
        if request.source_lang == request.target_lang:
            logger.info(
                f"Source and target language are the same ({request.source_lang}). "
                f"Returning text with maintained tone."
            )
            return TranslateResponse(
                translated_text=request.text,
                maintained_tone="news",
            )
        
        # Build translation prompt
        prompt = self._build_translation_prompt(
            text=request.text,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
        )
        
        logger.info(
            f"Translating from {LANGUAGE_NAMES[request.source_lang]} "
            f"to {LANGUAGE_NAMES[request.target_lang]}"
        )
        
        # Call LLM service
        try:
            response: LLMResponse = await self._llm_service.complete(
                prompt=prompt,
                system_prompt=TRANSLATION_SYSTEM_PROMPT,
                temperature=0.3,  # Lower temperature for consistency
                max_tokens=4096,
            )
            
            translated_text = response.content.strip()
            
            logger.info(
                f"Translation successful. Original length: {len(request.text)}, "
                f"Translated length: {len(translated_text)}"
            )
            
            return TranslateResponse(
                translated_text=translated_text,
                maintained_tone="news",
            )
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            raise
    
    async def translate_en_to_bm(self, text: str) -> TranslateResponse:
        """
        Convenience method for English to Bahasa Melayu translation.
        
        Args:
            text: English text to translate
            
        Returns:
            TranslateResponse with BM translation
        """
        request = TranslateRequest(
            text=text,
            source_lang="en",
            target_lang="bm",
        )
        return await self.translate(request)
    
    async def translate_bm_to_en(self, text: str) -> TranslateResponse:
        """
        Convenience method for Bahasa Melayu to English translation.
        
        Args:
            text: Bahasa Melayu text to translate
            
        Returns:
            TranslateResponse with English translation
        """
        request = TranslateRequest(
            text=text,
            source_lang="bm",
            target_lang="en",
        )
        return await self.translate(request)
    
    async def close(self):
        """Close the underlying LLM service."""
        await self._llm_service.close()
