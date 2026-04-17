"""
AI News Companion - Translator Service

BM↔English translation using argostranslate with LLM quality check.
"""
import logging
from typing import Optional

import argostranslate.package
import argostranslate.translate

from backend.services.llm_service import NanoGPTService, LLMResponse
from backend.models.schemas import TranslateRequest, TranslateResponse

logger = logging.getLogger(__name__)


# Language configuration
LANGUAGE_NAMES = {
    "en": "English",
    "bm": "Bahasa Melayu",  # Internal code
    "ms": "Bahasa Melayu",  # ISO 639-1 code (used by argostranslate)
}

# Internal to argostranslate language code mapping
# Internal code 'bm' maps to argostranslate's 'ms' (Malay)
INTERNAL_TO_ARGOS = {
    "en": "en",
    "bm": "ms",  # Internal 'bm' -> argostranslate 'ms'
}

# argostranslate to internal language code mapping
ARGOS_TO_INTERNAL = {
    "en": "en",
    "ms": "bm",  # argostranslate 'ms' -> internal 'bm'
}

# Supported language pairs for argostranslate
SUPPORTED_PAIRS = [
    ("en", "ms"),  # English to Malay
    ("ms", "en"),  # Malay to English
]

# Minimal LLM quality check prompt
QUALITY_CHECK_PROMPT = """Check this translation for:
1. Meaning accuracy - does it convey the same meaning as the source?
2. Grammar - is it grammatically correct?
3. Missing words - are any words or concepts missing?

Source ({source_lang}): {source_text}
Translation ({target_lang}): {translated_text}

Respond with exactly "OK" if acceptable, or "ERROR: <brief description>" if issues found."""


class TranslatorService:
    """
    Service for BM↔English translation using argostranslate.
    
    Features:
    - Bidirectional translation (en↔bm)
    - Formal news style preservation
    - Prompt engineering for Malaysian news context
    - Same-language passthrough handling
    - Automatic model installation on startup
    """
    
    # Class-level flag to track if models have been initialized
    _models_initialized = False
    _initialization_lock = False
    
    def __init__(self, llm_service: Optional[NanoGPTService] = None):
        """
        Initialize the translator service.
        
        Args:
            llm_service: Optional NanoGPTService instance. If not provided,
                        a new instance will be created.
        """
        self._llm_service = llm_service or NanoGPTService()
        self._ensure_models_installed()
    
    @property
    def llm_service(self) -> NanoGPTService:
        """Get the LLM service instance."""
        return self._llm_service
    
    def _ensure_models_installed(self) -> None:
        """
        Ensure required translation models are installed.
        
        This method checks if the required language pair models (en↔ms) are
        installed and downloads them if necessary. Uses a class-level flag
        to avoid redundant checks.
        
        Raises:
            RuntimeError: If model installation fails
        """
        # Skip if already initialized (class-level check)
        if TranslatorService._models_initialized:
            logger.debug("Translation models already initialized")
            return
        
        # Prevent concurrent initialization
        if TranslatorService._initialization_lock:
            logger.warning("Model initialization already in progress, waiting...")
            import time
            while TranslatorService._initialization_lock:
                time.sleep(0.5)
            return
        
        TranslatorService._initialization_lock = True
        
        try:
            logger.info("Checking for required translation models...")
            
            # Get list of installed packages
            installed_packages = argostranslate.package.get_installed_packages()
            installed_codes = {pkg.from_code: pkg.to_code for pkg in installed_packages}
            
            # Check which models need to be installed
            models_to_install = []
            for from_code, to_code in SUPPORTED_PAIRS:
                if (from_code, to_code) not in [(p.from_code, p.to_code) for p in installed_packages]:
                    models_to_install.append((from_code, to_code))
            
            if models_to_install:
                logger.info(f"Installing {len(models_to_install)} translation model(s): {models_to_install}")
                
                # Update package list from remote
                argostranslate.package.update_package_index()
                
                # Find and install required packages
                for from_code, to_code in models_to_install:
                    try:
                        package_to_install = argostranslate.package.get_package_from_codes(
                            from_code, to_code
                        )
                        if package_to_install:
                            package_to_install.download()
                            package_to_install.install()
                            logger.info(f"Successfully installed {from_code}→{to_code} model")
                        else:
                            logger.error(f"Package not found for {from_code}→{to_code}")
                    except Exception as e:
                        logger.error(f"Failed to install {from_code}→{to_code} model: {e}")
                        raise RuntimeError(f"Failed to install translation model {from_code}→{to_code}: {e}")
            else:
                logger.info("All required translation models are already installed")
            
            TranslatorService._models_initialized = True
            logger.info("Translation models initialization complete")
            
        except Exception as e:
            logger.error(f"Error during model initialization: {e}")
            raise RuntimeError(f"Failed to initialize translation models: {e}")
        finally:
            TranslatorService._initialization_lock = False
    
    def _get_argos_lang_code(self, internal_code: str) -> str:
        """
        Convert internal language code to argostranslate code.
        
        Args:
            internal_code: Internal language code ('en' or 'bm')
            
        Returns:
            argostranslate language code ('en' or 'ms')
            
        Raises:
            ValueError: If invalid language code provided
        """
        if internal_code not in INTERNAL_TO_ARGOS:
            raise ValueError(f"Invalid language code: {internal_code}. Must be 'en' or 'bm'.")
        return INTERNAL_TO_ARGOS[internal_code]
    
    def _get_internal_lang_code(self, argos_code: str) -> str:
        """
        Convert argostranslate language code to internal code.
        
        Args:
            argos_code: argostranslate language code ('en' or 'ms')
            
        Returns:
            Internal language code ('en' or 'bm')
        """
        return ARGOS_TO_INTERNAL.get(argos_code, argos_code)
    
    async def translate(self, request: TranslateRequest) -> TranslateResponse:
        """
        Translate text between English and Bahasa Melayu using argostranslate.
        
        Args:
            request: TranslateRequest with text and language codes
            
        Returns:
            TranslateResponse with translated text
            
        Raises:
            ValueError: If invalid language codes provided
            RuntimeError: If translation fails
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
        
        # Convert to argostranslate codes
        source_lang_argos = self._get_argos_lang_code(request.source_lang)
        target_lang_argos = self._get_argos_lang_code(request.target_lang)
        
        logger.info(
            f"Translating from {LANGUAGE_NAMES[request.source_lang]} "
            f"to {LANGUAGE_NAMES[request.target_lang]} using argostranslate"
        )
        
        # Perform translation using argostranslate
        try:
            translated_text = argostranslate.translate.translate(
                request.text,
                source_lang_argos,
                target_lang_argos,
            )
            translated_text = translated_text.strip() if translated_text else request.text
            
            if not translated_text:
                logger.warning("argostranslate returned empty translation")
                translated_text = request.text  # Fallback to original
            
        except Exception as e:
            logger.error(f"argostranslate error: {e}")
            raise RuntimeError(f"Translation failed: {e}")
        
        # LLM quality check
        try:
            quality_prompt = QUALITY_CHECK_PROMPT.format(
                source_lang=LANGUAGE_NAMES[request.source_lang],
                target_lang=LANGUAGE_NAMES[request.target_lang],
                source_text=request.text,
                translated_text=translated_text,
            )
            
            quality_response: LLMResponse = await self._llm_service.complete(
                prompt=quality_prompt,
                temperature=0.1,  # Very low temperature for deterministic check
                max_tokens=50,  # Minimal tokens for speed
            )
            
            quality_result = quality_response.content.strip()
            
            if quality_result.startswith("ERROR:"):
                logger.warning(f"Translation quality issue detected: {quality_result}")
            elif quality_result != "OK":
                logger.warning(f"Unexpected quality check response: {quality_result}")
            else:
                logger.debug("Translation quality check passed")
                
        except Exception as e:
            logger.warning(f"LLM quality check failed: {e}")
            # Continue with translation even if quality check fails
        
        logger.info(
            f"Translation successful. Original length: {len(request.text)}, "
            f"Translated length: {len(translated_text)}"
        )
        
        return TranslateResponse(
            translated_text=translated_text,
            maintained_tone="news",
        )
    
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
