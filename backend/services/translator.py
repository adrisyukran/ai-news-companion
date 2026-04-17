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

# LLM refinement prompt for high-quality translation
REFINEMENT_PROMPT = """You are a professional translator refining a machine translation.

Your task:
1. Fix any spelling errors, grammatical mistakes, and unnatural phrasing
2. Ensure the translation accurately conveys the meaning of the source text
3. Maintain a formal, professional tone suitable for news content
{dbp_instruction}
Output ONLY the refined translation. Do not add any explanations or notes.

Source ({source_lang}): {source_text}
Base Translation ({target_lang}): {base_translation}

Refined Translation ({target_lang}):"""

# DBP-specific instruction for Bahasa Melayu
DBP_INSTRUCTION = """4. For Bahasa Melayu, strictly follow Dewan Bahasa dan Pustaka (DBP) standards:
   - Use formal, standard Malay (not colloquial or Indonesian)
   - Use correct DBP terminology (e.g., "kerajaan" not "pemerintahan" for government)
   - Ensure proper Malay sentence structure and grammar
   - Avoid Indonesian loanwords or spellings"""


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
                        # Manually search through available packages since get_package_from_codes doesn't exist
                        available_packages = argostranslate.package.get_available_packages()
                        package_to_install = None
                        for pkg in available_packages:
                            if pkg.from_code == from_code and pkg.to_code == to_code:
                                package_to_install = pkg
                                break
                        
                        if package_to_install:
                            package_to_install.download()
                            package_to_install.install()
                            logger.info(f"Successfully installed {from_code}→{to_code} model")
                        else:
                            logger.error(f"Package not found for {from_code}→{to_code}")
                            raise RuntimeError(f"Package not found for {from_code}→{to_code}")
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
    
    def _detect_language(self, text: str) -> str:
        """
        Detect if text is English or Bahasa Melayu using heuristic analysis.
        
        Uses common word detection to determine the source language.
        Since we only support EN and BM, this simple heuristic is efficient and reliable.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Detected language code ('en' or 'bm')
        """
        # Common Bahasa Melayu words (including DBP-standard terms)
        bm_common_words = {
            # Articles, prepositions, conjunctions
            'yang', 'dan', 'atau', 'dengan', 'untuk', 'pada', 'dalam', 'ini', 'itu',
            'ada', 'tidak', 'bukan', 'akan', 'telah', 'sudah', 'belum', 'sangat',
            'lebih', 'paling', 'saja', 'juga', 'saja', 'kerana', 'kerana', 'supaya',
            'agar', 'walaupun', 'meskipun', 'jika', 'kalau', 'apabila', 'sejak',
            'hingga', 'sampai', 'dari', 'ke', 'di', 'kepada', 'daripada',
            # Common nouns
            'kerajaan', 'rakyat', 'negara', 'menteri', 'parlimen', 'undang-undang',
            'polis', 'sekolah', 'universiti', 'hospital', 'jalan', 'kampung',
            'bandar', 'negeri', 'wilayah', 'daerah', 'kawasan', 'tempat', 'orang',
            'hari', 'bulan', 'tahun', 'minggu', 'pagi', 'tengah', 'malam',
            # Common verbs
            'adalah', 'merupakan', 'menjadi', 'berada', 'terdapat', 'mempunyai',
            'memiliki', 'mengambil', 'memberi', 'membuat', 'melakukan', 'berkata',
            'mengatakan', 'menyatakan', 'menjelaskan', 'menambah', 'turut', 'turut',
            # Pronouns
            'saya', 'awak', 'anda', 'kita', 'kami', 'mereka', 'dia', 'beliau',
            'ia', 'ini', 'itu', 'siapa', 'apa', 'mana', 'bila', 'di mana',
            # Formal/DBP-specific terms
            'perdana', 'timbalan', 'ahli', 'jawatankuasa', 'mesyuarat', 'sidang',
            'rang', 'undang', 'dasar', 'polisi', 'ekonomi', 'sosial', 'pembangunan',
            'nasional', 'antarabangsa', 'serantau', 'tempatan', 'tempatan',
        }
        
        # Common English words
        en_common_words = {
            # Articles, prepositions, conjunctions
            'the', 'a', 'an', 'and', 'or', 'but', 'with', 'for', 'on', 'in', 'to',
            'of', 'at', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
            'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them',
            # Common nouns
            'government', 'people', 'country', 'minister', 'parliament', 'law',
            'police', 'school', 'university', 'hospital', 'road', 'village',
            'city', 'state', 'region', 'area', 'place', 'person', 'day',
            'month', 'year', 'week', 'morning', 'afternoon', 'evening', 'night',
            # Common verbs
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
            'had', 'having', 'do', 'does', 'did', 'doing', 'make', 'made',
            'take', 'took', 'give', 'gave', 'get', 'got', 'say', 'said',
            'tell', 'told', 'ask', 'asked', 'use', 'used', 'find', 'found',
            # Pronouns
            'i', 'you', 'we', 'they', 'he', 'she', 'it', 'who', 'what', 'which',
            'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both',
            'few', 'more', 'most', 'other', 'some', 'such', 'no', 'not',
            'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
        }
        
        # Normalize text to lowercase and split into words
        text_lower = text.lower()
        words = set(text_lower.split())
        
        # Count matches for each language
        bm_matches = len(words & bm_common_words)
        en_matches = len(words & en_common_words)
        
        # Debug logging for detection
        logger.debug(f"Language detection: BM matches={bm_matches}, EN matches={en_matches}")
        
        # Determine language based on matches
        if bm_matches > en_matches:
            return 'bm'
        elif en_matches > bm_matches:
            return 'en'
        else:
            # If no clear match, use LLM-based detection as fallback
            logger.debug("Heuristic detection inconclusive, using LLM fallback")
            return self._detect_language_llm(text)
    
    async def _detect_language_llm(self, text: str) -> str:
        """
        Fallback language detection using LLM when heuristic is inconclusive.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Detected language code ('en' or 'bm')
        """
        detection_prompt = f"""Determine if the following text is English or Bahasa Melayu.
Respond with ONLY "en" for English or "bm" for Bahasa Melayu.

Text: {text[:500]}

Language:"""
        
        try:
            response: LLMResponse = await self._llm_service.complete(
                prompt=detection_prompt,
                temperature=0.0,  # Deterministic for classification
                max_tokens=5,
            )
            
            detected_lang = response.content.strip().lower()
            
            # Validate response
            if detected_lang in ('en', 'english'):
                return 'en'
            elif detected_lang in ('bm', 'ms', 'bahasa', 'melayu', 'malay'):
                return 'bm'
            else:
                logger.warning(f"Unexpected LLM language detection response: {detected_lang}, defaulting to 'en'")
                return 'en'
                
        except Exception as e:
            logger.error(f"LLM language detection failed: {e}, defaulting to 'en'")
            return 'en'  # Default to English on error
    
    async def _refine_translation(self, source_text: str, base_translation: str,
                                   source_lang: str, target_lang: str) -> str:
        """
        Refine a base translation using LLM for higher quality.
        
        Args:
            source_text: Original source text
            base_translation: Base translation from argostranslate
            source_lang: Source language code ('en' or 'bm')
            target_lang: Target language code ('en' or 'bm')
            
        Returns:
            Refined translation
        """
        # Add DBP instruction if target is Bahasa Melayu
        dbp_instruction = ""
        if target_lang == 'bm':
            dbp_instruction = DBP_INSTRUCTION + "\n"
        
        # Build refinement prompt
        prompt = REFINEMENT_PROMPT.format(
            source_lang=LANGUAGE_NAMES[source_lang],
            target_lang=LANGUAGE_NAMES[target_lang],
            source_text=source_text,
            base_translation=base_translation,
            dbp_instruction=dbp_instruction,
        )
        
        try:
            response: LLMResponse = await self._llm_service.complete(
                prompt=prompt,
                temperature=0.3,  # Low temperature for focused refinement
                max_tokens=max(100, int(len(base_translation) * 1.2)),  # Allow some expansion
            )
            
            refined_text = response.content.strip()
            
            # Validate that refinement produced output
            if not refined_text:
                logger.warning("LLM refinement returned empty, using base translation")
                return base_translation
            
            logger.debug(f"Translation refined: {len(base_translation)} → {len(refined_text)} chars")
            return refined_text
            
        except Exception as e:
            logger.warning(f"LLM refinement failed: {e}, using base translation")
            return base_translation  # Fallback to base translation
    
    async def translate(self, request: TranslateRequest) -> TranslateResponse:
        """
        Translate text between English and Bahasa Melayu with auto-detection and LLM refinement.
        
        Features:
        - Auto-detection of source language (if not provided)
        - Automatic target language selection (opposite of detected/provided source)
        - Base translation via argostranslate
        - LLM refinement for high-quality output with DBP standards for BM
        
        Args:
            request: TranslateRequest with text and optional language codes
            
        Returns:
            TranslateResponse with refined translated text
            
        Raises:
            ValueError: If invalid language codes provided or language detection fails
            RuntimeError: If translation fails
        """
        valid_langs = {"en", "bm"}
        
        # Step 1: Auto-detect source language if not provided
        source_lang = request.source_lang
        target_lang = request.target_lang
        
        if not source_lang:
            logger.info("Auto-detecting source language")
            source_lang = self._detect_language(request.text)
            logger.info(f"Detected source language: {LANGUAGE_NAMES[source_lang]}")
        
        if not target_lang:
            # Auto-select opposite language as target
            target_lang = 'bm' if source_lang == 'en' else 'en'
            logger.info(f"Auto-selected target language: {LANGUAGE_NAMES[target_lang]}")
        
        # Validate language codes
        if source_lang not in valid_langs:
            raise ValueError(f"Invalid source_lang: {source_lang}. Must be 'en' or 'bm'.")
        if target_lang not in valid_langs:
            raise ValueError(f"Invalid target_lang: {target_lang}. Must be 'en' or 'bm'.")
        
        # Handle same language translation (passthrough)
        if source_lang == target_lang:
            logger.info(
                f"Source and target language are the same ({source_lang}). "
                f"Returning text with maintained tone."
            )
            return TranslateResponse(
                translated_text=request.text,
                maintained_tone="news",
            )
        
        # Step 2: Convert to argostranslate codes
        source_lang_argos = self._get_argos_lang_code(source_lang)
        target_lang_argos = self._get_argos_lang_code(target_lang)
        
        logger.info(
            f"Translating from {LANGUAGE_NAMES[source_lang]} "
            f"to {LANGUAGE_NAMES[target_lang]} using argostranslate + LLM refinement"
        )
        
        # Step 3: Get base translation from argostranslate
        try:
            base_translation = argostranslate.translate.translate(
                request.text,
                source_lang_argos,
                target_lang_argos,
            )
            base_translation = base_translation.strip() if base_translation else request.text
            
            if not base_translation:
                logger.warning("argostranslate returned empty translation")
                base_translation = request.text  # Fallback to original
            
        except Exception as e:
            logger.error(f"argostranslate error: {e}")
            raise RuntimeError(f"Translation failed: {e}")
        
        # Step 4: Refine translation using LLM
        logger.debug("Starting LLM refinement process")
        refined_translation = await self._refine_translation(
            source_text=request.text,
            base_translation=base_translation,
            source_lang=source_lang,
            target_lang=target_lang,
        )
        
        logger.info(
            f"Translation complete. Original length: {len(request.text)}, "
            f"Base length: {len(base_translation)}, "
            f"Refined length: {len(refined_translation)}"
        )
        
        return TranslateResponse(
            translated_text=refined_translation,
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
