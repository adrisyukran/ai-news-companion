"""
AI News Companion - Configuration Settings

nano-gpt integration configuration.
API key should be set via NANO_GPT_API_KEY environment variable.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# nano-gpt API Configuration
NANO_GPT_API_URL = "https://nano-gpt.com/api/v1"
NANO_GPT_API_KEY = os.environ.get("NANO_GPT_API_KEY", "")  # Set via environment variable
NANO_GPT_MODEL = "openai/gpt-oss-120b"  # Default model

# Chunking Configuration (per docs/init.md)
CHUNK_SIZE_TOKENS = 2000
CHUNK_OVERLAP_TOKENS = 200

# summarization Configuration
SHORT_SUMMARY_LINES = 1-2
MEDIUM_SUMMARY_LINES = 3-5

# Translation Configuration
SUPPORTED_LANGUAGES = ["en", "bm"]  # English, Bahasa Melayu

# RAG Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_STORE_TYPE = "chromadb"

# Application Configuration
APP_NAME = "AI News Companion"
APP_VERSION = "1.0.0"
DEBUG = os.environ.get("DEBUG", "false").lower() == "true"