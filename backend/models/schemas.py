"""
AI News Companion - Request/Response Schemas

Pydantic models for API request/response validation.
"""
from typing import Optional, Union
from pydantic import BaseModel, Field, field_validator


class SummarizeRequest(BaseModel):
    """
    Request schema for article summarization.
    
    Accepts input via:
    - url: URL to a web article
    - file_path: Path to a local file (PDF, DOCX, TXT)
    - text: Plain text content directly
    """
    url: Optional[str] = Field(
        default=None,
        description="URL of the article to summarize",
        examples=["https://example.com/news/article"]
    )
    file_path: Optional[str] = Field(
        default=None,
        description="Path to a local file (PDF, DOCX, TXT)",
        examples=["/path/to/document.pdf"]
    )
    text: Optional[str] = Field(
        default=None,
        description="Plain text content to summarize",
        examples=["Article content here..."]
    )
    
    @field_validator('url', 'file_path', 'text', mode='before')
    @classmethod
    def strip_whitespace(cls, v):
        """Strip whitespace from string fields."""
        if isinstance(v, str):
            return v.strip()
        return v
    
    def model_post_init(self, __context) -> None:
        """Validate that at least one input is provided."""
        # Check if all fields are None/empty
        has_url = bool(self.url)
        has_file = bool(self.file_path)
        has_text = bool(self.text)
        
        if not any([has_url, has_file, has_text]):
            raise ValueError("At least one of 'url', 'file_path', or 'text' must be provided")
    
    def get_input_source(self) -> tuple[str, str]:
        """
        Get the input source and value for processing.
        
        Returns:
            Tuple of (source_type, source_value)
            source_type is one of: 'url', 'file_path', 'text'
        """
        if self.url:
            return ('url', self.url)
        elif self.file_path:
            return ('file_path', self.file_path)
        elif self.text:
            return ('text', self.text)
        else:
            raise ValueError("No input source available")


class SummarizeResponse(BaseModel):
    """
    Response schema for article summarization.
    
    Contains three types of summaries:
    - short_summary: 1-2 lines overview
    - medium_summary: 3-5 lines detailed summary
    - headline: Single compelling headline
    """
    short_summary: str = Field(
        description="Short summary (1-2 lines) capturing the essence of the article",
        min_length=1,
        max_length=500,
        examples=["AI advances in healthcare show promising results for early diagnosis."]
    )
    medium_summary: str = Field(
        description="Medium summary (3-5 lines) with more detail",
        min_length=1,
        max_length=2000,
        examples=[
            "Researchers have developed a new AI system that can detect diseases earlier than traditional methods. "
            "The system uses advanced machine learning to analyze patient data. "
            "Early trials show significant improvements in diagnostic accuracy."
        ]
    )
    headline: str = Field(
        description="Single compelling headline for the article",
        min_length=1,
        max_length=200,
        examples=["Revolutionary AI System Promises Earlier Disease Detection"]
    )


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""
    status: str = Field(description="Service health status")
    version: str = Field(description="Application version")
    llm_connected: bool = Field(description="LLM API connectivity status")


class ErrorResponse(BaseModel):
    """Response schema for error responses."""
    error: str = Field(description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error information")


class TranslateRequest(BaseModel):
    """
    Request schema for BM↔English translation.
    
    Supports bidirectional translation between:
    - English (en)
    - Bahasa Melayu (bm)
    
    If source_lang and target_lang are not provided, the service will
    auto-detect the source language and translate to the opposite language.
    """
    text: str = Field(
        description="Text content to translate",
        min_length=1,
        max_length=10000,
        examples=["The Prime Minister announced new economic policies today."]
    )
    source_lang: Optional[str] = Field(
        default=None,
        description="Source language code (optional, will auto-detect if not provided)",
        pattern="^(en|bm)$",
        examples=["en", "bm"]
    )
    target_lang: Optional[str] = Field(
        default=None,
        description="Target language code (optional, will auto-detect opposite of source if not provided)",
        pattern="^(en|bm)$",
        examples=["bm", "en"]
    )
    
    @field_validator('source_lang', 'target_lang', mode='before')
    @classmethod
    def normalize_language_code(cls, v):
        """Normalize language codes to lowercase."""
        if isinstance(v, str):
            return v.lower().strip()
        return v
    
    @field_validator('source_lang', 'target_lang')
    @classmethod
    def validate_language_pair(cls, v, info):
        """Validate that if both languages are provided, they are different."""
        # This validation will be done in the service layer for more flexibility
        return v


class TranslateResponse(BaseModel):
    """
    Response schema for translation.
    
    Contains the translated text with tone/style preservation.
    """
    translated_text: str = Field(
        description="Translated text preserving news style tone",
        min_length=1,
        max_length=15000,
        examples=["Perdana Menteri mengumumkan dasar ekonomi baharu hari ini."]
    )
    maintained_tone: str = Field(
        description="Indicator that news style was maintained",
        default="news",
        examples=["news"]
    )


class ChatRequest(BaseModel):
    """
    Request schema for RAG-based chat about articles.
    
    Allows users to ask questions about previously loaded article content.
    """
    session_id: str = Field(
        description="Unique session identifier (UUID) for tracking conversation",
        pattern="^[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}$",
        examples=["550e8400-e29b-41d4-a716-446655440000"]
    )
    question: str = Field(
        description="Question about the article content",
        min_length=1,
        max_length=2000,
        examples=["What is the main topic of this article?"]
    )


class ChatResponse(BaseModel):
    """
    Response schema for RAG-based chat.
    
    Contains the generated answer based on retrieved article context.
    """
    answer: str = Field(
        description="Answer generated from article context",
        min_length=1,
        max_length=5000,
        examples=["Based on the article, the main topic is..."]
    )
    session_id: str = Field(
        description="Session identifier (echoed back)",
        examples=["550e8400-e29b-41d4-a716-446655440000"]
    )
