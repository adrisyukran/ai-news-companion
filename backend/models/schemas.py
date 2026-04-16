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
