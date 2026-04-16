"""
Parser Service - Article content extraction from various sources.

This service provides a unified interface for extracting text from:
- URLs (web articles)
- PDF files
- DOCX files
- Plain text
"""

from typing import Union, Optional
from pathlib import Path
import logging
import requests
from bs4 import BeautifulSoup
import PyPDF2
from docx import Document

logger = logging.getLogger(__name__)


class ParserError(Exception):
    """Exception raised when parsing fails."""
    pass


class ParserService:
    """
    Unified service for parsing article content from various sources.
    
    Usage:
        service = ParserService()
        text = service.parse("https://example.com/article")
        text = service.parse("path/to/document.pdf")
        text = service.parse("path/to/document.docx")
        text = service.parse("Plain text content")
    """
    
    def __init__(self):
        """Initialize the parser service."""
        self._supported_extensions = {'.pdf', '.docx', '.doc', '.txt'}
    
    def parse(self, source: str) -> str:
        """
        Parse content from a source (URL, file path, or plain text).
        
        Args:
            source: URL string, file path, or plain text content
            
        Returns:
            Extracted text content
            
        Raises:
            ParserError: If parsing fails for any reason
        """
        if not source or not source.strip():
            raise ParserError("Source cannot be empty")
        
        source = source.strip()
        
        # Determine input type and parse accordingly
        if self._is_url(source):
            return self._parse_url(source)
        elif self._is_file_path(source):
            return self._parse_file(source)
        else:
            # Treat as plain text
            return self._parse_plain_text(source)
    
    def _is_url(self, source: str) -> bool:
        """Check if source is a URL."""
        return source.startswith(('http://', 'https://'))
    
    def _is_file_path(self, source: str) -> bool:
        """Check if source is a file path with supported extension."""
        path = Path(source)
        return path.suffix.lower() in self._supported_extensions
    
    def _parse_plain_text(self, text: str) -> str:
        """
        Parse plain text content.
        
        Args:
            text: Plain text string
            
        Returns:
            The text as-is
        """
        logger.debug("Parsing plain text")
        return text.strip()
    
    def _parse_url(self, url: str) -> str:
        """
        Parse article content from a URL.
        
        Args:
            url: URL to fetch and parse
            
        Returns:
            Extracted article text
            
        Raises:
            ParserError: If URL fetch or parsing fails
        """
        logger.debug(f"Parsing URL: {url}")
        try:
            # Fetch the URL with a common user agent
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script, style, and other non-content elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                element.decompose()
            
            # Try to find main article content using common selectors
            article_content = self._extract_article_content(soup)
            
            if article_content:
                return self._clean_text(article_content.get_text(separator='\n', strip=True))
            
            # Fallback to body content
            body = soup.find('body')
            if body:
                return self._clean_text(body.get_text(separator='\n', strip=True))
            
            # Last resort: return all text
            return self._clean_text(soup.get_text(separator='\n', strip=True))
            
        except requests.exceptions.RequestException as e:
            raise ParserError(f"Failed to fetch URL: {e}")
        except Exception as e:
            raise ParserError(f"Failed to parse URL: {e}")
    
    def _extract_article_content(self, soup: BeautifulSoup) -> Optional:
        """
        Extract main article content from HTML.
        
        Tries common article selectors to find the main content,
        avoiding navigation, sidebars, and other boilerplate.
        
        Args:
            soup: BeautifulSoup object of the HTML
            
        Returns:
            BeautifulSoup tag containing article content, or None
        """
        # Common article selectors (in priority order)
        article_selectors = [
            'article',
            '[role="main"]',
            '.article-content',
            '.article-body',
            '.post-content',
            '.entry-content',
            '.content',
            '#content',
            '.main',
            '#main',
        ]
        
        for selector in article_selectors:
            element = soup.select_one(selector)
            if element:
                return element
        
        return None
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing excessive whitespace.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove multiple consecutive newlines
        lines = [line.strip() for line in text.split('\n')]
        lines = [line for line in lines if line]  # Remove empty lines
        
        return '\n'.join(lines)
    
    def _parse_file(self, file_path: str) -> str:
        """
        Parse content from a file (PDF, DOCX, etc.).
        
        Args:
            file_path: Path to the file
            
        Returns:
            Extracted text content
            
        Raises:
            ParserError: If file parsing fails
        """
        path = Path(file_path)
        
        if not path.exists():
            raise ParserError(f"File not found: {file_path}")
        
        extension = path.suffix.lower()
        
        if extension == '.pdf':
            return self._parse_pdf(file_path)
        elif extension in ('.docx', '.doc'):
            return self._parse_docx(file_path)
        elif extension == '.txt':
            return self._parse_txt(file_path)
        else:
            raise ParserError(f"Unsupported file format: {extension}")
    
    def _parse_pdf(self, file_path: str) -> str:
        """
        Parse PDF file content.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Extracted text
            
        Raises:
            ParserError: If PDF parsing fails
        """
        logger.debug(f"Parsing PDF file: {file_path}")
        try:
            text_parts = []
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text.strip())
            
            if not text_parts:
                logger.warning(f"No text extracted from PDF: {file_path}")
                return ""
            
            return '\n'.join(text_parts)
            
        except PyPDF2.errors.PdfReadError as e:
            raise ParserError(f"Failed to read PDF (corrupted or invalid): {e}")
        except Exception as e:
            raise ParserError(f"Failed to parse PDF: {e}")
    
    def _parse_docx(self, file_path: str) -> str:
        """
        Parse DOCX file content.
        
        Args:
            file_path: Path to DOCX file
            
        Returns:
            Extracted text
            
        Raises:
            ParserError: If DOCX parsing fails
        """
        logger.debug(f"Parsing DOCX file: {file_path}")
        try:
            doc = Document(file_path)
            text_parts = []
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_parts.append(paragraph.text.strip())
            
            if not text_parts:
                logger.warning(f"No text extracted from DOCX: {file_path}")
                return ""
            
            return '\n'.join(text_parts)
            
        except Exception as e:
            raise ParserError(f"Failed to parse DOCX: {e}")
    
    def _parse_txt(self, file_path: str) -> str:
        """
        Parse TXT file content.
        
        Args:
            file_path: Path to TXT file
            
        Returns:
            File content as text
            
        Raises:
            ParserError: If file reading fails
        """
        logger.debug(f"Parsing TXT file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except UnicodeDecodeError:
            # Try with latin-1 encoding as fallback
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read().strip()
        except Exception as e:
            raise ParserError(f"Failed to read TXT file: {e}")