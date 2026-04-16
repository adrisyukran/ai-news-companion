"""
AI News Companion - LLM Service

nano-gpt API integration with chunking, streaming, and error handling.
"""
import asyncio
import logging
from typing import AsyncGenerator, Generator, List, Optional, Dict, Any
from dataclasses import dataclass

import httpx

from backend.config import (
    NANO_GPT_API_URL,
    NANO_GPT_API_KEY,
    NANO_GPT_MODEL,
    CHUNK_SIZE_TOKENS,
    CHUNK_OVERLAP_TOKENS,
)

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Container for LLM response data."""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None


class NanoGPTService:
    """
    Service for interacting with nano-gpt API.
    
    Features:
    - Chunking for long inputs
    - Streaming support
    - Retry logic with exponential backoff
    - Error handling
    """
    
    def __init__(
        self,
        api_url: str = NANO_GPT_API_URL,
        api_key: str = NANO_GPT_API_KEY,
        model: str = NANO_GPT_MODEL,
        chunk_size: int = CHUNK_SIZE_TOKENS,
        chunk_overlap: int = CHUNK_OVERLAP_TOKENS,
        max_retries: int = 3,
        timeout: float = 30.0,
    ):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Validate API key
        if not self.api_key:
            logger.warning("NANO_GPT_API_KEY is not set. API calls will fail.")
        
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with proper headers."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client
    
    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Simple estimation: ~4 characters per token for English.
        For production, use tiktoken library for accuracy.
        """
        return len(text) // 4
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks with overlap.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        estimated_tokens = self._estimate_tokens(text)
        
        # If text fits in single chunk, return as-is
        if estimated_tokens <= self.chunk_size:
            return [text]
        
        chunks = []
        # Calculate chunk size in characters (approximate)
        chars_per_token = 4
        chunk_size_chars = self.chunk_size * chars_per_token
        overlap_chars = self.chunk_overlap * chars_per_token
        
        start = 0
        while start < len(text):
            end = start + chunk_size_chars
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings in last 20% of chunk
                search_start = int(len(chunk) * 0.8)
                for ending in ['. ', '.\n', '! ', '!\n', '? ', '?\n']:
                    last_ending = chunk.rfind(ending, search_start)
                    if last_ending != -1:
                        chunk = chunk[:last_ending + len(ending)]
                        break
            
            chunks.append(chunk.strip())
            start = end - overlap_chars if end < len(text) else len(text)
        
        return chunks
    
    async def _request_with_retry(
        self,
        endpoint: str,
        payload: Dict[str, Any],
        stream: bool = False,
    ) -> httpx.Response:
        """
        Make API request with exponential backoff retry.
        
        Args:
            endpoint: API endpoint path
            payload: Request payload
            stream: Whether to stream response
            
        Returns:
            httpx Response object
        """
        client = await self._get_client()
        url = f"{self.api_url}{endpoint}"
        
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                if stream:
                    # For streaming, use httpx stream context
                    response = await client.post(url, json=payload)
                else:
                    response = await client.post(url, json=payload)
                
                # Check for rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 2 ** attempt))
                    logger.warning(f"Rate limited. Retrying after {retry_after}s")
                    await asyncio.sleep(retry_after)
                    continue
                
                # Raise for other errors
                response.raise_for_status()
                return response
                
            except httpx.HTTPStatusError as e:
                last_exception = e
                logger.error(f"HTTP error on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    
            except httpx.RequestError as e:
                last_exception = e
                logger.error(f"Request error on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
        
        raise last_exception or Exception("Request failed after all retries")
    
    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Get completion from nano-gpt API.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            LLMResponse with completion result
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
        
        response = await self._request_with_retry("/chat/completions", payload)
        data = response.json()
        
        choice = data["choices"][0]
        return LLMResponse(
            content=choice["message"]["content"],
            model=data.get("model", self.model),
            usage=data.get("usage"),
            finish_reason=choice.get("finish_reason"),
        )
    
    async def complete_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream completion from nano-gpt API.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Yields:
            Chunks of completion text
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
        
        response = await self._request_with_retry("/chat/completions", payload, stream=True)
        
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                data = line[6:]  # Remove "data: " prefix
                if data.strip() == "[DONE]":
                    break
                try:
                    import json
                    parsed = json.loads(data)
                    choice = parsed["choices"][0]
                    delta = choice.get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content
                except (json.JSONDecodeError, KeyError):
                    continue
    
    async def complete_with_chunking(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Handle long prompts by chunking and combining results.
        
        Args:
            prompt: User prompt (may be very long)
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate per chunk
            
        Returns:
            Combined LLMResponse
        """
        chunks = self._chunk_text(prompt)
        
        if len(chunks) == 1:
            # Single chunk, use normal completion
            return await self.complete(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        
        logger.info(f"Splitting prompt into {len(chunks)} chunks")
        
        # Process each chunk
        chunk_results = []
        for i, chunk in enumerate(chunks):
            chunk_prompt = f"""Based on the following article excerpt, provide a concise summary:

{chunk}

Summary:"""
            
            result = await self.complete(
                prompt=chunk_prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            chunk_results.append(result.content)
            logger.info(f"Processed chunk {i + 1}/{len(chunks)}")
        
        # Combine chunk summaries
        combined_text = "\n\n".join(chunk_results)
        final_prompt = f"""Combine these summaries into a single coherent summary:

{combined_text}

Combined Summary:"""
        
        final_result = await self.complete(
            prompt=final_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        return final_result
    
    async def test_connectivity(self) -> bool:
        """
        Test API connectivity with a simple request.
        
        Returns:
            True if connection successful
        """
        try:
            response = await self.complete(
                prompt="Say 'Hello' in exactly one word.",
                max_tokens=10,
            )
            return len(response.content) > 0
        except Exception as e:
            logger.error(f"Connectivity test failed: {e}")
            return False
