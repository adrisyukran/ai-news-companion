# AI News Companion

AI News Companion is a powerful backend service that provides AI-powered summarization, translation, and RAG-based chat capabilities specifically designed for Malaysian news articles. Built with FastAPI and modern NLP libraries, it offers a unified pipeline for processing news content from various input formats.

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Web Framework** | FastAPI, Uvicorn |
| **LLM Integration** | nano-gpt API, LangChain |
| **Vector Store** | ChromaDB |
| **HTML Parsing** | BeautifulSoup4 |
| **Document Processing** | PyPDF2, python-docx |
| **Frontend** | Vanilla JavaScript, CSS, HTML |
| **Testing** | pytest, pytest-asyncio |

## Pipeline Explanation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AI News Companion Pipeline                         │
└─────────────────────────────────────────────────────────────────────────────┘

   ┌─────────────┐     ┌─────────────┐     ┌─────────────────────┐     ┌─────────────┐
   │    INPUT    │ ──▶    PARSING     ──▶     LLM PROCESSING     ──▶      OUTPUT   
   └─────────────┘     └─────────────┘     └─────────────────────┘     └─────────────┘

   Input Types:                 Parser Service:            Processing Modes:          Output Types:
   ─────────────               ─────────────              ──────────────             ───────────
   • URL (web article)         • HTML → BeautifulSoup     • Summarization           • Short Summary (1-2 lines)
   • PDF file                  • PDF → PyPDF2            • Translation             • Medium Summary (3-5 lines)
   • DOCX file                • DOCX → python-docx      • RAG Chat                • Headline
   • Plain text               • Plain text pass-through  •                         • Translated text
                                                                                   • Chat response
```

### Pipeline Flow

1. **Input Reception**: Accept content via URL, file upload, or direct text input
2. **Parsing**: Extract text content using format-appropriate parsers
3. **LLM Processing**: Route to appropriate processing pipeline (summarize/translate/chat)
4. **Output Generation**: Return structured, formatted results

## Technical Details

### Input Handling

The [`ParserService`](backend/services/parser_service.py:27) handles multiple input formats through a unified interface:

| Input Type | Detection Method | Parser Used |
|------------|------------------|-------------|
| URL | `source.startswith(('http://', 'https://'))` | BeautifulSoup4 |
| PDF | Path with `.pdf` extension | PyPDF2 |
| DOCX | Path with `.docx`/`.doc` extension | python-docx |
| Plain Text | Any other string | Direct pass-through |

**Long Input Chunking**: Content exceeding token limits is split into manageable chunks:

- **Chunk Size**: 2000 tokens ([`CHUNK_SIZE_TOKENS`](backend/config.py:15))
- **Overlap**: 200 tokens between chunks ([`CHUNK_OVERLAP_TOKENS`](backend/config.py:16))

This ensures context continuity while respecting LLM context windows.

### LLM Processing

**Prompt Engineering Strategies**:

1. **System Prompt Constraints**: All prompts include strict hallucination prevention rules
2. **Explicit Information Boundaries**: Models are instructed to only use provided content
3. **Fallback Instructions**: When information is insufficient, models state missing information rather than guessing

**Summarization Approach (Two-Stage)**:

```
Stage 1: Chunk Summaries
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Original Text  │ ──▶ │   Split into    │ ──▶ │  Summarize each │ ──▶ Chunk Summaries
│  (Long Article) │     │    chunks       │     │     chunk       │
└─────────────────┘     └─────────────────┘     └─────────────────┘

Stage 2: Final Summaries
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Chunk Summaries │ ──▶ │ Combine and     │ ──▶ │  Generate final │
│                 │     │ summarize       │     │   summaries     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                            │
                            ▼
                   ┌─────────────────────┐
                   │  3 Output Types:    │
                   │  • Short (1-2 lines)│
                   │  • Medium (3-5)     │
                   │  • Headline         │
                   └─────────────────────┘
```

**RAG Retrieval Chain**:

```
User Question
      │
      ▼
┌───────────────┐     ┌──────────────┐     ┌───────────────┐
│ Query Embedding│ ──▶ │ Chroma Vector│ ──▶ │ Top-k Chunks  │
│  (MiniLM-L6)   │     │   Search     │     │  Retrieved    │
└───────────────┘     └──────────────┘     └───────────────┘
                                                │
                                                ▼
                                    ┌───────────────────────┐
                                    │ LLM generates answer  │
                                    │ using retrieved context│
                                    └───────────────────────┘
```

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` ([`EMBEDDING_MODEL`](backend/config.py:26))
- **Vector Store**: ChromaDB with in-memory storage
- **Retrieval**: Top-k (k=5) most relevant chunks

### Output Formatting

All LLM operations return structured responses:

**Summarization Response**:
```json
{
  "short_summary": "Brief 1-2 line overview of the article",
  "medium_summary": "3-5 line detailed summary",
  "headline": "Single compelling headline"
}
```

**Translation Response**:
```json
{
  "original_text": "The input text",
  "translated_text": "The translated output",
  "source_language": "en|bm",
  "target_language": "en|bm"
}
```

**RAG Chat Response**:
```json
{
  "answer": "Generated response based on article context",
  "session_id": "unique-session-id"
}
```

### Error Handling

**Custom Exceptions**:

| Exception | Location | Raised When |
|-----------|----------|-------------|
| `ParserError` | [`parser_service.py:22`](backend/services/parser_service.py:22) | Content parsing fails |
| `SummarizerError` | [`summarizer.py:32`](backend/services/summarizer.py:32) | Summarization fails |
| `LLMError` | [`llm_service.py`](backend/services/llm_service.py) | API communication fails |

**Retry Logic**:

- API requests include automatic retry with exponential backoff
- HTTP status codes 429 (rate limit) trigger automatic retry
- Maximum 3 retry attempts before failing

**API Validation**:

- Pydantic models validate all request/response data
- Input length validation before processing
- File type validation for uploaded documents

## How to Use

### Starting the Backend

1. **Configure Environment**:

   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env and add your nano-gpt API key
   # NANO_GPT_API_KEY=your_api_key_here
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Server**:

   ```bash
   uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
   ```

   The API will be available at `http://localhost:8000`
   - API docs: `http://localhost:8000/docs`
   - Health check: `http://localhost:8000/health`

### Opening the Frontend

After starting the backend, open [`frontend/index.html`](frontend/index.html) in a web browser. The frontend provides:

- URL input for web article summarization
- File upload for PDF/DOCX documents
- Plain text input for direct processing
- Translation interface (EN ↔ BM)
- Chat interface for article-based Q&A

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/summarize` | POST | Summarize article from URL, file, or text |
| `/api/translate` | POST | Translate text between English and Bahasa Melayu |
| `/api/chat/load` | POST | Load article into chat session |
| `/api/chat/ask` | POST | Ask question about loaded article |
| `/health` | GET | Health check endpoint |
| `/` | GET | Root endpoint with API info |