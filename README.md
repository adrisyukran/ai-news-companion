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

The AI News Companion provides a unified single-page workflow for news processing:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AI News Companion - Single Page Flow                      │
└─────────────────────────────────────────────────────────────────────────────┘

   ┌─────────────┐     ┌─────────────┐     ┌─────────────────────┐     ┌─────────────┐
   │    INPUT    │ ──▶    PARSING     ──▶     PROCESSING         ──▶      OUTPUT
   └─────────────┘     └─────────────┘     └─────────────────────┘     └─────────────┘

   Input Types:                 Parser Service:            Operations:               Output Types:
   ─────────────               ─────────────              ─────────────             ───────────
   • PDF file                  • PDF → PyPDF2            • Summarize               • Short Summary (1-2 lines)
   • DOCX file                • DOCX → python-docx      • Translate               • Medium Summary (3-5 lines)
   • Plain text               • Plain text pass-through  • Load to Chat (RAG)      • Headline
                                                                                    • Translated text
                                                                                    • Chat response
   Note: URL input is available via backend API only
```

### Pipeline Flow

1. **Input Reception**: Accept content via file upload or direct text input
2. **Parsing**: Extract text content using format-appropriate parsers
3. **Content Filtering**: Remove advertisements, sponsored content, and navigation menus
4. **Processing**: Route to appropriate operation (summarize/translate/load-to-chat)
5. **Output Generation**: Return structured, formatted results

### Frontend Flow

The frontend presents a unified single-page interface:

```
┌─────────────────────────────────────────────────────────────────┐
│                      Single Page Interface                       │
├─────────────────────────────────────────────────────────────────┤
│  [Input] → [Select Operation] → [View Output] → [Chat/RAG]      │
└─────────────────────────────────────────────────────────────────┘
```

- **Input**: Paste text or upload PDF/DOCX files
- **Operations**: Choose between Summarize, Translate, or Load to Chat
- **Output**: View generated summaries, translations, or confirmation
- **Chat**: Ask questions about loaded content using RAG

## Technical Details

### Input Handling

The [`ParserService`](backend/services/parser_service.py:27) handles multiple input formats through a unified interface:

| Input Type | Detection Method | Parser Used |
|------------|------------------|-------------|
| PDF | Path with `.pdf` extension | PyPDF2 |
| DOCX | Path with `.docx`/`.doc` extension | python-docx |
| Plain Text | Any other string | Direct pass-through |

**Note**: URL input is supported via the backend API only. The frontend provides file upload and text paste options.

**Content Filtering for Summarization**:

When processing pasted text, the system automatically filters out:
- Advertisements and sponsored content
- Navigation menus and headers
- Non-article content to focus on main article body

This ensures cleaner input for summarization and improved headline generation.

**Long Input Chunking**: Content exceeding token limits is split into manageable chunks:

- **Chunk Size**: 2000 tokens ([`CHUNK_SIZE_TOKENS`](backend/config.py:15))
- **Overlap**: 200 tokens between chunks ([`CHUNK_OVERLAP_TOKENS`](backend/config.py:16))

This ensures context continuity while respecting LLM context windows.

### Translation

The translation system uses a two-stage approach for efficient and accurate translations:

**Stage 1: ArgosTranslate (Offline/Open-Source)**
- Primary translation via ArgosTranslate library
- Supports English ↔ Bahasa Melayu bidirectional translation
- Runs completely offline without external API calls
- More efficient than full LLM generation for translation tasks

**Stage 2: LLM Refinement**
- Secondary LLM-based refinement of translated text
- Fixes spelling, grammar, and phrasing issues
- Ensures natural, fluent output quality

**DBP Standards for Bahasa Melayu**:
- When translating to Bahasa Melayu, refinement follows **Dewan Bahasa dan Pustaka (DBP)** standards
- Ensures formal and accurate language usage
- Avoids Indonesian influence where possible
- Produces professionally-appropriate output

**Auto-Detection**:
- The system automatically detects the source language (English or Bahasa Melayu)
- Translates to the opposite language automatically
- No manual language direction selection required

This approach balances efficiency, quality, and linguistic accuracy for Malaysian news content.

### LLM Processing

**Prompt Engineering Strategies**:

1. **System Prompt Constraints**: All prompts include strict hallucination prevention rules
2. **Explicit Information Boundaries**: Models are instructed to only use provided content
3. **Fallback Instructions**: When information is insufficient, models state missing information rather than guessing

**Summarization Improvements**:

The summarization system includes enhanced content filtering and headline generation:

1. **Ad/Sponsor Filtering**: Automatically removes advertisements, sponsored content, and navigation elements from pasted text before processing
2. **Improved Headlines**: News-style headline generation that is factual, concise, and article-appropriate

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
- **Vector Store**: ChromaDB with versioned collections for session isolation
- **Retrieval**: Top-k (k=5) most relevant chunks

**Session Isolation**:
- Each article load creates a new versioned collection
- Loading a new article completely clears the previous context
- Prevents data leakage between articles in different sessions

**Clean Output**:
- Internal chunk markers (`【Chunk X】`) are removed from final responses
- Chat responses support full markdown rendering (bold, lists, code blocks)
- Uses `marked.js` for markdown parsing in the frontend

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

After starting the backend, open [`frontend/index.html`](frontend/index.html) in a web browser. The frontend provides a unified single-page interface:

- **Input**: Paste text directly or upload PDF/DOCX files
- **Operations**: Select from Summarize, Translate, or Load to Chat
- **Output**: View generated content in the results area
- **Chat**: Ask questions about loaded content using RAG

**Note**: URL input is available via the backend API (`/api/summarize` with URL source), but the frontend focuses on file and text input for better user experience.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/summarize` | POST | Summarize article from text or file |
| `/api/translate` | POST | Translate text auto-detecting English ↔ Bahasa Melayu (via ArgosTranslate + LLM refinement) |
| `/api/chat/load` | POST | Load article into RAG chat session with versioned collection isolation |
| `/api/chat/ask` | POST | Ask question about loaded article (returns markdown-formatted response) |
| `/health` | GET | Health check endpoint |
| `/` | GET | Root endpoint with API info |