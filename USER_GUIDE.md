# AI News Companion - User Guide

This guide provides detailed instructions for setting up and using the AI News Companion application.

## Table of Contents

- [Setup Guide](#setup-guide)
- [Usage Guide](#usage-guide)
- [Testing Guide](#testing-guide)
- [Troubleshooting](#troubleshooting)

---

## Setup Guide

### Prerequisites

- Python 3.10 or higher
- A nano-gpt API key (obtain from https://nano-gpt.com/)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd ai-news-companion
```

### Step 2: Create a Virtual Environment

Creating a virtual environment is recommended to isolate project dependencies:

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages:
- FastAPI and Uvicorn (web framework)
- LangChain and ChromaDB (LLM and vector store)
- BeautifulSoup4 (HTML parsing)
- PyPDF2 and python-docx (document processing)
- pytest (testing)

### Step 4: Configure Environment Variables

1. Copy the example environment file:

   ```bash
   cp .env.example .env
   ```

2. Open `.env` in a text editor and add your nano-gpt API key:

   ```
   NANO_GPT_API_KEY=your_actual_api_key_here
   ```

   > **Note**: Your API key is required for all LLM operations. Without it, the application will fail to process requests.

3. (Optional) Enable debug mode by setting:

   ```
   DEBUG=true
   ```

### Step 5: Verify Installation

Run a quick connectivity test to verify your setup:

```bash
python -m pytest tests/test_llm_connectivity.py -v
```

If the test passes, your setup is complete.

### Step 6: Start the Backend Server

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

The server will start and show output similar to:

```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

### Step 7: Access the Frontend

Open [`frontend/index.html`](frontend/index.html) in your web browser. You can also access the API documentation at `http://localhost:8000/docs`.

---

## Usage Guide

### Summarize an Article

The summarization feature accepts three input types: URL, file, or plain text.

#### Method 1: URL Input

1. Open the frontend interface in your browser
2. Navigate to the Summarize tab
3. Select **URL** as the input type
4. Enter the article URL (e.g., `https://example.com/news/article`)
5. Click **Summarize**
6. Wait for the processing to complete

The system will:
1. Fetch and parse the web page using BeautifulSoup4
2. Extract the article content
3. Generate three summary types:
   - **Short Summary**: 1-2 line overview
   - **Medium Summary**: 3-5 line detailed summary
   - **Headline**: A single compelling headline

#### Method 2: File Upload

1. Navigate to the Summarize tab
2. Select **File** as the input type
3. Click to upload a PDF or DOCX file
4. Click **Summarize**

Supported file formats:
- PDF (`.pdf`)
- Word Document (`.docx`, `.doc`)
- Text files (`.txt`)

#### Method 3: Plain Text Input

1. Navigate to the Summarize tab
2. Select **Text** as the input type
3. Paste or type your article text directly
4. Click **Summarize**

#### Long Article Handling

For articles that exceed the token limit (2000 tokens), the system automatically:

1. Splits the content into overlapping chunks (200 token overlap)
2. Summarizes each chunk individually
3. Combines all chunk summaries
4. Generates final summaries from the combined chunk summaries

This two-stage approach ensures comprehensive coverage while preventing information loss at boundaries.

### Translate Text

The translation feature supports bidirectional translation between English and Bahasa Melayu.

#### To Translate English to Bahasa Melayu (BM):

1. Navigate to the **Translate** tab
2. Select **English → BM** direction
3. Enter or paste your English text
4. Click **Translate**
5. View the translated Bahasa Melayu output

#### To Translate Bahasa Melayu to English:

1. Navigate to the **Translate** tab
2. Select **BM → English** direction
3. Enter or paste your Bahasa Melayu text
4. Click **Translate**
5. View the translated English output

> **Note**: Each translation request consumes API tokens. The system maintains your input for reference.

### Chat About an Article

The chat feature allows you to ask questions about a loaded article using RAG (Retrieval-Augmented Generation).

#### Step 1: Load an Article

1. Navigate to the **Chat** tab
2. Select your input method (URL, File, or Text)
3. Provide the article content
4. Click **Load Article**

The system will:
1. Parse and extract the article content
2. Split it into semantic chunks
3. Generate embeddings for each chunk
4. Store chunks in the vector database

A session ID will be returned to track your conversation.

#### Step 2: Ask Questions

1. Enter your question in the chat input field
2. Click **Send** or press Enter

The system will:
1. Embed your question
2. Retrieve the most relevant chunks from the vector store
3. Generate an answer using only the retrieved context

You can continue asking follow-up questions about the same article. Each question uses the same loaded article context.

#### Chat Tips

- Ask specific questions about facts in the article
- Questions about dates, names, statistics, and events work best
- If the answer isn't in the article, the system will indicate this
- You can load a new article at any time to change the context

---

## Testing Guide

The project includes a comprehensive test suite covering all major components.

### Running All Tests

To run the entire test suite:

```bash
pytest tests/ -v
```

### Running Specific Test Files

| Test File | Purpose |
|-----------|---------|
| `test_llm_connectivity.py` | Test API key and connectivity |
| `test_llm_service.py` | Test LLM service layer |
| `test_parser_service.py` | Test content parsing |
| `test_summarizer.py` | Test summarization logic |
| `test_translator.py` | Test translation logic |
| `test_rag_service.py` | Test RAG pipeline |
| `test_summarize_api.py` | Test `/api/summarize` endpoint |
| `test_translate_api.py` | Test `/api/translate` endpoint |
| `test_chat_api.py` | Test `/api/chat` endpoints |

### Running Individual Tests

```bash
# Run only parser tests
pytest tests/test_parser_service.py -v

# Run only API tests
pytest tests/test_summarize_api.py tests/test_translate_api.py -v
```

### Understanding Test Output

- **PASSED**: The test succeeded
- **FAILED**: The test did not pass (check error message for details)
- **ERROR**: The test couldn't run (likely a setup issue)

### Test Coverage

The test suite covers:
- Input validation and error handling
- Parsing for all supported formats (URL, PDF, DOCX, text)
- Summarization with various input sizes
- Translation between both language pairs
- RAG retrieval and generation
- API endpoint responses

---

## Troubleshooting

### Common Issues and Solutions

#### Issue: "Missing API Key" Error

**Symptom**: Application returns error about missing `NANO_GPT_API_KEY`

**Solution**:
1. Check that `.env` file exists in the project root
2. Verify the file contains `NANO_GPT_API_KEY=your_key_here`
3. Ensure there are no spaces around the `=` sign
4. Restart the server after modifying `.env`

#### Issue: "Connection Timeout" Error

**Symptom**: Requests timeout or take very long to respond

**Solution**:
1. Check your internet connection
2. Verify the nano-gpt API is accessible
3. Try a smaller input (shorter text or simpler URL)
4. Check if you've exceeded API rate limits

#### Issue: "Parser Error" for URLs

**Symptom**: Web article parsing fails with `ParserError`

**Solution**:
1. Verify the URL is accessible in your browser
2. Check if the website requires authentication
3. Try a different article URL
4. Some websites block automated access

#### Issue: "Empty Results" from Summarization

**Symptom**: Summarization returns empty or very short results

**Solution**:
1. Verify the source content has sufficient text
2. For URLs, the page may have mostly images/video
3. For files, ensure the file contains readable text
4. PDFs that are scanned images (not text) cannot be parsed

#### Issue: Chat Doesn't Remember Context

**Symptom**: Chat appears to forget previous questions

**Solution**:
1. Ensure you're using the same session ID
2. Check that the article was successfully loaded (no errors)
3. Verify the question is about the loaded article
4. Try reloading the article if issues persist

#### Issue: Tests Fail with Import Error

**Symptom**: `ModuleNotFoundError` when running tests

**Solution**:
1. Ensure all dependencies are installed: `pip install -r requirements.txt`
2. Verify you're in the project root directory
3. Check that the virtual environment is activated
4. Try running with explicit Python path:

   ```bash
   set PYTHONPATH=. && pytest tests/
   ```

#### Issue: CORS Errors in Browser

**Symptom**: Browser shows CORS policy errors when using frontend

**Solution**:
1. Ensure the backend server is running
2. Check that the backend is configured to allow the frontend origin
3. For development, the backend allows all origins (`*`)
4. Clear browser cache or try incognito mode

### Getting Help

If you encounter issues not covered here:

1. Check the [project README](README.md) for additional information
2. Review API documentation at `http://localhost:8000/docs`
3. Enable debug mode in `.env` for detailed logging
4. Check server logs for error details

### Debug Mode

Enable debug mode for verbose logging:

```
DEBUG=true
```

Restart the server after changing this setting. Debug logs will show:
- Request/response details
- Parsing intermediate results
- LLM prompt construction
- Vector store operations