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

The AI News Companion uses a unified single-page interface with four main steps:

```
Input → Select Operation → View Output → Chat (optional)
```

### Step 1: Provide Input

The system accepts two input types:

#### File Upload

1. Click to upload a PDF or DOCX file
2. Supported formats: PDF (`.pdf`), Word Document (`.docx`, `.doc`)

#### Plain Text Input

1. Paste or type your article text directly into the input area

**Content Filtering**: The system automatically filters out advertisements, sponsored content, and navigation menus from pasted text to focus on the main article content.

### Step 2: Choose an Operation

The interface provides direct action buttons for each operation:

| Button | Description |
|--------|-------------|
| **Summarize** | Generate short/medium summaries and headline |
| **Translate** | Auto-translate between English and Bahasa Melayu |
| **Load to Chat** | Initialize a RAG session for article-based Q&A |

Simply click the appropriate button to execute the operation. Previous results are automatically cleared when starting a new operation.

### Step 3: View Output

The output appears in the results area:
- **Summarize**: Shows short summary, medium summary, and headline
- **Translate**: Shows translated text
- **Load to Chat**: Shows confirmation when article is loaded

### Step 4: Chat (Optional)

For **Load to Chat**, a chat interface becomes available:

1. Enter your question about the article
2. Click **Send** or press Enter
3. The system retrieves relevant content and generates an answer

You can ask follow-up questions about the same article. To discuss a different article, select "Load to Chat" again with new content.

### Long Article Handling

For articles that exceed the token limit (2000 tokens), the system automatically:

1. Splits the content into overlapping chunks (200 token overlap)
2. Summarizes each chunk individually
3. Combines all chunk summaries
4. Generates final summaries from the combined chunk summaries

This two-stage approach ensures comprehensive coverage while preventing information loss at boundaries.

### Summarize an Article

1. Paste text or upload a file in the input area
2. Select **Summarize** as the operation
3. Click **Submit**
4. View the generated summaries:
   - **Short Summary**: 1-2 line overview
   - **Medium Summary**: 3-5 line detailed summary
   - **Headline**: A single news-style headline

The system filters out ads and navigation content to focus on the article body, resulting in cleaner summaries and more accurate headlines.

### Translate Text

The translation feature supports bidirectional translation between English and Bahasa Melayu.

1. Paste or enter text in the input area
2. Click **Translate**
3. View the translated output

**Auto-Detection**: The system automatically detects whether your input is in English or Bahasa Melayu and translates to the opposite language. No manual direction selection is needed.

**How Translation Works**:
- Primary translation via ArgosTranslate (offline/open-source library)
- LLM-based refinement to fix spelling, grammar, and phrasing
- For Bahasa Melayu output, refinement follows Dewan Bahasa dan Pustaka (DBP) standards to ensure formal, accurate language
- This two-stage approach ensures accurate, professionally-appropriate translations

### Load to Chat

The "Load to Chat" operation initializes a RAG (Retrieval-Augmented Generation) session for article-based Q&A.

1. Paste text or upload a file in the input area
2. Select **Load to Chat** as the operation
3. Click **Submit**

The system will:
1. Parse and extract the article content
2. Filter out ads and navigation content
3. Split it into semantic chunks
4. Generate embeddings for each chunk
5. Store chunks in the vector database
6. Enable the chat interface

Once loaded, you can ask questions about the article.

### Chat About an Article

After using **Load to Chat** to initialize a session:

1. Enter your question in the chat input field
2. Click **Send** or press Enter

The system will:
1. Embed your question
2. Retrieve the most relevant chunks from the vector store
3. Generate an answer using only the retrieved context

**Markdown Support**: Chat responses support full markdown formatting including:
- **Bold** and *italic* text
- Bullet and numbered lists
- Code blocks
- Other standard markdown syntax

**Session Isolation**: Each article is loaded into a separate versioned collection. Loading a new article completely clears the previous context, ensuring questions about one article don't leak into discussions about another.

**Results Auto-Clear**: When you change the input content or start a new operation, previous results and chat history are automatically cleared to prevent confusion.

#### Chat Tips

- Ask specific questions about facts in the article
- Questions about dates, names, statistics, and events work best
- If the answer isn't in the article, the system will indicate this
- Use "Load to Chat" with new content to change the discussion topic

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