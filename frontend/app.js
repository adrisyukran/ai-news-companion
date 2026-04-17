/**
 * AI News Companion - Frontend Application
 * Unified single-page flow for Summarization, Translation, and Chat features.
 */

// ============================================
// Configuration
// ============================================
const API_BASE_URL = 'http://localhost:8000';

// ============================================
// State Management
// ============================================
const state = {
    currentInputType: 'file',
    sessionId: null,
    articleText: null,
    articleSource: null,
    operationResults: {
        summarize: null,
        translate: null,
    },
};

// ============================================
// Utility Functions
// ============================================

/**
 * Show a loading state on a button
 */
function setButtonLoading(buttonId, loading) {
    const button = document.getElementById(buttonId);
    const loader = button.querySelector('.btn-loader');
    const btnText = button.querySelector('.btn-text');
    
    if (loading) {
        button.disabled = true;
        loader?.classList.remove('hidden');
        if (btnText) btnText.textContent = 'Processing...';
    } else {
        button.disabled = false;
        loader?.classList.add('hidden');
        if (btnText) btnText.textContent = 'Process Article';
    }
}

/**
 * Show an error message
 */
function showError(containerId, message) {
    const errorContainer = document.getElementById(containerId);
    const errorMsg = errorContainer.querySelector('span[id$="-error-msg"]') || 
                     errorContainer.querySelector('span:last-child');
    errorMsg.textContent = message;
    errorContainer.classList.remove('hidden');
}

/**
 * Hide an error message
 */
function hideError(containerId) {
    const errorContainer = document.getElementById(containerId);
    errorContainer.classList.add('hidden');
}

/**
 * Generate a UUID for session management
 */
function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        const r = Math.random() * 16 | 0;
        const v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

/**
 * Save session ID to localStorage
 */
function saveSessionId(sessionId) {
    localStorage.setItem('aiNewsSessionId', sessionId);
    state.sessionId = sessionId;
}

/**
 * Load session ID from localStorage
 */
function loadSessionId() {
    return localStorage.getItem('aiNewsSessionId');
}

/**
 * Clear session ID from localStorage
 */
function clearSessionId() {
    localStorage.removeItem('aiNewsSessionId');
    state.sessionId = null;
}

/**
 * Read file as text
 */
function readFileAsText(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => resolve(e.target.result);
        reader.onerror = (e) => reject(new Error('Failed to read file'));
        reader.readAsText(file);
    });
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ============================================
// Input Panel Switching
// ============================================

function initInputTabs() {
    const tabButtons = document.querySelectorAll('#input-section .input-tab-btn');
    
    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const inputType = btn.dataset.input;
            
            // Update active button
            tabButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Update active panel
            document.querySelectorAll('#input-section .input-panel').forEach(panel => {
                panel.classList.remove('active');
            });
            document.getElementById(`${inputType}-panel`).classList.add('active');
            
            state.currentInputType = inputType;
        });
    });
}

// ============================================
// Input Validation
// ============================================

/**
 * Get the current article text from input
 * Returns null if no valid input
 */
function getArticleText() {
    if (state.currentInputType === 'file') {
        const fileInput = document.getElementById('article-file');
        const file = fileInput.files[0];
        if (!file) {
            return null;
        }
        return readFileAsText(file);
    } else if (state.currentInputType === 'text') {
        const text = document.getElementById('article-text').value.trim();
        if (!text) {
            return null;
        }
        return Promise.resolve(text);
    }
    return null;
}

// ============================================
// Action Button Handlers
// ============================================

/**
 * Handle Summarize button click
 */
async function handleSummarize() {
    hideError('main-error');
    
    const textPromise = getArticleText();
    if (!textPromise) {
        showError('main-error', 'Please provide article text or upload a file');
        return;
    }
    
    try {
        const text = await textPromise;
        
        // Store article text for later use
        state.articleText = text;
        state.articleSource = { type: state.currentInputType, value: state.currentInputType === 'file' ? document.getElementById('article-file').files[0].name : 'inline' };
        
        // Set loading state
        setButtonLoading('summarize-btn', true);
        
        // Hide previous results
        document.getElementById('summarize-results').classList.add('hidden');
        document.getElementById('results-section').classList.add('hidden');
        
        // Call summarize API
        const formData = new FormData();
        formData.append('text', text);
        
        const response = await fetch(`${API_BASE_URL}/api/summarize`, {
            method: 'POST',
            body: formData,
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Store and display results
        state.operationResults.summarize = data;
        document.getElementById('headline-result').textContent = data.headline;
        document.getElementById('short-summary-result').textContent = data.short_summary;
        document.getElementById('medium-summary-result').textContent = data.medium_summary;
        
        // Show results section
        document.getElementById('summarize-results').classList.remove('hidden');
        document.getElementById('results-section').classList.remove('hidden');
        
    } catch (error) {
        console.error('Summarization error:', error);
        showError('main-error', error.message || 'Summarization failed');
    } finally {
        setButtonLoading('summarize-btn', false);
    }
}

/**
 * Handle Translate button click
 */
async function handleTranslate() {
    hideError('main-error');
    
    const textPromise = getArticleText();
    if (!textPromise) {
        showError('main-error', 'Please provide article text or upload a file');
        return;
    }
    
    try {
        const text = await textPromise;
        
        // Store article text for later use
        state.articleText = text;
        state.articleSource = { type: state.currentInputType, value: state.currentInputType === 'file' ? document.getElementById('article-file').files[0].name : 'inline' };
        
        // Set loading state
        setButtonLoading('translate-btn', true);
        
        // Hide previous results
        document.getElementById('translate-results').classList.add('hidden');
        document.getElementById('results-section').classList.add('hidden');
        
        // Default to English -> Bahasa Melayu
        const sourceLang = 'en';
        const targetLang = 'bm';
        
        const response = await fetch(`${API_BASE_URL}/api/translate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text,
                source_lang: sourceLang,
                target_lang: targetLang,
            }),
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Store and display results
        state.operationResults.translate = data;
        document.getElementById('translated-text-result').textContent = data.translated_text;
        document.getElementById('maintained-tone-result').textContent =
            data.maintained_tone === 'news' ? 'Formal News Style' : data.maintained_tone;
        
        // Show results section
        document.getElementById('translate-results').classList.remove('hidden');
        document.getElementById('results-section').classList.remove('hidden');
        
    } catch (error) {
        console.error('Translation error:', error);
        showError('main-error', error.message || 'Translation failed');
    } finally {
        setButtonLoading('translate-btn', false);
    }
}

/**
 * Handle Load to Chat button click
 */
async function handleLoadChat() {
    hideError('main-error');
    
    const textPromise = getArticleText();
    if (!textPromise) {
        showError('main-error', 'Please provide article text or upload a file');
        return;
    }
    
    try {
        const text = await textPromise;
        
        // Store article text for later use
        state.articleText = text;
        const sourceType = state.currentInputType;
        const sourceValue = sourceType === 'file' ? document.getElementById('article-file').files[0].name : 'inline';
        state.articleSource = { type: sourceType, value: sourceValue };
        
        // Set loading state
        setButtonLoading('load-chat-btn', true);
        
        // Generate or reuse session ID
        let sessionId = state.sessionId || generateUUID();
        
        const response = await fetch(`${API_BASE_URL}/api/chat/load`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                session_id: sessionId,
                text: text,
                source_type: sourceType,
                source_value: sourceValue,
            }),
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Save session
        saveSessionId(data.session_id || sessionId);
        
        // Show chat section and scroll to it
        showChatSession(data.session_id || sessionId);
        
    } catch (error) {
        console.error('Chat load error:', error);
        showError('main-error', error.message || 'Chat loading failed');
    } finally {
        setButtonLoading('load-chat-btn', false);
    }
}

// ============================================
// Chat Feature
// ============================================

function showChatSession(sessionId) {
    const chatSection = document.getElementById('chat-section');
    chatSection.classList.remove('hidden');
    
    document.getElementById('chat-session-info').classList.remove('hidden');
    document.getElementById('session-id-display').textContent = sessionId;
    document.getElementById('chat-window').classList.remove('hidden');
    document.getElementById('chat-input-container').classList.remove('hidden');
    
    // Clear previous messages
    document.getElementById('chat-messages').innerHTML = '';
    
    // Add welcome message
    addChatMessage('assistant', 'Article loaded! Ask me anything about it.');
    
    // Scroll to chat section
    chatSection.scrollIntoView({ behavior: 'smooth' });
}

function addChatMessage(role, content) {
    const messagesContainer = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${role}`;
    
    const avatar = role === 'assistant' ? '🤖' : '👤';
    
    messageDiv.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">
            <p>${escapeHtml(content)}</p>
        </div>
    `;
    
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

async function handleChatSend() {
    const questionInput = document.getElementById('chat-question');
    const question = questionInput.value.trim();
    
    if (!question) {
        return;
    }
    
    if (!state.sessionId) {
        showError('chat-error', 'No active session. Please load an article first.');
        return;
    }
    
    // Add user message
    addChatMessage('user', question);
    questionInput.value = '';
    
    // Show loading on send button
    const sendBtn = document.getElementById('chat-send-btn');
    const sendLoader = sendBtn.querySelector('.send-loader');
    const sendText = sendBtn.querySelector('span:not(.send-loader)');
    sendBtn.disabled = true;
    sendLoader?.classList.remove('hidden');
    sendText.textContent = 'Sending...';
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                session_id: state.sessionId,
                question: question,
            }),
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Add assistant response
        addChatMessage('assistant', data.answer);
        
    } catch (error) {
        console.error('Chat error:', error);
        showError('chat-error', error.message || 'Failed to get answer');
    } finally {
        sendBtn.disabled = false;
        sendLoader?.classList.add('hidden');
        sendText.textContent = 'Send';
    }
}

async function handleClearSession() {
    if (state.sessionId) {
        try {
            await fetch(`${API_BASE_URL}/api/chat/session/${state.sessionId}`, {
                method: 'DELETE',
            });
        } catch (error) {
            console.error('Failed to delete session:', error);
        }
    }
    
    clearSessionId();
    document.getElementById('chat-section').classList.add('hidden');
    hideError('chat-error');
}

async function checkSessionExists() {
    const savedSessionId = loadSessionId();
    if (savedSessionId) {
        try {
            const response = await fetch(`${API_BASE_URL}/api/chat/session/${savedSessionId}/exists`);
            if (response.ok) {
                const data = await response.json();
                if (data.exists) {
                    state.sessionId = savedSessionId;
                    // Don't auto-show chat, wait for user to load article
                } else {
                    clearSessionId();
                }
            }
        } catch (error) {
            console.error('Session check failed:', error);
            clearSessionId();
        }
    }
}

// ============================================
// Event Listeners Initialization
// ============================================

function initEventListeners() {
    // Action buttons
    document.getElementById('summarize-btn').addEventListener('click', handleSummarize);
    document.getElementById('translate-btn').addEventListener('click', handleTranslate);
    document.getElementById('load-chat-btn').addEventListener('click', handleLoadChat);
    
    // Chat
    document.getElementById('chat-send-btn').addEventListener('click', handleChatSend);
    document.getElementById('clear-session-btn').addEventListener('click', handleClearSession);
    
    // Allow Enter key to send chat message
    document.getElementById('chat-question').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            handleChatSend();
        }
    });
}

// ============================================
// Application Initialization
// ============================================

function init() {
    initInputTabs();
    initEventListeners();
    checkSessionExists();
    
    console.log('AI News Companion initialized');
}

// Start the application when DOM is ready
document.addEventListener('DOMContentLoaded', init);
