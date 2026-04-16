/**
 * AI News Companion - Frontend Application
 * Handles API integration for Summarization, Translation, and Chat features.
 */

// ============================================
// Configuration
// ============================================
const API_BASE_URL = 'http://localhost:8000';

// ============================================
// State Management
// ============================================
const state = {
    currentTab: 'summarize',
    currentInputType: 'url',
    currentChatInputType: 'url',
    sessionId: null,
    lastArticleText: null,
    lastArticleSource: null,
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
        if (btnText) btnText.textContent = 'Loading...';
    } else {
        button.disabled = false;
        loader?.classList.add('hidden');
        if (btnText) btnText.textContent = buttonId.includes('summarize') ? 'Summarize' :
                                            buttonId.includes('translate') ? 'Translate' :
                                            buttonId.includes('chat-load') ? 'Load Article' : 'Send';
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

// ============================================
// Tab Navigation
// ============================================

function initTabNavigation() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    
    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.dataset.tab;
            
            // Update active tab button
            tabButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Update active tab content
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });
            document.getElementById(tabId).classList.add('active');
            
            state.currentTab = tabId;
        });
    });
}

// ============================================
// Input Panel Switching (Summarization)
// ============================================

function initSummarizeInputTabs() {
    const tabButtons = document.querySelectorAll('#summarize .input-tab-btn');
    
    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const inputType = btn.dataset.input;
            
            // Update active button
            tabButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Update active panel
            document.querySelectorAll('#summarize .input-panel').forEach(panel => {
                panel.classList.remove('active');
            });
            document.getElementById(`${inputType}-panel`).classList.add('active');
            
            state.currentInputType = inputType;
        });
    });
}

// ============================================
// Input Panel Switching (Chat Load)
// ============================================

function initChatInputTabs() {
    const tabButtons = document.querySelectorAll('#chat .input-tab-btn');
    
    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const inputType = btn.dataset.chatInput;
            
            // Update active button
            tabButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Update active panel
            document.querySelectorAll('#chat .input-panel').forEach(panel => {
                panel.classList.remove('active');
            });
            document.getElementById(`chat-${inputType}-panel`).classList.add('active');
            
            state.currentChatInputType = inputType;
        });
    });
}

// ============================================
// Summarization Feature
// ============================================

async function handleSummarize() {
    hideError('summarize-error');
    document.getElementById('summarize-results').classList.add('hidden');
    
    let url, file, text;
    
    // Get input based on current type
    if (state.currentInputType === 'url') {
        url = document.getElementById('article-url').value.trim();
        if (!url) {
            showError('summarize-error', 'Please enter an article URL');
            return;
        }
    } else if (state.currentInputType === 'file') {
        file = document.getElementById('article-file').files[0];
        if (!file) {
            showError('summarize-error', 'Please select a file to upload');
            return;
        }
    } else if (state.currentInputType === 'text') {
        text = document.getElementById('article-text').value.trim();
        if (!text) {
            showError('summarize-error', 'Please paste article text');
            return;
        }
    }
    
    setButtonLoading('summarize-btn', true);
    
    try {
        const formData = new FormData();
        
        if (url) {
            formData.append('url', url);
            state.lastArticleSource = { type: 'url', value: url };
        } else if (file) {
            formData.append('file', file);
            state.lastArticleSource = { type: 'file', value: file.name };
        } else if (text) {
            formData.append('text', text);
            state.lastArticleText = text;
            state.lastArticleSource = { type: 'text', value: 'inline' };
        }
        
        const response = await fetch(`${API_BASE_URL}/api/summarize`, {
            method: 'POST',
            body: formData,
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Display results
        document.getElementById('headline-result').textContent = data.headline;
        document.getElementById('short-summary-result').textContent = data.short_summary;
        document.getElementById('medium-summary-result').textContent = data.medium_summary;
        document.getElementById('summarize-results').classList.remove('hidden');
        
        // Store article text for chat loading
        if (!state.lastArticleText && text) {
            state.lastArticleText = text;
        }
        
    } catch (error) {
        console.error('Summarization error:', error);
        showError('summarize-error', error.message || 'Failed to summarize article');
    } finally {
        setButtonLoading('summarize-btn', false);
    }
}

// ============================================
// Translation Feature
// ============================================

async function handleTranslate() {
    hideError('translate-error');
    document.getElementById('translate-results').classList.add('hidden');
    
    const text = document.getElementById('translate-input').value.trim();
    if (!text) {
        showError('translate-error', 'Please enter text to translate');
        return;
    }
    
    const sourceLang = document.getElementById('source-lang').value;
    const targetLang = document.getElementById('target-lang').value;
    
    setButtonLoading('translate-btn', true);
    
    try {
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
        
        // Display results
        document.getElementById('translated-text-result').textContent = data.translated_text;
        document.getElementById('maintained-tone-result').textContent = 
            data.maintained_tone === 'news' ? 'Formal News Style' : data.maintained_tone;
        document.getElementById('translate-results').classList.remove('hidden');
        
    } catch (error) {
        console.error('Translation error:', error);
        showError('translate-error', error.message || 'Failed to translate text');
    } finally {
        setButtonLoading('translate-btn', false);
    }
}

function initLanguageSwap() {
    document.getElementById('swap-langs-btn').addEventListener('click', () => {
        const sourceLang = document.getElementById('source-lang');
        const targetLang = document.getElementById('target-lang');
        
        const temp = sourceLang.value;
        sourceLang.value = targetLang.value;
        targetLang.value = temp;
    });
}

// ============================================
// Chat Feature
// ============================================

async function handleChatLoad() {
    hideError('chat-error');
    
    let url, file, text;
    
    // Get input based on current type
    if (state.currentChatInputType === 'url') {
        url = document.getElementById('chat-article-url').value.trim();
        if (!url) {
            showError('chat-error', 'Please enter an article URL');
            return;
        }
    } else if (state.currentChatInputType === 'file') {
        file = document.getElementById('chat-article-file').files[0];
        if (!file) {
            showError('chat-error', 'Please select a file to upload');
            return;
        }
    } else if (state.currentChatInputType === 'text') {
        text = document.getElementById('chat-article-text').value.trim();
        if (!text) {
            showError('chat-error', 'Please paste article text');
            return;
        }
    }
    
    setButtonLoading('chat-load-btn', true);
    
    try {
        let articleText = text;
        let sourceType = state.currentChatInputType;
        let sourceValue = 'inline';
        
        // If URL or file, we need to get the text content first
        // For now, we'll send the URL/file info and let backend handle it
        if (url) {
            // For URL, we'll need to fetch content or let backend handle it
            // Since backend expects text, we'll send URL as source marker
            articleText = `[Article from URL: ${url}]`;
            sourceType = 'url';
            sourceValue = url;
        } else if (file) {
            // For file, read as text
            articleText = await readFileAsText(file);
            sourceType = 'file';
            sourceValue = file.name;
        }
        
        // Generate or reuse session ID
        let sessionId = state.sessionId || generateUUID();
        
        const response = await fetch(`${API_BASE_URL}/api/chat/load`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                session_id: sessionId,
                text: articleText,
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
        state.lastArticleText = articleText;
        state.lastArticleSource = { type: sourceType, value: sourceValue };
        
        // Update UI
        showChatSession(data.session_id || sessionId);
        
    } catch (error) {
        console.error('Chat load error:', error);
        showError('chat-error', error.message || 'Failed to load article');
    } finally {
        setButtonLoading('chat-load-btn', false);
    }
}

function readFileAsText(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => resolve(e.target.result);
        reader.onerror = (e) => reject(new Error('Failed to read file'));
        reader.readAsText(file);
    });
}

function showChatSession(sessionId) {
    document.getElementById('chat-load-panel').classList.add('hidden');
    document.getElementById('chat-window').classList.remove('hidden');
    document.getElementById('chat-input-container').classList.remove('hidden');
    document.getElementById('chat-session-info').classList.remove('hidden');
    document.getElementById('session-id-display').textContent = sessionId;
    
    // Clear previous messages
    document.getElementById('chat-messages').innerHTML = '';
    
    // Add welcome message
    addChatMessage('assistant', 'Article loaded! Ask me anything about it.');
}

function hideChatSession() {
    document.getElementById('chat-load-panel').classList.remove('hidden');
    document.getElementById('chat-window').classList.add('hidden');
    document.getElementById('chat-input-container').classList.add('hidden');
    document.getElementById('chat-session-info').classList.add('hidden');
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

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
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
    hideChatSession();
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
                    showChatSession(savedSessionId);
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
// Load to Chat Button (from Summarization)
// ============================================

function initLoadToChat() {
    document.getElementById('load-to-chat-btn').addEventListener('click', () => {
        // Switch to chat tab
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelector('[data-tab="chat"]').classList.add('active');
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        document.getElementById('chat').classList.add('active');
        
        // Pre-fill chat text input with article text if available
        if (state.lastArticleText) {
            document.getElementById('chat-article-text').value = state.lastArticleText;
            // Switch to text input
            document.querySelectorAll('#chat .input-tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelector('[data-chat-input="text"]').classList.add('active');
            document.querySelectorAll('#chat .input-panel').forEach(p => p.classList.remove('active'));
            document.getElementById('chat-text-panel').classList.add('active');
        }
    });
}

// ============================================
// Event Listeners Initialization
// ============================================

function initEventListeners() {
    // Summarization
    document.getElementById('summarize-btn').addEventListener('click', handleSummarize);
    
    // Translation
    document.getElementById('translate-btn').addEventListener('click', handleTranslate);
    initLanguageSwap();
    
    // Chat
    document.getElementById('chat-load-btn').addEventListener('click', handleChatLoad);
    document.getElementById('chat-send-btn').addEventListener('click', handleChatSend);
    document.getElementById('clear-session-btn').addEventListener('click', handleClearSession);
    
    // Allow Enter key to send chat message
    document.getElementById('chat-question').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            handleChatSend();
        }
    });
    
    // Load to Chat from summarization
    initLoadToChat();
}

// ============================================
// Application Initialization
// ============================================

function init() {
    initTabNavigation();
    initSummarizeInputTabs();
    initChatInputTabs();
    initEventListeners();
    checkSessionExists();
    
    console.log('AI News Companion initialized');
}

// Start the application when DOM is ready
document.addEventListener('DOMContentLoaded', init);
