import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { 
  Send, 
  RotateCcw, 
  Loader2, 
  BookOpen, 
  AlertCircle,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  FileText
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import './App.css';

// Configure axios
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const api = axios.create({
  baseURL: API_URL,
  withCredentials: false
});

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [error, setError] = useState(null);
  const messagesEndRef = useRef(null);

  // Create session on mount
  useEffect(() => {
    createSession();
  }, []);

  // Auto-scroll to bottom
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const createSession = async () => {
    try {
      const response = await api.post('/session');
      setSessionId(response.data.session_id);
      setError(null);
    } catch (err) {
      setError('Failed to create session. Please check your connection.');
      console.error('Session creation error:', err);
    }
  };

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage = {
      role: 'user',
      content: input.trim(),
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);
    setError(null);

    try {
      const response = await api.post('/query', {
        question: userMessage.content,
        session_id: sessionId
      });

      const assistantMessage = {
        role: 'assistant',
        content: response.data.answer,
        citations: response.data.citations || [],
        confidence: response.data.confidence,
        chunks_used: response.data.chunks_used,
        timestamp: new Date().toISOString()
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (err) {
      setError('Failed to get response. Please try again.');
      console.error('Query error:', err);
      
      const errorMessage = {
        role: 'assistant',
        content: 'Sorry, I encountered an error processing your request. Please try again.',
        confidence: 'error',
        timestamp: new Date().toISOString()
      };
      
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const resetConversation = async () => {
    if (window.confirm('Clear conversation history?')) {
      try {
        if (sessionId) {
          await api.delete(`/session/${sessionId}`);
        }
        setMessages([]);
        await createSession();
      } catch (err) {
        console.error('Reset error:', err);
      }
    }
  };

  const getConfidenceIcon = (confidence) => {
    switch (confidence?.toLowerCase()) {
      case 'high':
        return <CheckCircle2 className="confidence-icon high" />;
      case 'medium':
        return <AlertTriangle className="confidence-icon medium" />;
      case 'low':
        return <XCircle className="confidence-icon low" />;
      default:
        return <AlertCircle className="confidence-icon" />;
    }
  };

  return (
    <div className="app">
      <div className="chat-container">
        {/* Header */}
        <header className="chat-header">
          <div className="header-content">
            <div className="header-title">
              <BookOpen className="header-icon" />
              <div>
                <h1>Internal Knowledge Base</h1>
                <p>Confluence Documentation Assistant</p>
              </div>
            </div>
            <button 
              className="btn-reset" 
              onClick={resetConversation}
              title="Reset conversation"
            >
              <RotateCcw size={18} />
              <span>Reset</span>
            </button>
          </div>
        </header>

        {/* Error Banner */}
        {error && (
          <div className="error-banner">
            <AlertCircle size={16} />
            <span>{error}</span>
            <button onClick={() => setError(null)}>✕</button>
          </div>
        )}

        {/* Messages */}
        <div className="messages-container">
          {messages.length === 0 ? (
            <div className="welcome-message">
              <BookOpen size={48} className="welcome-icon" />
              <h2>Welcome to the Knowledge Base</h2>
              <p>Ask me anything about our internal documentation.</p>
              <div className="example-queries">
                <span>Try asking:</span>
                <button onClick={() => setInput('What is bare metal?')}>
                  "What is bare metal?"
                </button>
                <button onClick={() => setInput('What are the pricing options?')}>
                  "What are the pricing options?"
                </button>
              </div>
            </div>
          ) : (
            messages.map((msg, idx) => (
              <Message 
                key={idx} 
                message={msg} 
                getConfidenceIcon={getConfidenceIcon}
              />
            ))
          )}
          
          {loading && (
            <div className="message assistant">
              <div className="message-bubble loading-bubble">
                <Loader2 className="spinner" />
                <span>Processing your question...</span>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <form className="input-container" onSubmit={sendMessage}>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question about our documentation..."
            disabled={loading}
            autoFocus
          />
          <button 
            type="submit" 
            disabled={loading || !input.trim()}
            className="btn-send"
          >
            {loading ? (
              <Loader2 className="spinner" size={20} />
            ) : (
              <Send size={20} />
            )}
          </button>
        </form>

        {/* Footer */}
        <footer className="chat-footer">
          🔒 Internal Use Only | All conversations are logged
        </footer>
      </div>
    </div>
  );
}

function Message({ message, getConfidenceIcon }) {
  const [showCitations, setShowCitations] = useState(false);

  return (
    <div className={`message ${message.role}`}>
      <div className="message-bubble">
        <div className="message-content">
          <ReactMarkdown>{message.content}</ReactMarkdown>
        </div>



        <div className="message-timestamp">
          {new Date(message.timestamp).toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit'
          })}
        </div>
      </div>
    </div>
  );
}

export default App;