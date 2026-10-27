import React, { useEffect, useState } from 'react';
import '../styles/pages/ChatbotPage.css';
import { chatbotAPI, authAPI } from '../api';

const ChatbotPage = () => {
  const [loading, setLoading] = useState(true);
  const [sessionId, setSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [error, setError] = useState('');

  useEffect(() => {
    const init = async () => {
      try {
        const token = localStorage.getItem('auth_token');
        if (!token) {
          setError('Please log in to use the medical chatbot.');
          setLoading(false);
          return;
        }

        const user = await authAPI.getCurrentUser();
        const res = await chatbotAPI.startChatSession(user.id);
        setSessionId(res.session_id);
        setMessages(prev => [...prev, { type: 'bot', content: res.message }]);
      } catch (e) {
        console.error('Chat init error', e);
        if (e.response?.status === 401) {
          setError('Please log in to use the medical chatbot.');
        } else {
          setError(e.response?.data?.detail || 'Chat initialization failed');
        }
      } finally {
        setLoading(false);
      }
    };
    init();
  }, []);

  const sendMessage = async () => {
    if (!input.trim()) return;
    setError('');
    const text = input.trim();
    setMessages(prev => [...prev, { type: 'user', content: text }]);
    setInput('');
    try {
      const res = await chatbotAPI.sendMessage(sessionId, text);
      setMessages(prev => [...prev, { type: 'bot', content: res.message }]);
    } catch (e) {
      console.error('Send message error', e);
      setError(e.response?.data?.detail || 'Failed to send message');
    }
  };

  return (
    <div className="chatbot-page">
      <h1>💬 Medical AI Chatbot</h1>
      <p>Get instant answers to your medical questions</p>

      <div className="chatbot-container">
        {loading ? (
          <div className="placeholder">Connecting to chatbot...</div>
        ) : error ? (
          <div className="error-message">{error}</div>
        ) : (
          <>
            <div className="chat-window">
              {messages.map((m, idx) => (
                <div key={idx} className={`chat-message ${m.type}`}>
                  <div className="message-content">{m.content}</div>
                </div>
              ))}
            </div>

            <div className="chat-input">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Type your question..."
                onKeyDown={(e) => { if (e.key === 'Enter') sendMessage(); }}
              />
              <button onClick={sendMessage} className="btn-send">Send</button>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default ChatbotPage;
