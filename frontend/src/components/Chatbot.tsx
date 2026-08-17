import React, { useState, useRef, useEffect } from 'react';
import { Button, Card, Form, InputGroup, Spinner } from 'react-bootstrap';
import axios from 'axios';

interface Message {
  text: string;
  sender: 'user' | 'bot';
  source?: string;
}

const Chatbot: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    { text: "Hi! Ask me anything about deepfakes, detection, or ethics.", sender: 'bot' }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || loading) return;
    const userMsg: Message = { text: input, sender: 'user' };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      const response = await axios.post('http://localhost:5000/chat', { query: input });
      const botReply = response.data.response || "I'm not sure. Please ask something else.";
      const source = response.data.source || '';
      setMessages(prev => [...prev, { text: botReply, sender: 'bot', source }]);
    } catch (err) {
      setMessages(prev => [...prev, { text: "Sorry, I'm having trouble right now. Please try again.", sender: 'bot' }]);
    }
    setLoading(false);
  };

  return (
    <>
      <Button
        variant="primary"
        className="rounded-circle position-fixed"
        style={{ bottom: '20px', right: '20px', zIndex: 999, width: '60px', height: '60px' }}
        onClick={() => setIsOpen(!isOpen)}
      >
        {isOpen ? <i className="bi bi-x-lg"></i> : <i className="bi bi-chat-dots"></i>}
      </Button>

      {isOpen && (
        <Card
          className="shadow-lg position-fixed"
          style={{ bottom: '100px', right: '20px', zIndex: 999, width: '380px', maxHeight: '550px' }}
        >
          <Card.Header className="bg-primary text-white d-flex justify-content-between align-items-center">
            <span><i className="bi bi-robot me-2"></i>Deepfake Assistant</span>
            <small className="text-light">Powered by AI</small>
          </Card.Header>
          <Card.Body style={{ overflowY: 'auto', maxHeight: '380px' }}>
            {messages.map((msg, idx) => (
              <div key={idx} className={`d-flex ${msg.sender === 'user' ? 'justify-content-end' : 'justify-content-start'} mb-2`}>
                <div
                  className={`p-2 rounded-3 ${msg.sender === 'user' ? 'bg-primary text-white' : 'bg-light border'}`}
                  style={{ maxWidth: '80%' }}
                >
                  {msg.text}
                  {msg.source && msg.source === 'ai' && (
                    <div className="mt-1"><small className="text-muted fst-italic">AI‑generated</small></div>
                  )}
                </div>
              </div>
            ))}
            {loading && (
              <div className="d-flex justify-content-start mb-2">
                <div className="p-2 bg-light border rounded-3">
                  <Spinner animation="grow" size="sm" />
                  <Spinner animation="grow" size="sm" className="ms-1" />
                  <Spinner animation="grow" size="sm" className="ms-1" />
                </div>
              </div>
            )}
            <div ref={chatEndRef} />
          </Card.Body>
          <Card.Footer>
            <InputGroup>
              <Form.Control
                type="text"
                placeholder="Ask about deepfakes..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSend()}
                disabled={loading}
              />
              <Button variant="primary" onClick={handleSend} disabled={loading}>
                {loading ? <Spinner animation="border" size="sm" /> : <i className="bi bi-send"></i>}
              </Button>
            </InputGroup>
          </Card.Footer>
        </Card>
      )}
    </>
  );
};

export default Chatbot;