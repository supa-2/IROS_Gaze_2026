import React, { useState, useRef, useEffect } from 'react';
import { Send, Paperclip, Image as ImageIcon, Menu, X, Settings, User, LogOut, Plus, Trash2 } from 'lucide-react';

const OpenAIApp = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [selectedImage, setSelectedImage] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);

  // Scroll to bottom when messages change
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(scrollToBottom, [messages]);

  // Handle image upload
  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setSelectedImage({
          file,
          preview: reader.result,
          name: file.name,
        });
      };
      reader.readAsDataURL(file);
    }
  };

  // Handle message submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() && !selectedImage) return;

    // Add user message
    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: input,
      timestamp: new Date(),
      image: selectedImage?.preview,
    };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsTyping(true);
    setSelectedImage(null);

    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1500));

      // Add assistant response
      const assistantMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: `I understand you want to create an interface similar to OpenAI's ChatGPT. Here's what I can do:

1. **Text Generation**: I can generate text based on your prompts
2. **Image Analysis**: I can analyze images and provide insights
3. **Code Generation**: I can write code for various programming languages
4. **Problem Solving**: I can help solve problems and provide explanations

What would you like to do first?`,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error generating response:', error);
    } finally {
      setIsTyping(false);
    }
  };

  // Clear all messages
  const clearMessages = () => {
    setMessages([]);
  };

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar */}
      <div className={`${sidebarOpen ? 'w-64' : 'w-0'} bg-white border-r border-gray-200 transition-all duration-300 overflow-hidden`}>
        <div className="p-4">
          <button className="w-full mb-4 p-3 bg-gray-100 hover:bg-gray-200 rounded-lg flex items-center justify-center gap-2 text-sm font-medium">
            <Plus size={18} />
            New chat
          </button>

          <div className="space-y-2 mb-6">
            <div className="text-xs text-gray-500 px-3 py-2">Recent</div>
            {messages.length > 0 ? (
              <div className="p-3 bg-gray-100 rounded-lg cursor-pointer">
                <div className="text-sm font-medium">Current Chat</div>
                <div className="text-xs text-gray-500 truncate">{messages[0]?.content || 'No messages yet'}</div>
              </div>
            ) : (
              <div className="p-3 text-gray-400">
                <div className="text-sm">No recent chats</div>
              </div>
            )}
          </div>

          <div className="absolute bottom-0 left-0 w-64 p-4 border-t border-gray-200">
            <button className="w-full p-3 hover:bg-gray-100 rounded-lg flex items-center justify-between text-sm">
              <div className="flex items-center gap-2">
                <User size={18} />
                <span>User</span>
              </div>
              <Settings size={18} />
            </button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <header className="bg-white border-b border-gray-200 p-4 flex items-center justify-between">
          <button 
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-2 hover:bg-gray-100 rounded-lg"
          >
            <Menu size={20} />
          </button>
          <div className="flex items-center gap-2">
            <button 
              onClick={clearMessages}
              className="p-2 hover:bg-gray-100 rounded-lg text-gray-500"
            >
              <Trash2 size={20} />
            </button>
          </div>
        </header>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6 space-y-8">
          {messages.length === 0 ? (
            <div className="max-w-2xl mx-auto text-center mt-20">
              <div className="flex justify-center mb-6">
                <div className="w-20 h-20 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
                  <Send size={40} className="text-white" />
                </div>
              </div>
              <h1 className="text-3xl font-bold text-gray-900 mb-4">Chat with IROS Gaze</h1>
              <p className="text-gray-500 mb-8">
                I can help you with text generation, image analysis, code writing, and more.
              </p>
            </div>
          ) : (
            messages.map((message) => (
              <div key={message.id} className={`max-w-3xl mx-auto ${message.role === 'user' ? 'text-right' : 'text-left'}`}>
                <div className={`inline-block p-4 rounded-lg ${message.role === 'user' ? 'bg-blue-500 text-white' : 'bg-gray-200 text-gray-900'}`}>
                  {message.image && (
                    <div className="mb-4 rounded overflow-hidden">
                      <img src={message.image} alt="Uploaded" className="max-w-full h-auto max-h-64 object-contain" />
                    </div>
                  )}
                  <p className="whitespace-pre-wrap">{message.content}</p>
                </div>
              </div>
            ))
          )}

          {isTyping && (
            <div className="max-w-3xl mx-auto text-left">
              <div className="inline-block p-4 bg-gray-200 rounded-lg">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                  <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                  <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="border-t border-gray-200 p-4 bg-white">
          {selectedImage && (
            <div className="mb-4 p-3 bg-gray-100 rounded-lg flex items-center gap-3">
              <img src={selectedImage.preview} alt="Preview" className="w-12 h-12 object-cover rounded" />
              <div className="flex-1 min-w-0">
                <div className="text-sm font-medium truncate">{selectedImage.name}</div>
                <div className="text-xs text-gray-500">Ready to send</div>
              </div>
              <button 
                onClick={() => setSelectedImage(null)}
                className="p-2 hover:bg-gray-200 rounded-lg"
              >
                <X size={18} />
              </button>
            </div>
          )}

          <form onSubmit={handleSubmit} className="max-w-3xl mx-auto">
            <div className="flex items-center gap-4">
              <input
                ref={fileInputRef}
                type="file"
                onChange={handleImageUpload}
                className="hidden"
                accept="image/*"
              />
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="p-3 hover:bg-gray-100 rounded-lg text-gray-500"
              >
                <ImageIcon size={20} />
              </button>

              <div className="flex-1 relative">
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleSubmit(e)}
                  placeholder="Message IROS Gaze..."
                  className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>

              <button
                type="submit"
                disabled={!input.trim() && !selectedImage}
                className="p-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors disabled:bg-gray-300"
              >
                <Send size={20} />
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default OpenAIApp;