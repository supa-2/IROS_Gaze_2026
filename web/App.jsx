import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Send,
  Image as ImageIcon,
  X,
  Plus,
  Minimize2,
  Maximize2,
  Trash2,
  Download,
  Eye,
  Activity,
  Loader2,
  Sparkles,
  MessageSquare,
  Settings,
  Moon,
  Sun,
  Upload,
  FileImage,
  Network,
  Brain,
  Wand2,
} from 'lucide-react';

// API Base URL
const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:5000/api';

// Main App Component
function App() {
  const [messages, setMessages] = useState([
    {
      id: 1,
      role: 'assistant',
      content: 'Welcome to IROS Gaze! 👁️\n\nI can help you:\n• 🧠 Analyze images with Qwen Vision AI\n• 🔥 Generate eye-tracking heatmaps\n• 📈 Create trajectory visualizations\n• 💬 Chat about gaze patterns and predictions\n\nTry uploading an image or typing a message to get started!',
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [uploadedImage, setUploadedImage] = useState(null);
  const [uploadedImagePreview, setUploadedImagePreview] = useState(null);
  const [showSettings, setShowSettings] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [analysisType, setAnalysisType] = useState('both'); // 'heatmap', 'trajectory', 'both'
  const [viewMode, setViewMode] = useState('chat'); // 'chat', 'split', 'full'

  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);
  const imageContainerRef = useRef(null);

  // Auto-scroll to bottom
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, analysisResult]);

  // Handle file upload
  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setUploadedImage(file);
        setUploadedImagePreview(reader.result);
        setAnalysisResult(null);
      };
      reader.readAsDataURL(file);
    }
  };

  // Remove uploaded image
  const removeImage = () => {
    setUploadedImage(null);
    setUploadedImagePreview(null);
    setAnalysisResult(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Analyze image
  const analyzeImage = async () => {
    if (!uploadedImage) return;

    setIsLoading(true);

    const formData = new FormData();
    formData.append('image', uploadedImage);
    formData.append('options', JSON.stringify({ type: analysisType }));

    try {
      // Qwen Vision Analysis
      if (analysisType === 'qwen-vision') {
        const response = await fetch(`${API_BASE}/analyze-image`, {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) throw new Error('Vision analysis failed');

        const data = await response.json();

        // Add analysis result as a message
        const resultMessage = {
          id: Date.now(),
          role: 'assistant',
          content: data.analysis || 'Analysis complete.',
          timestamp: new Date(),
          image: uploadedImagePreview,
        };
        setMessages((prev) => [...prev, resultMessage]);
        setUploadedImage(null);
        setUploadedImagePreview(null);
        setIsLoading(false);
        return;
      }

      // Standard visualization endpoints
      const endpoint =
        analysisType === 'heatmap'
          ? '/heatmap'
          : analysisType === 'trajectory'
            ? '/trajectory'
            : '/analyze';

      const response = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Analysis failed');

      const data = await response.json();
      setAnalysisResult(data);

      // Add result message
      const resultMessage = {
        id: Date.now(),
        role: 'assistant',
        content: `Analysis complete! Generated ${analysisType} visualization for your image.`,
        timestamp: new Date(),
        images: data,
      };
      setMessages((prev) => [...prev, resultMessage]);
    } catch (error) {
      console.error('Analysis error:', error);
      const errorMessage = {
        id: Date.now(),
        role: 'assistant',
        content: `Sorry, there was an error analyzing the image: ${error.message}`,
        timestamp: new Date(),
        isError: true,
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // Send message
  const sendMessage = async () => {
    if (!input.trim() && !uploadedImage) return;

    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: input || 'Analyze this image',
      timestamp: new Date(),
      image: uploadedImagePreview,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    // If image is uploaded, analyze it
    if (uploadedImage) {
      await analyzeImage();
      return;
    }

    // Otherwise, send chat message
    try {
      const response = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMessage.content,
          history: messages.slice(-10).map((m) => ({
            role: m.role,
            content: m.content,
          })),
        }),
      });

      if (!response.ok) throw new Error('Chat failed');

      const data = await response.json();

      const assistantMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: data.content,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Chat error:', error);
      const errorMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: `Sorry, there was an error: ${error.message}`,
        timestamp: new Date(),
        isError: true,
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="h-screen flex flex-col bg-[#0a0a0a] text-white">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-4 border-b border-[#2a2a2a] bg-[#111111]/80 backdrop-blur-xl">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-[#00d1ff] to-[#0066ff] flex items-center justify-center">
            <Eye className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold">IROS Gaze</h1>
            <p className="text-xs text-gray-500">Eye Tracking Visualization</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* View mode toggle */}
          <div className="hidden md:flex items-center bg-[#1a1a1a] rounded-lg p-1 border border-[#2a2a2a]">
            <button
              onClick={() => setViewMode('chat')}
              className={`px-3 py-1.5 rounded-md text-sm font-medium transition-all ${
                viewMode === 'chat'
                  ? 'bg-[#00d1ff] text-black'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              <MessageSquare className="w-4 h-4 inline mr-1" />
              Chat
            </button>
            <button
              onClick={() => setViewMode('split')}
              className={`px-3 py-1.5 rounded-md text-sm font-medium transition-all ${
                viewMode === 'split'
                  ? 'bg-[#00d1ff] text-black'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              <Layout className="w-4 h-4 inline mr-1" />
              Split
            </button>
            <button
              onClick={() => setViewMode('full')}
              className={`px-3 py-1.5 rounded-md text-sm font-medium transition-all ${
                viewMode === 'full'
                  ? 'bg-[#00d1ff] text-black'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              <Maximize2 className="w-4 h-4 inline mr-1" />
              Full
            </button>
          </div>

          <button
            onClick={() => setShowSettings(!showSettings)}
            className="p-2 rounded-lg hover:bg-[#1a1a1a] transition-colors"
          >
            <Settings className="w-5 h-5 text-gray-400" />
          </button>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Chat Area */}
        <div
          className={`flex flex-col ${
            viewMode === 'full' ? 'hidden' : viewMode === 'split' ? 'w-1/2' : 'w-full'
          } border-r border-[#2a2a2a]`}
        >
          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.map((message) => (
              <MessageBubble key={message.id} message={message} />
            ))}
            {isLoading && (
              <div className="flex items-center gap-3 text-gray-400">
                <Loader2 className="w-5 h-5 animate-spin text-[#00d1ff]" />
                <span className="loading-dots">
                  <span>Analyzing</span>
                  <span>.</span>
                  <span>.</span>
                  <span>.</span>
                </span>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <div className="p-4 border-t border-[#2a2a2a] bg-[#111111]/50">
            {/* Image Preview */}
            {uploadedImagePreview && (
              <div className="mb-3 p-3 bg-[#1a1a1a] rounded-xl border border-[#2a2a2a]">
                <div className="flex items-start gap-3">
                  <div className="relative w-16 h-16 rounded-lg overflow-hidden flex-shrink-0">
                    <img
                      src={uploadedImagePreview}
                      alt="Preview"
                      className="w-full h-full object-cover"
                    />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium truncate">{uploadedImage?.name}</p>
                    <p className="text-xs text-gray-500">
                      {(uploadedImage?.size / 1024 / 1024).toFixed(2)} MB
                    </p>

                    {/* Analysis Type Selector */}
                    <div className="flex gap-2 mt-2 flex-wrap">
                      <button
                        onClick={() => setAnalysisType('qwen-vision')}
                        className={`px-2 py-1 rounded text-xs font-medium transition-all flex items-center gap-1 ${
                          analysisType === 'qwen-vision'
                            ? 'bg-purple-500 text-white'
                            : 'bg-[#2a2a2a] text-gray-400 hover:text-white'
                        }`}
                      >
                        <Brain className="w-3 h-3" />
                        AI Analysis
                      </button>
                      <button
                        onClick={() => setAnalysisType('heatmap')}
                        className={`px-2 py-1 rounded text-xs font-medium transition-all ${
                          analysisType === 'heatmap'
                            ? 'bg-[#00d1ff] text-black'
                            : 'bg-[#2a2a2a] text-gray-400 hover:text-white'
                        }`}
                      >
                        <Activity className="w-3 h-3 inline mr-1" />
                        Heatmap
                      </button>
                      <button
                        onClick={() => setAnalysisType('trajectory')}
                        className={`px-2 py-1 rounded text-xs font-medium transition-all ${
                          analysisType === 'trajectory'
                            ? 'bg-[#00d1ff] text-black'
                            : 'bg-[#2a2a2a] text-gray-400 hover:text-white'
                        }`}
                      >
                        <Network className="w-3 h-3 inline mr-1" />
                        Trajectory
                      </button>
                      <button
                        onClick={() => setAnalysisType('both')}
                        className={`px-2 py-1 rounded text-xs font-medium transition-all ${
                          analysisType === 'both'
                            ? 'bg-[#00d1ff] text-black'
                            : 'bg-[#2a2a2a] text-gray-400 hover:text-white'
                        }`}
                      >
                        Both
                      </button>
                    </div>
                  </div>
                  <button
                    onClick={removeImage}
                    className="p-1 hover:bg-[#2a2a2a] rounded transition-colors"
                  >
                    <X className="w-4 h-4 text-gray-400" />
                  </button>
                </div>
              </div>
            )}

            {/* Input */}
            <div className="flex items-end gap-3">
              <input
                ref={fileInputRef}
                type="file"
                onChange={handleFileUpload}
                className="hidden"
                accept="image/*"
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                className={`p-3 rounded-xl transition-all border ${
                  uploadedImage
                    ? 'bg-[#00d1ff] text-black border-[#00d1ff]'
                    : 'bg-[#1a1a1a] text-gray-400 border-[#2a2a2a] hover:border-[#00d1ff]'
                }`}
              >
                <ImageIcon className="w-5 h-5" />
              </button>

              <div className="flex-1 relative">
                <textarea
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder={
                    uploadedImage
                      ? analysisType === 'qwen-vision'
                        ? 'Ask AI to analyze this image...'
                        : 'Describe the image or click send to analyze...'
                      : 'Type a message or upload an image...'
                  }
                  className="w-full bg-[#1a1a1a] text-white placeholder-gray-500 rounded-xl pl-4 pr-12 py-3 resize-none focus:outline-none focus:ring-2 focus:ring-[#00d1ff]/20 border border-[#2a2a2a] focus:border-[#00d1ff] transition-all"
                  rows={1}
                  style={{ minHeight: '48px', maxHeight: '120px' }}
                />
                <button
                  onClick={sendMessage}
                  disabled={(!input.trim() && !uploadedImage) || isLoading}
                  className={`absolute right-2 bottom-2 p-2 rounded-lg transition-all ${
                    (input.trim() || uploadedImage) && !isLoading
                      ? 'bg-[#00d1ff] text-black hover:bg-[#00b8e0]'
                      : 'bg-[#2a2a2a] text-gray-500'
                  }`}
                >
                  {isLoading ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <Send className="w-4 h-4" />
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Result Panel */}
        <div
          ref={imageContainerRef}
          className={`flex flex-col ${
            viewMode === 'chat' ? 'hidden' : viewMode === 'full' ? 'w-full' : 'w-1/2'
          } bg-[#0a0a0a]`}
        >
          <div className="flex-1 flex items-center justify-center p-6 overflow-auto">
            {analysisResult ? (
              <div className="w-full h-full flex flex-col gap-4">
                {/* Original Image */}
                <div className="flex-1 rounded-2xl overflow-hidden border border-[#2a2a2a] bg-[#1a1a1a] relative group">
                  <img
                    src={uploadedImagePreview}
                    alt="Original"
                    className="w-full h-full object-contain"
                  />
                  <div className="absolute top-3 left-3 px-3 py-1 bg-black/70 backdrop-blur rounded-full text-xs font-medium">
                    Original
                  </div>
                </div>

                {/* Results Grid */}
                <div className="grid grid-cols-2 gap-4 flex-1">
                  {analysisResult.heatmapUrl && (
                    <div className="rounded-2xl overflow-hidden border border-[#2a2a2a] bg-[#1a1a1a] relative">
                      <img
                        src={`${API_BASE.replace('/api', '')}${analysisResult.heatmapUrl}`}
                        alt="Heatmap"
                        className="w-full h-full object-contain"
                      />
                      <div className="absolute top-3 left-3 px-3 py-1 bg-black/70 backdrop-blur rounded-full text-xs font-medium flex items-center gap-1">
                        <Activity className="w-3 h-3 text-orange-400" />
                        Heatmap
                      </div>
                      <a
                        href={`${API_BASE.replace('/api', '')}${analysisResult.heatmapUrl}`}
                        download
                        className="absolute bottom-3 right-3 p-2 bg-[#00d1ff] rounded-lg opacity-0 group-hover:opacity-100 transition-opacity"
                      >
                        <Download className="w-4 h-4 text-black" />
                      </a>
                    </div>
                  )}
                  {analysisResult.trajectoryUrl && (
                    <div className="rounded-2xl overflow-hidden border border-[#2a2a2a] bg-[#1a1a1a] relative">
                      <img
                        src={`${API_BASE.replace('/api', '')}${analysisResult.trajectoryUrl}`}
                        alt="Trajectory"
                        className="w-full h-full object-contain"
                      />
                      <div className="absolute top-3 left-3 px-3 py-1 bg-black/70 backdrop-blur rounded-full text-xs font-medium flex items-center gap-1">
                        <Network className="w-3 h-3 text-cyan-400" />
                        Trajectory
                      </div>
                      <a
                        href={`${API_BASE.replace('/api', '')}${analysisResult.trajectoryUrl}`}
                        download
                        className="absolute bottom-3 right-3 p-2 bg-[#00d1ff] rounded-lg opacity-0 group-hover:opacity-100 transition-opacity"
                      >
                        <Download className="w-4 h-4 text-black" />
                      </a>
                    </div>
                  )}
                </div>
              </div>
            ) : uploadedImagePreview ? (
              <div className="text-center">
                <div className="w-64 h-64 rounded-2xl overflow-hidden border border-[#2a2a2a] bg-[#1a1a1a] mb-4 inline-block">
                  <img
                    src={uploadedImagePreview}
                    alt="Preview"
                    className="w-full h-full object-contain"
                  />
                </div>
                <p className="text-gray-400 mb-4">Image uploaded. Click send to analyze.</p>
                <button
                  onClick={analyzeImage}
                  disabled={isLoading}
                  className={`px-6 py-3 rounded-xl font-semibold transition-colors flex items-center gap-2 mx-auto ${
                    analysisType === 'qwen-vision'
                      ? 'bg-purple-500 text-white hover:bg-purple-600'
                      : 'bg-[#00d1ff] text-black hover:bg-[#00b8e0]'
                  }`}
                >
                  {analysisType === 'qwen-vision' ? (
                    <>
                      <Brain className="w-5 h-5" />
                      {isLoading ? 'Analyzing with AI...' : 'Analyze with Qwen Vision'}
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-5 h-5" />
                      {isLoading ? 'Analyzing...' : 'Analyze Image'}
                    </>
                  )}
                </button>
              </div>
            ) : (
              <div className="text-center text-gray-500">
                <Upload className="w-16 h-16 mx-auto mb-4 opacity-50" />
                <p className="text-lg font-medium mb-2">No image uploaded</p>
                <p className="text-sm">Upload an image to generate visualizations</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Settings Panel */}
      <AnimatePresence>
        {showSettings && (
          <>
            <div
              className="fixed inset-0 bg-black/50 backdrop-blur-sm z-40"
              onClick={() => setShowSettings(false)}
            />
            <motion.div
              initial={{ x: 300, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              exit={{ x: 300, opacity: 0 }}
              className="fixed right-0 top-0 bottom-0 w-80 bg-[#111111] border-l border-[#2a2a2a] z-50 p-6"
            >
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-lg font-bold">Settings</h2>
                <button
                  onClick={() => setShowSettings(false)}
                  className="p-2 hover:bg-[#1a1a1a] rounded-lg"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              <div className="space-y-6">
                {/* API Settings */}
                <div>
                  <label className="block text-sm font-medium text-gray-400 mb-2">
                    API URL
                  </label>
                  <input
                    type="text"
                    defaultValue={API_BASE}
                    className="w-full bg-[#1a1a1a] border border-[#2a2a2a] rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-[#00d1ff]"
                  />
                </div>

                {/* Analysis Options */}
                <div>
                  <label className="block text-sm font-medium text-gray-400 mb-2">
                    Default Analysis Type
                  </label>
                  <select
                    value={analysisType}
                    onChange={(e) => setAnalysisType(e.target.value)}
                    className="w-full bg-[#1a1a1a] border border-[#2a2a2a] rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-[#00d1ff]"
                  >
                    <option value="qwen-vision">AI Analysis (Qwen Vision)</option>
                    <option value="heatmap">Heatmap Only</option>
                    <option value="trajectory">Trajectory Only</option>
                    <option value="both">Both (Recommended)</option>
                  </select>
                </div>

                {/* Info */}
                <div className="p-4 bg-[#1a1a1a] rounded-xl border border-[#2a2a2a]">
                  <h3 className="text-sm font-medium mb-2">About</h3>
                  <p className="text-xs text-gray-500">
                    IROS Gaze is an eye-tracking visualization system that generates
                    heatmaps and trajectory overlays from gaze data.
                  </p>
                </div>
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </div>
  );
}

// Message Bubble Component
function MessageBubble({ message }) {
  const isUser = message.role === 'user';
  const isSystem = message.role === 'system';

  if (isSystem) {
    return (
      <div className="flex justify-center">
        <span className="text-xs text-gray-500 bg-[#1a1a1a] px-3 py-1 rounded-full">
          {message.content}
        </span>
      </div>
    );
  }

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-[80%] rounded-2xl px-4 py-3 ${
          isUser
            ? 'bg-[#00d1ff] text-black rounded-br-md'
            : message.isError
              ? 'bg-red-500/20 border border-red-500/30 text-red-300'
              : 'bg-[#1a1a1a] border border-[#2a2a2a] rounded-bl-md'
        }`}
      >
        {/* Image attachment */}
        {message.image && (
          <div className="mb-2 rounded-lg overflow-hidden">
            <img src={message.image} alt="Attached" className="max-w-full h-auto" />
          </div>
        )}

        {/* Text content */}
        <p className="text-sm whitespace-pre-wrap break-words">{message.content}</p>

        {/* Result images */}
        {message.images && (
          <div className="mt-3 grid grid-cols-2 gap-2">
            {message.images.heatmapUrl && (
              <img
                src={`${API_BASE.replace('/api', '')}${message.images.heatmapUrl}`}
                alt="Heatmap"
                className="rounded-lg w-full h-auto"
              />
            )}
            {message.images.trajectoryUrl && (
              <img
                src={`${API_BASE.replace('/api', '')}${message.images.trajectoryUrl}`}
                alt="Trajectory"
                className="rounded-lg w-full h-auto"
              />
            )}
          </div>
        )}

        {/* Timestamp */}
        <span className="text-xs opacity-50 mt-1 block">
          {new Date(message.timestamp).toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit',
          })}
        </span>
      </div>
    </div>
  );
}

// Layout component for the icon
function Layout({ className }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <rect width="7" height="9" x="3" y="3" rx="1" />
      <rect width="7" height="5" x="14" y="3" rx="1" />
      <rect width="7" height="9" x="14" y="12" rx="1" />
      <rect width="7" height="5" x="3" y="16" rx="1" />
    </svg>
  );
}

export default App;
