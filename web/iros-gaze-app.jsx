import React, { useState, useEffect, useRef } from 'react';
import { Eye, Paperclip, Image as ImageIcon, X, FileText } from 'lucide-react';

const IrosgazeApp = ({ onBack }) => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [selectedImage, setSelectedImage] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);

  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'auto' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isProcessing]);

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

  const handleAnalyze = async () => {
    if (!selectedImage) return;

    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: input || 'Analyze this image',
      timestamp: new Date(),
      image: selectedImage.preview,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsProcessing(true);

    try {
      const formData = new FormData();
      formData.append('image', selectedImage.file);
      formData.append('maskPrompt', 'auto');
      formData.append('vlmPrompt', 'auto');

      const response = await fetch('http://localhost:5000/api/iros-gaze/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const result = await response.json();

      const assistantMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: `> IROS GAZE ANALYSIS RESULT\n\n技术参数 (TECHNICAL SPECS):\n--------------------------------\nSAM2 分割 (Masks): ${result.segmentation?.num_masks || 0} 个\nVLM 场景 (Exhibits): ${result.topology?.exhibits?.length || 0} 个\n热力图 (Heatmap): 已生成\n轨迹图 (Trajectory): 已生成`,
        timestamp: new Date(),
        result,
      };

      setMessages((prev) => [...prev, assistantMessage]);
      setAnalysisResult(result);
    } catch (error) {
      const errorMessage = {
        id: Date.now() + 1,
        role: 'error',
        content: `ERROR: ${error.message}\n\n请检查:\n1. 后端服务器是否运行 (端口 5000)\n2. Python 依赖是否安装\n3. 图片格式是否支持`,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsProcessing(false);
      setSelectedImage(null);
    }
  };

  return (
    <div className="h-screen bg-black flex flex-col pt-16">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-3 border-b border-gray-800">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-white flex items-center justify-center rounded">
            <Eye className="w-4 h-4 text-black" />
          </div>
          <div>
            <h1 className="text-sm font-bold text-white">IROS Gaze Analysis</h1>
            <p className="text-[10px] text-gray-500 font-mono">SAM2 + VLM + Heatmap + Trajectory</p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <span className="hidden md:flex items-center gap-2 text-[10px] font-mono text-gray-500 bg-gray-900 px-3 py-1 rounded-full border border-gray-800">
            <span className="w-1.5 h-1.5 rounded-full bg-white"></span>
            READY
          </span>
          <button onClick={onBack} className="flex items-center gap-2 px-3 py-1.5 bg-white hover:bg-gray-200 rounded-lg text-xs font-bold transition-colors">
            <X className="w-3 h-3" />
            EXIT
          </button>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Messages Panel */}
        <div className="flex-1 flex flex-col border-r border-gray-800">
          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-6">
            <div className="max-w-3xl mx-auto space-y-6">
              {messages.map((msg) => (
                <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`max-w-[85%] rounded-lg px-5 py-4 ${
                    msg.role === 'user'
                      ? 'bg-white text-black rounded-br-sm'
                      : msg.role === 'error'
                      ? 'bg-red-900/20 border border-red-800 text-red-400 rounded-bl-sm'
                      : 'bg-gray-900 border border-gray-800 text-white rounded-bl-sm'
                  }`}>
                    {msg.image && (
                      <div className="mb-3 rounded overflow-hidden bg-gray-800">
                        <img src={msg.image} alt="Upload" className="max-w-full h-auto max-h-64 object-contain" />
                      </div>
                    )}
                    <p className="text-sm font-mono whitespace-pre-wrap">{msg.content}</p>
                    <span className="text-[10px] text-gray-500 mt-2 block font-mono">
                      {new Date(msg.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                </div>
              ))}
              
              {isProcessing && (
                <div className="flex justify-start">
                  <div className="bg-gray-900 border border-gray-800 rounded-lg px-5 py-4 rounded-bl-sm">
                    <div className="flex items-center gap-3">
                      <div className="w-2 h-2 bg-white rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                      <div className="w-2 h-2 bg-white rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                      <div className="w-2 h-2 bg-white rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                      <span className="text-xs font-mono text-gray-400 ml-2">Processing with SAM2 + VLM...</span>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          </div>

          {/* Input Area */}
          <div className="p-4 border-t border-gray-800">
            <div className="max-w-3xl mx-auto">
              {selectedImage && (
                <div className="mb-3 p-2 bg-gray-900 rounded border border-gray-800 flex items-center gap-3">
                  <img src={selectedImage.preview} alt="Preview" className="w-12 h-12 object-cover rounded" />
                  <div className="flex-1 min-w-0">
                    <p className="text-xs font-medium text-white truncate">{selectedImage.name}</p>
                    <p className="text-[10px] text-gray-500 font-mono">Ready to analyze</p>
                  </div>
                  <button 
                    onClick={() => setSelectedImage(null)} 
                    className="p-1.5 hover:bg-gray-800 rounded transition-colors"
                  >
                    <X className="w-3 h-3 text-gray-400" />
                  </button>
                </div>
              )}

              <div className="flex gap-2">
                <input
                  ref={fileInputRef}
                  type="file"
                  onChange={handleImageUpload}
                  className="hidden"
                  accept="image/*"
                />
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className={`px-4 py-3 rounded border transition-all flex items-center gap-2 ${
                    selectedImage
                      ? 'bg-white text-black border-white'
                      : 'bg-gray-900 text-gray-400 border-gray-800 hover:border-gray-600'
                  }`}
                >
                  {selectedImage ? <ImageIcon className="w-4 h-4" /> : <Paperclip className="w-4 h-4" />}
                  <span className="text-xs font-mono">{selectedImage ? 'CHANGED' : 'UPLOAD'}</span>
                </button>

                <div className="flex-1 relative">
                  <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleAnalyze()}
                    placeholder="Enter prompt (optional)..."
                    className="w-full bg-gray-900 text-white placeholder-gray-600 font-mono text-sm rounded px-4 py-3 pr-14 focus:outline-none border border-gray-800 focus:border-white transition-colors"
                  />
                  <button
                    onClick={handleAnalyze}
                    disabled={(!input.trim() && !selectedImage) || isProcessing}
                    className={`absolute right-2 top-1/2 -translate-y-1/2 px-3 py-1.5 rounded text-xs font-bold transition-all ${
                      (input.trim() || selectedImage) && !isProcessing
                        ? 'bg-white text-black hover:bg-gray-200'
                        : 'bg-gray-800 text-gray-600'
                    }`}
                  >
                    {isProcessing ? '...' : 'RUN'}
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Results Panel */}
        <div className="w-96 bg-black border-l border-gray-800 flex flex-col">
          <div className="p-3 border-b border-gray-800 flex items-center justify-between">
            <h3 className="font-bold text-white text-xs font-mono flex items-center gap-2">
              <FileText className="w-3 h-3" />
              RESULTS
            </h3>
            {analysisResult && (
              <span className="text-[10px] text-gray-500 font-mono">
                {analysisResult.segmentation?.num_masks || 0} masks
              </span>
            )}
          </div>

          <div className="flex-1 p-4 overflow-auto">
            {analysisResult ? (
              <div className="space-y-4">
                {analysisResult.outputs.original && (
                  <div>
                    <div className="aspect-video bg-gray-900 rounded border border-gray-800 overflow-hidden mb-2">
                      <img 
                        src={analysisResult.outputs.original} 
                        alt="Original" 
                        className="w-full h-full object-contain"
                      />
                    </div>
                    <p className="text-[10px] text-gray-500 font-mono text-center">ORIGINAL</p>
                  </div>
                )}

                {analysisResult.outputs.heatmap && (
                  <div>
                    <div className="aspect-video bg-gray-900 rounded border border-gray-800 overflow-hidden mb-2">
                      <img 
                        src={analysisResult.outputs.heatmap} 
                        alt="Heatmap" 
                        className="w-full h-full object-contain"
                      />
                    </div>
                    <p className="text-[10px] text-white font-mono text-center">HEATMAP</p>
                  </div>
                )}

                {analysisResult.outputs.trajectory && (
                  <div>
                    <div className="aspect-video bg-gray-900 rounded border border-gray-800 overflow-hidden mb-2">
                      <img 
                        src={analysisResult.outputs.trajectory} 
                        alt="Trajectory" 
                        className="w-full h-full object-contain"
                      />
                    </div>
                    <p className="text-[10px] text-white font-mono text-center">TRAJECTORY</p>
                  </div>
                )}
              </div>
            ) : (
              <div className="h-full flex items-center justify-center text-gray-700">
                <div className="text-center">
                  <div className="w-12 h-12 mx-auto mb-3 opacity-50 flex items-center justify-center">
                    <Eye className="w-8 h-8" />
                  </div>
                  <p className="text-xs font-mono">WAITING FOR INPUT</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default IrosgazeApp;