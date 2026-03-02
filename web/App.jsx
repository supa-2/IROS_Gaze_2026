import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  ArrowRight,
  Box,
  Layers,
  Cpu,
  Menu,
  X,
  Send,
  Code,
  Download,
  Sparkles,
  FileText,
  Paperclip,
  Image as ImageIcon,
  Loader2,
  Eye,
  MapPin,
  ChevronRight,
} from 'lucide-react';

// ==================== NAVBAR ====================
const Navbar = ({ onStartCreate, isCreatorMode, onBack }) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-black/90 backdrop-blur-xl border-b border-gray-800 py-4">
      <div className="max-w-[1400px] mx-auto px-6 lg:px-12 flex justify-between items-center">
        <div className="flex items-center gap-3 cursor-pointer" onClick={onBack}>
          <div className="w-10 h-10 bg-white text-black flex items-center justify-center font-bold text-xl">
            P
          </div>
          <span className="font-bold text-2xl tracking-tighter text-white">PalX</span>
        </div>

        {!isCreatorMode ? (
          <div className="hidden md:flex items-center space-x-10">
            {['Technology', 'Workflow', 'Products', 'Pricing'].map((item) => (
              <a key={item} href={`#${item.toLowerCase()}`} className="text-sm font-medium text-gray-400 hover:text-white transition-colors uppercase tracking-wider">
                {item}
              </a>
            ))}
            <button onClick={onStartCreate} className="bg-white text-black px-8 py-3 font-semibold hover:bg-gray-200 transition-colors text-sm tracking-wide flex items-center gap-2">
              <Sparkles size={16} />
              START CREATING
            </button>
          </div>
        ) : (
          <div className="hidden md:flex items-center gap-6">
            <span className="flex items-center gap-2 text-xs font-mono text-gray-500 bg-gray-900 px-3 py-1 rounded-full border border-gray-700">
              <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
              PalX_Core_Engine Active
            </span>
            <button onClick={onBack} className="text-sm font-bold text-gray-400 hover:text-white">
              EXIT TERMINAL
            </button>
          </div>
        )}

        <div className="md:hidden">
          <button onClick={() => setIsOpen(!isOpen)} className="text-white p-2">
            {isOpen ? <X size={24} /> : <Menu size={24} />}
          </button>
        </div>
      </div>
    </nav>
  );
};

// ==================== LANDING PAGE ====================
const LandingPage = ({ onStartCreate }) => (
  <div className="min-h-screen bg-black">
    <div className="absolute inset-0 bg-[linear-gradient(to_right,#1a1a1a_1px,transparent_1px),linear-gradient(to_bottom,#1a1a1a_1px,transparent_1px)] bg-[size:4rem_4rem) opacity-20"></div>

    <section className="relative min-h-screen flex items-center justify-center px-6">
      <div className="text-center z-10 max-w-4xl">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <h1 className="text-5xl md:text-7xl font-bold text-white mb-6 tracking-tight">
            Design the Future
          </h1>
          <p className="text-xl text-gray-400 mb-8">
            AI-powered spatial design system for intelligent environments
          </p>
          <button
            onClick={onStartCreate}
            className="bg-white text-black px-10 py-4 rounded-full font-bold text-lg hover:bg-gray-200 transition-all flex items-center gap-3 mx-auto"
          >
            <Sparkles size={20} />
            Start Creating
            <ArrowRight size={20} />
          </button>
        </motion.div>
      </div>

      {/* Floating elements */}
      {[...Array(6)].map((_, i) => (
        <motion.div
          key={i}
          className="absolute w-16 h-16 border border-gray-700 rounded-lg"
          style={{
            left: `${10 + i * 15}%`,
            top: `${20 + (i % 3) * 25}%`,
          }}
          animate={{
            y: [0, -15, 0],
            rotate: [0, 3, 0],
          }}
          transition={{
            duration: 4 + i,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />
      ))}
    </section>

    {/* Features */}
    <section className="py-24 px-6">
      <div className="max-w-6xl mx-auto">
        <div className="grid md:grid-cols-3 gap-8">
          {[
            { icon: Box, title: 'Smart Generation', desc: 'AI-powered design with spatial awareness' },
            { icon: Layers, title: 'Multi-Layer Output', desc: 'Complex designs with hierarchical structures' },
            { icon: Cpu, title: 'Real-time Processing', desc: 'Instant previews and iterations' },
          ].map((feature, i) => (
            <div key={i} className="bg-gray-900/50 border border-gray-800 rounded-2xl p-8">
              <div className="w-14 h-14 bg-gray-800 rounded-xl flex items-center justify-center mb-6">
                <feature.icon className="w-7 h-7 text-white" />
              </div>
              <h3 className="text-xl font-bold text-white mb-3">{feature.title}</h3>
              <p className="text-gray-400">{feature.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  </div>
);

// ==================== CREATOR APP (DESIGN TERMINAL) ====================
const CreatorApp = ({ onBack }) => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      role: 'assistant',
      content: '> PalX Design Terminal v2.0\n> Connected to: TH_Museum_Map\n> System ready.\n\nAvailable commands:\n• Type design descriptions\n• Upload reference images\n• Generate DXF blueprints\n• Export designs\n\nWaiting for input...',
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [selectedImage, setSelectedImage] = useState(null);
  const [blueprintData, setBlueprintData] = useState(null);
  const [showExport, setShowExport] = useState(false);

  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

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

  const handleSend = async () => {
    if (!input.trim() && !selectedImage) return;

    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: input || 'Analyze reference image',
      timestamp: new Date(),
      image: selectedImage?.preview,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsTyping(true);

    // Simulate AI response
    setTimeout(() => {
      const newBlueprint = {
        id: 'TH-E01',
        name: 'Entrance Hall',
        exits: ['TH-I-B01', 'TH-B02'],
        features: 'High visibility area with optimized flow',
        coordinates: { x: 120, y: 80 },
      };

      const responseMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: `> Processing: "${userMessage.content}"\n\n> Analyzing spatial requirements...\n> Computing topology...\n> Generating blueprint...\n\n✅ Design complete!\n\nBlueprint ID: ${newBlueprint.id}\nExits: ${newBlueprint.exits.join(', ')}`,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, responseMessage]);
      setBlueprintData(newBlueprint);
      setIsTyping(false);
      setSelectedImage(null);
    }, 2000);
  };

  return (
    <div className="h-screen bg-black flex flex-col pt-16">
      {/* Terminal Header */}
      <header className="flex items-center justify-between px-6 py-3 border-b border-gray-800 bg-gray-900/50">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-white text-black flex items-center justify-center font-bold text-sm">
            P
          </div>
          <div>
            <h1 className="text-sm font-bold text-white">Design Terminal</h1>
            <p className="text-[10px] text-gray-500 font-mono">TH_Map :: PalX_Engine_v2.0</p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <span className="hidden md:flex items-center gap-2 text-[10px] font-mono text-gray-500 bg-gray-800 px-3 py-1 rounded-full border border-gray-700">
            <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse"></span>
            ONLINE
          </span>
          <button onClick={() => setShowExport(!showExport)} className="p-2 hover:bg-gray-800 rounded-lg">
            <Download className="w-4 h-4 text-gray-400" />
          </button>
          <button onClick={onBack} className="hidden md:flex items-center gap-2 px-3 py-1.5 bg-gray-800 hover:bg-gray-700 rounded-lg text-xs">
            <X className="w-3 h-3" />
            EXIT
          </button>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Chat Panel */}
        <div className="flex-1 flex flex-col border-r border-gray-800">
          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-6 space-y-4">
            {messages.map((msg) => (
              <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`max-w-[80%] rounded-xl px-4 py-3 ${
                  msg.role === 'user'
                    ? 'bg-white text-black rounded-br-md'
                    : 'bg-gray-900 border border-gray-800 text-gray-200 rounded-bl-md'
                }`}>
                  {msg.image && (
                    <div className="mb-3 rounded-lg overflow-hidden">
                      <img src={msg.image} alt="Reference" className="max-w-full h-auto" />
                    </div>
                  )}
                  <p className="text-sm whitespace-pre-wrap font-mono">{msg.content}</p>
                  <span className="text-[10px] opacity-40 mt-2 block font-mono">
                    {new Date(msg.timestamp).toLocaleTimeString()}
                  </span>
                </div>
              </div>
            ))}
            {isTyping && (
              <div className="flex items-center gap-3 text-gray-400">
                <Loader2 className="w-4 h-4 animate-spin text-white" />
                <span className="text-xs font-mono">Processing</span>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <div className="p-4 border-t border-gray-800 bg-gray-900/30">
            {selectedImage && (
              <div className="mb-3 p-2 bg-gray-900 rounded-lg border border-gray-800 flex items-center gap-3">
                <img src={selectedImage.preview} alt="Preview" className="w-10 h-10 object-cover rounded" />
                <div className="flex-1 min-w-0">
                  <p className="text-xs font-medium truncate text-white">{selectedImage.name}</p>
                  <p className="text-[10px] text-gray-500">Reference attached</p>
                </div>
                <button onClick={() => setSelectedImage(null)} className="p-1 hover:bg-gray-800 rounded">
                  <X className="w-3 h-3 text-gray-400" />
                </button>
              </div>
            )}

            <div className="flex gap-2 items-end">
              <input
                ref={fileInputRef}
                type="file"
                onChange={handleImageUpload}
                className="hidden"
                accept="image/*"
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                className={`p-2.5 rounded-lg transition-all border ${
                  selectedImage
                    ? 'bg-white text-black border-white'
                    : 'bg-gray-800 text-gray-400 border-gray-700 hover:border-gray-500'
                }`}
              >
                {selectedImage ? <ImageIcon className="w-4 h-4" /> : <Paperclip className="w-4 h-4" />}
              </button>

              <div className="flex-1 relative">
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                  placeholder="Input design command..."
                  className="w-full bg-gray-800 text-white placeholder-gray-500 font-mono text-sm rounded-lg px-4 pr-10 py-2.5 focus:outline-none focus:ring-1 focus:ring-white/20 border border-gray-700 focus:border-gray-500"
                />
                <button
                  onClick={handleSend}
                  disabled={(!input.trim() && !selectedImage) || isTyping}
                  className={`absolute right-2 top-1.5 p-1.5 rounded transition-all ${
                    (input.trim() || selectedImage) && !isTyping
                      ? 'bg-white text-black'
                      : 'bg-gray-700 text-gray-500'
                  }`}
                >
                  {isTyping ? <Loader2 className="w-3 h-3 animate-spin" /> : <Send className="w-3 h-3" />}
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Blueprint Preview Panel */}
        <div className="w-72 bg-black border-l border-gray-800 flex flex-col">
          <div className="p-3 border-b border-gray-800">
            <h3 className="font-bold text-white text-sm flex items-center gap-2">
              <FileText className="w-3 h-3" />
              Blueprint Preview
            </h3>
          </div>

          <div className="flex-1 p-4 overflow-auto">
            {blueprintData ? (
              <div className="space-y-3">
                {/* Blueprint Card */}
                <div className="bg-gray-900 rounded-xl overflow-hidden border border-gray-800">
                  <div className="aspect-square bg-gradient-to-br from-gray-900 to-black flex items-center justify-center relative p-4">
                    {/* Grid overlay */}
                    <div className="absolute inset-0 opacity-20" style={{
                      backgroundImage: 'linear-gradient(to right, #333 1px, transparent 1px), linear-gradient(to bottom, #333 1px, transparent 1px)',
                      backgroundSize: '20px 20px'
                    }}></div>

                    {/* Location marker */}
                    <div className="relative z-10 text-center">
                      <div className="w-16 h-16 mx-auto mb-2 rounded-full border-2 border-white/20 flex items-center justify-center">
                        <MapPin className="w-6 h-6 text-white" />
                      </div>
                      <p className="text-[10px] font-mono text-gray-400">{blueprintData.id}</p>
                    </div>

                    {/* DXF Badge */}
                    <div className="absolute top-2 right-2 bg-black/70 backdrop-blur text-white text-[8px] font-bold px-2 py-0.5 rounded border border-white/10">
                      DXF
                    </div>
                  </div>

                  {/* Card Footer */}
                  <div className="p-3 bg-gray-800/50 flex justify-between items-center border-t border-gray-700">
                    <div className="overflow-hidden">
                      <div className="font-bold text-white text-[10px] truncate">{blueprintData.name}</div>
                      <div className="text-[8px] text-gray-500 font-mono">{blueprintData.id}</div>
                    </div>
                    <button className="text-[8px] font-bold bg-white hover:bg-gray-200 text-black px-2 py-1 rounded-full flex items-center gap-1">
                      <Download size={8} />
                      GET
                    </button>
                  </div>
                </div>

                {/* Info */}
                <div className="bg-gray-900 rounded-lg p-3 border border-gray-800 text-xs space-y-2">
                  <div className="flex justify-between">
                    <span className="text-gray-500">Exits:</span>
                    <span className="text-white">{blueprintData.exits.length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">Type:</span>
                    <span className="text-white">Hall</span>
                  </div>
                  <div className="pt-2 border-t border-gray-800">
                    <p className="text-gray-400 text-[10px]">{blueprintData.features}</p>
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center text-gray-600 py-12">
                <FileText className="w-12 h-12 mx-auto mb-3 opacity-50" />
                <p className="text-xs">No blueprint</p>
                <p className="text-[10px] mt-1">Submit a command to create</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Export Panel */}
      <AnimatePresence>
        {showExport && (
          <>
            <div
              className="fixed inset-0 bg-black/50 backdrop-blur-sm z-40"
              onClick={() => setShowExport(false)}
            />
            <motion.div
              initial={{ x: 300, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              exit={{ x: 300, opacity: 0 }}
              className="fixed right-0 top-0 bottom-0 w-64 bg-gray-900 border-l border-gray-800 z-50 p-4"
            >
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-sm font-bold">Export</h2>
                <button onClick={() => setShowExport(false)}>
                  <X className="w-4 h-4" />
                </button>
              </div>
              <div className="space-y-2">
                <button className="w-full p-3 bg-gray-800 hover:bg-gray-700 rounded-lg text-left text-sm">
                  <div className="font-bold">DXF Format</div>
                  <div className="text-[10px] text-gray-500">AutoCAD</div>
                </button>
                <button className="w-full p-3 bg-gray-800 hover:bg-gray-700 rounded-lg text-left text-sm">
                  <div className="font-bold">JSON</div>
                  <div className="text-[10px] text-gray-500">Raw data</div>
                </button>
                <button className="w-full p-3 bg-gray-800 hover:bg-gray-700 rounded-lg text-left text-sm">
                  <div className="font-bold">PNG</div>
                  <div className="text-[10px] text-gray-500">Image export</div>
                </button>
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </div>
  );
};

// ==================== MAIN APP ====================
const App = () => {
  const [view, setView] = useState('landing');

  return (
    <div className="font-sans antialiased text-white bg-black overflow-x-hidden">
      <Navbar
        onStartCreate={() => setView('creator')}
        onBack={() => setView('landing')}
        isCreatorMode={view === 'creator'}
      />

      <AnimatePresence mode="wait">
        {view === 'landing' ? (
          <LandingPage key="landing" onStartCreate={() => setView('creator')} />
        ) : (
          <CreatorApp key="creator" onBack={() => setView('landing')} />
        )}
      </AnimatePresence>
    </div>
  );
};

export default App;
