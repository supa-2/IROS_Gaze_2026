import React, { useState, useEffect, useRef } from 'react';
import { motion, useScroll, useTransform, AnimatePresence, useInView } from 'framer-motion';
import { ArrowRight, Box, Layers, Cpu, ShoppingCart, Menu, X, Globe, Play, CheckCircle, ChevronRight, Zap, Send, Code, Maximize2, RotateCcw, Download, Sparkles, FileText, Ruler, Paperclip, Image as ImageIcon, Loader2, ZoomIn, Check, LayoutGrid, Hammer, Package, Heart, MapPin, User, Eye, Settings, MessageSquare, Brain, Activity, Network, Wand2, Upload } from 'lucide-react';

// --- API Configuration ---
const API_BASE = 'http://localhost:5000/api';

// --- Global Styles & Utils ---
const GridBackground = () => (
  <div className="absolute inset-0 z-0 pointer-events-none overflow-hidden">
    <div className="absolute inset-0 bg-[linear-gradient(to_right,#1a1a1a_1px,transparent_1px),linear-gradient(to_bottom,#1a1a1a_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_0%,#000_70%,transparent_100%)] opacity-70"></div>
  </div>
);

// --- Components ---

// 1. Navigation Bar
const Navbar = ({ onStartCreate, isCreatorMode, onBack }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 50);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <nav className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${scrolled || isCreatorMode ? 'bg-[#0a0a0a]/90 backdrop-blur-xl border-b border-[#2a2a2a] py-4' : 'bg-transparent py-6'}`}>
      <div className="max-w-[1400px] mx-auto px-6 lg:px-12">
        <div className="flex justify-between items-center">
          {/* Logo */}
          <div className="flex items-center gap-3 cursor-pointer group" onClick={onBack}>
            <div className="w-10 h-10 bg-gradient-to-br from-[#00d1ff] to-[#0066ff] flex items-center justify-center font-bold text-xl relative overflow-hidden group-hover:scale-110 transition-transform duration-300">
               <Eye className="w-5 h-5 text-white" />
            </div>
            <span className="font-bold text-2xl tracking-tighter text-white">PalX</span>
          </div>

          {/* Desktop Menu */}
          {!isCreatorMode ? (
            <div className="hidden md:flex items-center space-x-10">
              {['Technology', 'Workflow', 'Products', 'Pricing'].map((item) => (
                <a key={item} href={`#${item.toLowerCase()}`} className="text-sm font-medium text-gray-400 hover:text-white transition-colors uppercase tracking-wider">
                  {item}
                </a>
              ))}
              <button
                onClick={onStartCreate}
                className="bg-[#00d1ff] text-black px-8 py-3 font-semibold hover:bg-[#00b8e0] transition-colors duration-300 text-sm tracking-wide flex items-center gap-2"
              >
                <Sparkles size={16} />
                START CREATING
              </button>
            </div>
          ) : (
             <div className="hidden md:flex items-center gap-6">
                <span className="flex items-center gap-2 text-xs font-mono text-gray-500 bg-[#1a1a1a] px-3 py-1 rounded-full border border-[#2a2a2a]">
                  <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
                  PalX_Core_Engine Active
                </span>
                <button
                  onClick={onBack}
                  className="text-sm font-bold text-gray-400 hover:text-white transition-colors"
                >
                  EXIT TERMINAL
                </button>
             </div>
          )}

          {/* Mobile Menu Button */}
          <div className="md:hidden">
            <button onClick={() => setIsOpen(!isOpen)} className="text-white p-2">
              {isOpen ? <X size={24} /> : <Menu size={24} />}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Dropdown */}
      <AnimatePresence>
        {isOpen && !isCreatorMode && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="md:hidden bg-[#0a0a0a] border-b border-[#2a2a2a] overflow-hidden"
          >
            <div className="px-6 py-8 space-y-6 flex flex-col">
              {['Technology', 'Workflow', 'Products', 'Pricing'].map((item) => (
                <a key={item} href={`#${item.toLowerCase()}`} onClick={() => setIsOpen(false)} className="text-xl font-medium text-white">
                  {item}
                </a>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </nav>
  );
};

// 2. Hero Section
const HeroSection = ({ onStartCreate }) => {
  const { scrollY } = useScroll();
  const y = useTransform(scrollY, [0, 500], [0, 200]);
  const opacity = useTransform(scrollY, [0, 300], [1, 0]);

  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
      <motion.div style={{ y, opacity }} className="text-center z-10 px-6">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <h1 className="text-5xl md:text-7xl font-bold text-white mb-6 tracking-tight">
            Design with <span className="text-[#00d1ff]">AI Intelligence</span>
          </h1>
          <p className="text-xl text-gray-400 mb-8 max-w-2xl mx-auto">
            Transform your ideas into professional designs with spatial awareness and intelligent automation.
          </p>
          <button
            onClick={onStartCreate}
            className="group bg-[#00d1ff] text-black px-10 py-4 rounded-full font-bold text-lg hover:bg-[#00b8e0] transition-all duration-300 flex items-center gap-3 mx-auto"
          >
            Start Creating
            <ArrowRight className="group-hover:translate-x-1 transition-transform" />
          </button>
        </motion.div>
      </motion.div>

      {/* Floating Elements */}
      <div className="absolute inset-0 pointer-events-none">
        {[...Array(6)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-20 h-20 border border-[#00d1ff]/20 rounded-xl"
            style={{
              left: `${10 + i * 15}%`,
              top: `${20 + (i % 3) * 25}%`,
            }}
            animate={{
              y: [0, -20, 0],
              rotate: [0, 5, 0],
            }}
            transition={{
              duration: 3 + i,
              repeat: Infinity,
              ease: "easeInOut",
            }}
          />
        ))}
      </div>
    </section>
  );
};

// 3. Feature Cards
const FeatureCard = ({ icon: Icon, title, description, delay }) => (
  <motion.div
    initial={{ opacity: 0, y: 50 }}
    whileInView={{ opacity: 1, y: 0 }}
    viewport={{ once: true }}
    transition={{ duration: 0.5, delay }}
    className="bg-[#111111] border border-[#2a2a2a] rounded-2xl p-8 hover:border-[#00d1ff]/50 transition-all duration-300 group"
  >
    <div className="w-14 h-14 bg-[#1a1a1a] rounded-xl flex items-center justify-center mb-6 group-hover:bg-[#00d1ff]/20 transition-colors">
      <Icon className="w-7 h-7 text-[#00d1ff]" />
    </div>
    <h3 className="text-xl font-bold text-white mb-3">{title}</h3>
    <p className="text-gray-400">{description}</p>
  </motion.div>
);

const FeaturesSection = () => {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true });

  const features = [
    { icon: Box, title: 'Smart Generation', description: 'AI-powered design generation with spatial awareness and intelligent layout optimization.' },
    { icon: Layers, title: 'Multi-Layer Output', description: 'Generate complex designs with multiple layers, components, and hierarchical structures.' },
    { icon: Cpu, title: 'Real-time Processing', description: 'Lightning-fast processing with real-time previews and instant iterations.' },
  ];

  return (
    <section ref={ref} id="technology" className="py-24 px-6">
      <div className="max-w-6xl mx-auto">
        <motion.h2
          initial={{ opacity: 0, y: 30 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          className="text-4xl font-bold text-white text-center mb-4"
        >
          Powerful Technology
        </motion.h2>
        <p className="text-gray-400 text-center mb-16 max-w-2xl mx-auto">
          Built with cutting-edge AI models to transform your ideas into reality
        </p>
        <div className="grid md:grid-cols-3 gap-8">
          {features.map((feature, i) => (
            <FeatureCard key={i} {...feature} delay={i * 0.1} />
          ))}
        </div>
      </div>
    </section>
  );
};

// 4. Workflow Section
const WorkflowSection = () => {
  const steps = [
    { num: '01', title: 'Describe', desc: 'Input your design requirements in natural language' },
    { num: '02', title: 'Generate', desc: 'AI processes and generates initial design concepts' },
    { num: '03', title: 'Refine', desc: 'Iterate and refine until perfection' },
    { num: '04', title: 'Export', desc: 'Download in multiple formats including DXF' },
  ];

  return (
    <section id="workflow" className="py-24 px-6 bg-[#0a0a0a]">
      <div className="max-w-6xl mx-auto">
        <h2 className="text-4xl font-bold text-white text-center mb-16">Simple Workflow</h2>
        <div className="grid md:grid-cols-4 gap-6">
          {steps.map((step, i) => (
            <div key={i} className="relative">
              <div className="bg-[#111111] border border-[#2a2a2a] rounded-2xl p-6 text-center">
                <span className="text-4xl font-bold text-[#00d1ff]/30">{step.num}</span>
                <h3 className="text-lg font-bold text-white mt-4 mb-2">{step.title}</h3>
                <p className="text-sm text-gray-400">{step.desc}</p>
              </div>
              {i < 3 && (
                <ChevronRight className="hidden md:block absolute top-1/2 -right-3 text-[#00d1ff]" />
              )}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

// 5. CTA Section
const CTASection = ({ onStartCreate }) => (
  <section className="py-24 px-6">
    <div className="max-w-4xl mx-auto text-center bg-gradient-to-r from-[#00d1ff]/10 to-[#0066ff]/10 border border-[#00d1ff]/20 rounded-3xl p-12">
      <h2 className="text-3xl font-bold text-white mb-4">Ready to Create?</h2>
      <p className="text-gray-400 mb-8">Join thousands of designers already using PalX</p>
      <button
        onClick={onStartCreate}
        className="bg-[#00d1ff] text-black px-10 py-4 rounded-full font-bold hover:bg-[#00b8e0] transition-colors"
      >
        Get Started Free
      </button>
    </div>
  </section>
);

// 6. Landing Page
const LandingPage = ({ onStartCreate }) => (
  <div className="min-h-screen bg-[#0a0a0a]">
    <GridBackground />
    <Navbar onStartCreate={onStartCreate} isCreatorMode={false} />
    <HeroSection onStartCreate={onStartCreate} />
    <FeaturesSection />
    <WorkflowSection />
    <CTASection onStartCreate={onStartCreate} />
  </div>
);

// 7. Creator App (Design Terminal)
const CreatorApp = ({ onBack }) => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      role: 'assistant',
      content: 'PalX Design Terminal initialized.\n\nConnected to: TH Museum Map\nAvailable commands:\n• Input design descriptions\n• Upload reference images\n• Generate DXF blueprints\n• Export designs\n\nReady to create.',
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [selectedImage, setSelectedImage] = useState(null);
  const [blueprintPreview, setBlueprintPreview] = useState(null);
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
      content: input || 'Analyze this reference image',
      timestamp: new Date(),
      image: selectedImage?.preview,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsTyping(true);

    // Simulate AI response
    setTimeout(() => {
      const responseMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: `Processing design command: "${userMessage.content}"\n\nAnalyzing spatial requirements...\nGenerating topology-based layout...\nOptimizing for museum flow...\n\n✅ Design ready! Preview available on the right panel.`,
        timestamp: new Date(),
        blueprint: {
          id: 'TH-E01',
          name: 'Entrance Hall Design',
          description: 'Optimized layout for visitor flow and exhibit visibility',
        },
      };
      setMessages((prev) => [...prev, responseMessage]);
      setBlueprintPreview({
        id: 'TH-E01',
        name: 'Entrance Hall Design',
        exits: ['TH-I-B01', 'TH-B02'],
        features: 'High visibility area with dual exit points',
      });
      setIsTyping(false);
      setSelectedImage(null);
    }, 2000);
  };

  return (
    <div className="h-screen bg-[#0a0a0a] flex flex-col">
      {/* Terminal Header */}
      <header className="flex items-center justify-between px-6 py-4 border-b border-[#2a2a2a] bg-[#111111]/80 backdrop-blur-xl">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-[#00d1ff] to-[#0066ff] flex items-center justify-center">
            <Code className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-white">Design Terminal</h1>
            <p className="text-xs text-gray-500 font-mono">TH_Museum_Map :: PalX_Engine_v2.0</p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <div className="hidden md:flex items-center gap-2 text-xs font-mono text-gray-500 bg-[#1a1a1a] px-3 py-1 rounded-full border border-[#2a2a2a]">
            <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
            System Online
          </div>
          <button
            onClick={() => setShowExport(!showExport)}
            className="p-2 hover:bg-[#1a1a1a] rounded-lg transition-colors"
          >
            <Download className="w-5 h-5 text-gray-400" />
          </button>
          <button
            onClick={onBack}
            className="hidden md:flex items-center gap-2 px-4 py-2 bg-[#1a1a1a] hover:bg-[#2a2a2a] rounded-lg transition-colors text-sm font-medium"
          >
            <X className="w-4 h-4" />
            Exit
          </button>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Chat Panel */}
        <div className="flex-1 flex flex-col border-r border-[#2a2a2a]">
          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-6 space-y-6">
            {messages.map((msg) => (
              <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`max-w-[80%] rounded-2xl px-5 py-4 ${
                  msg.role === 'user'
                    ? 'bg-[#00d1ff] text-black rounded-br-md'
                    : 'bg-[#1a1a1a] border border-[#2a2a2a] text-gray-200 rounded-bl-md'
                }`}>
                  {msg.image && (
                    <div className="mb-3 rounded-lg overflow-hidden">
                      <img src={msg.image} alt="Reference" className="max-w-full h-auto" />
                    </div>
                  )}
                  <p className="text-sm whitespace-pre-wrap font-mono">{msg.content}</p>
                  {msg.blueprint && setBlueprintPreview(msg.blueprint)}
                  <span className="text-xs opacity-50 mt-2 block font-mono">
                    {new Date(msg.timestamp).toLocaleTimeString()}
                  </span>
                </div>
              </div>
            ))}
            {isTyping && (
              <div className="flex items-center gap-3 text-gray-400">
                <Loader2 className="w-5 h-5 animate-spin text-[#00d1ff]" />
                <span className="loading-dots font-mono text-sm">
                  <span>Processing</span>
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
            {selectedImage && (
              <div className="mb-3 p-3 bg-[#1a1a1a] rounded-xl border border-[#2a2a2a] flex items-center gap-3">
                <img src={selectedImage.preview} alt="Preview" className="w-12 h-12 object-cover rounded-lg" />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium truncate">{selectedImage.name}</p>
                  <p className="text-xs text-gray-500">Reference image attached</p>
                </div>
                <button onClick={() => setSelectedImage(null)} className="p-1 hover:bg-[#2a2a2a] rounded">
                  <X className="w-4 h-4 text-gray-400" />
                </button>
              </div>
            )}

            <div className="flex gap-3 items-end">
              <input
                ref={fileInputRef}
                type="file"
                onChange={handleImageUpload}
                className="hidden"
                accept="image/*"
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                className={`p-3 rounded-xl transition-all border ${
                  selectedImage
                    ? 'bg-[#00d1ff] text-black border-[#00d1ff]'
                    : 'bg-[#1a1a1a] text-gray-400 border-[#2a2a2a] hover:border-[#00d1ff]'
                }`}
              >
                {selectedImage ? <ImageIcon className="w-5 h-5" /> : <Paperclip className="w-5 h-5" />}
              </button>

              <div className="flex-1 relative">
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                  placeholder="Input design command..."
                  className="w-full bg-[#1a1a1a] text-white placeholder-gray-500 font-mono text-sm rounded-xl pl-5 pr-14 py-4 focus:outline-none focus:ring-2 focus:ring-[#00d1ff]/20 border border-[#2a2a2a] focus:border-[#00d1ff] transition-all"
                />
                <button
                  onClick={handleSend}
                  disabled={(!input.trim() && !selectedImage) || isTyping}
                  className={`absolute right-3 top-3 p-2 rounded-lg transition-all ${
                    (input.trim() || selectedImage) && !isTyping
                      ? 'bg-[#00d1ff] text-black hover:bg-[#00b8e0]'
                      : 'bg-[#2a2a2a] text-gray-500'
                  }`}
                >
                  {isTyping ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Blueprint Preview Panel */}
        <div className="w-80 bg-[#0a0a0a] border-l border-[#2a2a2a] flex flex-col">
          <div className="p-4 border-b border-[#2a2a2a]">
            <h3 className="font-bold text-white flex items-center gap-2">
              <FileText className="w-4 h-4 text-[#00d1ff]" />
              Blueprint Preview
            </h3>
          </div>

          <div className="flex-1 p-4 overflow-auto">
            {blueprintPreview ? (
              <div className="space-y-4">
                {/* Preview Card */}
                <div className="bg-[#1a1a1a] rounded-xl overflow-hidden border border-[#2a2a2a]">
                  <div className="aspect-square bg-gradient-to-br from-[#1a1a1a] to-[#0a0a0a] flex items-center justify-center relative">
                    <div className="text-center">
                      <div className="w-20 h-20 mx-auto mb-3 rounded-full border-2 border-[#00d1ff]/30 flex items-center justify-center">
                        <MapPin className="w-8 h-8 text-[#00d1ff]" />
                      </div>
                      <p className="text-xs font-mono text-gray-500">{blueprintPreview.id}</p>
                    </div>

                                            <FileText size={48} className="text-gray-600" />
                                        )}
                                        {/* Download Badge */}
                                        <div className="absolute top-2 right-2 bg-black/70 backdrop-blur text-white text-[10px] font-bold px-2 py-1 rounded border border-white/10">
                                            DXF
                                        </div>
                                    </div>

                                    {/* Footer Info */}
                                    <div className="p-3 bg-[#1e293b] flex justify-between items-center border-t border-gray-700">
                                        <div className="overflow-hidden mr-2">
                                            <div className="font-bold text-gray-300 text-xs truncate">Blueprint</div>
                                            <div className="text-[10px] text-gray-500 font-mono truncate">{msg.blueprint || "design_v1.dxf"}</div>
                                        </div>
                                        <button className="text-[10px] font-bold bg-[#00D1FF] hover:bg-[#00b8e0] text-white px-3 py-1.5 rounded-full flex items-center gap-1 transition-colors">
                                            <Download size={10} /> GET
                                        </button>
                                    </div>
                                </div>
                            </div>
                        )}
                     </div>
                 )}
              </div>
            </div>
          )}
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
              className="fixed right-0 top-0 bottom-0 w-80 bg-[#111111] border-l border-[#2a2a2a] z-50 p-6"
            >
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-lg font-bold">Export Options</h2>
                <button onClick={() => setShowExport(false)}>
                  <X className="w-5 h-5" />
                </button>
              </div>
              <div className="space-y-4">
                <button className="w-full p-4 bg-[#1a1a1a] hover:bg-[#2a2a2a] rounded-xl border border-[#2a2a2a] text-left transition-colors">
                  <div className="font-bold mb-1">DXF Format</div>
                  <div className="text-xs text-gray-500">AutoCAD compatible</div>
                </button>
                <button className="w-full p-4 bg-[#1a1a1a] hover:bg-[#2a2a2a] rounded-xl border border-[#2a2a2a] text-left transition-colors">
                  <div className="font-bold mb-1">JSON Data</div>
                  <div className="text-xs text-gray-500">Raw topology data</div>
                </button>
                <button className="w-full p-4 bg-[#1a1a1a] hover:bg-[#2a2a2a] rounded-xl border border-[#2a2a2a] text-left transition-colors">
                  <div className="font-bold mb-1">Image Export</div>
                  <div className="text-xs text-gray-500">PNG with annotations</div>
                </button>
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </div>
  );
};

// Main App Component
const App = () => {
  const [view, setView] = useState('landing');

  return (
    <div className="font-sans antialiased text-white bg-[#0a0a0a] selection:bg-[#00d1ff] selection:text-black overflow-x-hidden">
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
