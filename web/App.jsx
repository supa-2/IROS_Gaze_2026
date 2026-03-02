import React, { useState, useEffect, useRef } from 'react';
import { motion, useScroll, useTransform, AnimatePresence, useInView } from 'framer-motion';
import { ArrowRight, Box, Layers, Cpu, ShoppingCart, Menu, X, Globe, Play, CheckCircle, ChevronRight, Zap, Send, Code, Maximize2, RotateCcw, Download, Sparkles, FileText, Ruler, Paperclip, Image as ImageIcon, Loader2, ZoomIn, Check, LayoutGrid, Hammer, Package, Heart, MapPin, User, Eye, Twitter, Github } from 'lucide-react';

// --- API Configuration ---
const apiKey = ""; // Gemini API Key injected by environment

// --- Global Styles & Utils ---
const GridBackground = () => (
  <div className="absolute inset-0 z-0 pointer-events-none overflow-hidden">
    <div className="absolute inset-0 bg-[linear-gradient(to_right,#f0f0f0_1px,transparent_1px),linear-gradient(to_bottom,#f0f0f0_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_0%,#000_70%,transparent_100%)] opacity-70"></div>
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
    <nav className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${scrolled || isCreatorMode ? 'bg-white/90 backdrop-blur-xl border-b border-gray-100 py-4' : 'bg-transparent py-6'}`}>
      <div className="max-w-[1400px] mx-auto px-6 lg:px-12">
        <div className="flex justify-between items-center">
          {/* Logo */}
          <div className="flex items-center gap-3 cursor-pointer group" onClick={onBack}>
            <div className="w-10 h-10 bg-black text-white flex items-center justify-center font-bold text-xl relative overflow-hidden group-hover:bg-[#00D1FF] transition-colors duration-300">
               <span className="relative z-10">P</span>
               <div className="absolute inset-0 bg-white/20 translate-y-full group-hover:translate-y-0 transition-transform duration-300"></div>
            </div>
            <span className="font-bold text-2xl tracking-tighter text-gray-900">PalX</span>
          </div>

          {/* Desktop Menu */}
          {!isCreatorMode ? (
            <div className="hidden md:flex items-center space-x-10">
              {['Technology', 'Workflow', 'Products', 'Pricing'].map((item) => (
                <a key={item} href={`#${item.toLowerCase()}`} className="text-sm font-medium text-gray-500 hover:text-black transition-colors uppercase tracking-wider">
                  {item}
                </a>
              ))}
              <button 
                onClick={onStartCreate}
                className="bg-black text-white px-8 py-3 font-bold hover:bg-[#00D1FF] transition-colors duration-300 flex items-center gap-3">
                <Sparkles size={18} />
                Start Creating
              </button>
            </div>
          ) : (
            <button 
              onClick={onBack}
              className="bg-black text-white px-8 py-3 font-bold hover:bg-[#00D1FF] transition-colors duration-300 flex items-center gap-3">
              <RotateCcw size={18} />
              Back to Home
            </button>
          )}

          {/* Mobile Menu */}
          <button 
            className="md:hidden text-gray-900 hover:text-black"
            onClick={() => setIsOpen(!isOpen)}
          >
            {isOpen ? <X size={24} /> : <Menu size={24} />}
          </button>
        </div>

        {/* Mobile Menu Dropdown */}
        {isOpen && (
          <motion.div 
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="md:hidden mt-6 py-4 space-y-4 border-t border-gray-100"
          >
            {['Technology', 'Workflow', 'Products', 'Pricing'].map((item) => (
              <a key={item} href={`#${item.toLowerCase()}`} className="block text-sm font-medium text-gray-500 hover:text-black transition-colors uppercase tracking-wider">
                {item}
              </a>
            ))}
            <button 
              onClick={onStartCreate}
              className="bg-black text-white w-full py-4 font-bold hover:bg-[#00D1FF] transition-colors duration-300 flex items-center justify-center gap-3">
              <Sparkles size={18} />
              Start Creating
            </button>
          </motion.div>
        )}
      </div>
    </nav>
  );
};

// 2. Hero Section
const Hero = ({ onStartCreate, onOpenIrosgaze }) => {
  return (
    <section className="relative min-h-screen flex items-center bg-white overflow-hidden pt-20">
      <GridBackground />
      <div className="max-w-[1400px] mx-auto px-6 lg:px-12 w-full relative z-10 grid lg:grid-cols-2 gap-16 items-center">
        <motion.div 
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, ease: [0.16, 1, 0.3, 1] }}
          className="w-full"
        >
          <div className="flex items-center gap-3 mb-8">
            <div className="h-[1px] w-12 bg-black"></div>
            <span className="text-xs font-bold tracking-[0.2em] text-gray-400 uppercase">Architecture x AI</span>
          </div>
          <h1 className="text-5xl md:text-7xl lg:text-8xl font-bold text-gray-900 leading-[1.1] tracking-tight mb-8">
            Design Flows <br/>
            Like <span className="text-[#00D1FF]">Language.</span>
          </h1>
          <div className="space-y-4 mb-12 border-l-2 border-gray-200 pl-6">
            <p className="text-xl lg:text-2xl text-gray-900 font-medium">让设计像语言一样自由流动</p>
            <p className="text-gray-500 max-w-md leading-relaxed">
              第一门能直接"造物"的 AI 通用语言。从 Prompt 到实物，只需一句话。
            </p>
          </div>
          <div className="flex flex-col sm:flex-row gap-5">
            <button onClick={onStartCreate} className="group flex items-center justify-center gap-3 bg-black text-white px-10 py-5 font-bold hover:bg-[#00D1FF] transition-all duration-300">
              <Sparkles size={18} />
              试用 PalX Try PalX
            </button>
            <button className="flex items-center justify-center gap-3 border border-gray-200 hover:border-black text-gray-900 px-10 py-5 font-bold transition-all duration-300 bg-white/50 backdrop-blur-sm">
              <Play size={18} fill="currentColor" />
              Watch Demo
            </button>
          </div>
          {/* IROS Gaze Entry Point */}
          <div className="mt-12 pt-8 border-t border-gray-200">
            <p className="text-sm text-gray-500 mb-4">For Research & Analysis</p>
            <button 
              onClick={onOpenIrosgaze}
              className="flex items-center gap-3 bg-gradient-to-r from-[#00D1FF] to-blue-600 text-white px-8 py-4 font-bold hover:from-[#00b8e0] hover:to-blue-500 transition-all duration-300 shadow-lg hover:shadow-xl"
            >
              <Eye size={20} />
              IROS Gaze Analysis
            </button>
          </div>
        </motion.div>
        {/* Visual Animation */}
        <div className="relative h-[60vh] hidden lg:flex items-center justify-center">
            {[0, 1, 2].map((i) => (
              <motion.div
                key={i}
                className="absolute w-[40vw] h-[40vw] border border-gray-200 rounded-full opacity-30"
                animate={{ rotate: i % 2 === 0 ? 360 : -360, scale: [1, 1.05, 1] }}
                transition={{ rotate: { duration: 60 + i * 10, repeat: Infinity, ease: "linear" }, scale: { duration: 10, repeat: Infinity, ease: "easeInOut" } }}
              />
            ))}
            <motion.div
              initial={{ rotateX: 60, rotateZ: 45 }}
              animate={{ rotateZ: [45, 405] }}
              transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
              className="absolute w-[20vw] h-[20vw] border border-gray-200 rounded-full opacity-50"
            />
            <motion.div
              className="absolute w-[10vw] h-[10vw] bg-gradient-to-br from-[#00D1FF] to-blue-600 rounded-full"
              animate={{ scale: [1, 1.2, 1], opacity: [0.8, 1, 0.8] }}
              transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
            />
        </div>
      </div>
    </section>
  );
};

// 3. Workflow Section
const Workflow = () => {
  const steps = [
    { title: "Describe", icon: <Globe />, description: "用自然语言描述你的设计需求" },
    { title: "Generate", icon: <Sparkles />, description: "AI 自动生成多种设计方案" },
    { title: "Refine", icon: <Code />, description: "通过对话迭代优化设计细节" },
    { title: "Export", icon: <Download />, description: "导出为可生产的工程文件" },
  ];

  return (
    <section className="py-24 bg-white" id="workflow">
      <div className="max-w-[1400px] mx-auto px-6 lg:px-12">
        <div className="text-center mb-20">
          <h2 className="text-4xl md:text-6xl font-bold text-gray-900 mb-8">4-Step Design Flow</h2>
          <p className="text-xl text-gray-500 max-w-3xl mx-auto">从概念到实物的无缝衔接，让创意不再受技术限制</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          {steps.map((step, index) => (
            <motion.div
              key={step.title}
              initial={{ opacity: 0, y: 50 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: index * 0.1 }}
              viewport={{ once: true }}
              className="bg-white p-8 rounded-2xl border border-gray-200 hover:border-black hover:shadow-xl transition-all duration-300"
            >
              <div className="w-16 h-16 bg-black text-white flex items-center justify-center rounded-lg mb-6">
                {step.icon}
              </div>
              <h3 className="text-2xl font-bold text-gray-900 mb-4">{step.title}</h3>
              <p className="text-gray-500">{step.description}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

// 4. Technology Section
const TechStack = () => {
  const technologies = [
    { name: "Generative AI", icon: <Cpu />, description: "基于大模型的智能设计生成" },
    { name: "Parametric Design", icon: <Ruler />, description: "参数化驱动的可生产设计" },
    { name: "Real-time Collaboration", icon: <User />, description: "多人实时协同工作流" },
    { name: "Cloud Rendering", icon: <Globe />, description: "云端高性能渲染引擎" },
  ];

  return (
    <section className="py-24 bg-white" id="technology">
      <div className="max-w-[1400px] mx-auto px-6 lg:px-12">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
          <div>
            <div className="flex items-center gap-3 mb-8">
              <div className="h-[1px] w-12 bg-black"></div>
              <span className="text-xs font-bold tracking-[0.2em] text-gray-400 uppercase">Technology</span>
            </div>
            <h2 className="text-4xl md:text-6xl font-bold text-gray-900 mb-8">Powered by Cutting-edge AI</h2>
            <p className="text-xl text-gray-500 mb-12">
              融合生成式AI、参数化设计和实时渲染技术，打造下一代智能设计平台
            </p>
            <button className="bg-black text-white px-10 py-5 font-bold hover:bg-[#00D1FF] transition-colors duration-300 flex items-center gap-3">
              <Code size={18} />
              Explore Technology
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {technologies.map((tech, index) => (
              <motion.div
                key={tech.name}
                initial={{ opacity: 0, x: 50 }}
                whileInView={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                viewport={{ once: true }}
                className="bg-white p-6 rounded-xl border border-gray-200 hover:border-black transition-all duration-300"
              >
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 bg-black text-white flex items-center justify-center rounded-lg">
                    {tech.icon}
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-gray-900">{tech.name}</h3>
                    <p className="text-sm text-gray-500">{tech.description}</p>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

// 5. Products Section
const Products = () => {
  const products = [
    { name: "PalX Pro", price: "$99/month", features: ["无限设计生成", "高级渲染", "团队协作", "API 访问"] },
    { name: "PalX Team", price: "$299/month", features: ["无限团队成员", "企业级安全", "定制化训练", "专属支持"] },
    { name: "PalX Enterprise", price: "Custom", features: ["私有部署", "定制模型", "SLA 保障", "战略合作伙伴"] },
  ];

  return (
    <section className="py-24 bg-white" id="products">
      <div className="max-w-[1400px] mx-auto px-6 lg:px-12">
        <div className="text-center mb-20">
          <h2 className="text-4xl md:text-6xl font-bold text-gray-900 mb-8">Choose Your Plan</h2>
          <p className="text-xl text-gray-500 max-w-3xl mx-auto">
            从个人创作者到企业级部署，我们提供全方位的解决方案
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {products.map((product, index) => (
            <motion.div
              key={product.name}
              initial={{ opacity: 0, y: 50 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: index * 0.1 }}
              viewport={{ once: true }}
              className={`bg-white p-8 rounded-2xl border ${index === 1 ? 'border-black' : 'border-gray-200'} hover:border-black hover:shadow-xl transition-all duration-300 relative overflow-hidden`}
            >
              {index === 1 && (
                <div className="absolute top-0 right-0 bg-black text-white px-6 py-2 text-sm font-bold">
                  MOST POPULAR
                </div>
              )}
              <h3 className="text-2xl font-bold text-gray-900 mb-4">{product.name}</h3>
              <p className="text-4xl font-bold text-gray-900 mb-6">{product.price}</p>
              <ul className="space-y-4 mb-10">
                {product.features.map((feature, i) => (
                  <li key={i} className="flex items-center gap-3">
                    <CheckCircle size={18} className="text-green-500" />
                    <span className="text-gray-900">{feature}</span>
                  </li>
                ))}
              </ul>
              <button className={`w-full py-5 font-bold rounded-lg transition-all duration-300 ${index === 1 ? 'bg-black text-white' : 'bg-white border border-gray-200 text-gray-900 hover:bg-black hover:text-white'}`}>
                Get Started
              </button>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

// 6. Pricing Section
const Pricing = () => {
  return (
    <section className="py-24 bg-white" id="pricing">
      <div className="max-w-[1400px] mx-auto px-6 lg:px-12">
        <div className="text-center mb-20">
          <h2 className="text-4xl md:text-6xl font-bold text-gray-900 mb-8">Simple Transparent Pricing</h2>
          <p className="text-xl text-gray-500 max-w-3xl mx-auto">
            无隐藏费用，按使用量付费，随时升级或降级
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-12">
          <div className="bg-white p-10 rounded-2xl border border-gray-200 hover:border-black transition-all duration-300">
            <h3 className="text-2xl font-bold text-gray-900 mb-6">Pay-as-you-go</h3>
            <p className="text-xl text-gray-500 mb-8">
              适合偶尔使用的个人创作者
            </p>
            <div className="space-y-6">
              <div className="flex justify-between items-center">
                <span className="text-gray-900">设计生成</span>
                <span className="text-gray-900 font-bold">$0.99/次</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-900">渲染时长</span>
                <span className="text-gray-900 font-bold">$0.01/分钟</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-900">文件导出</span>
                <span className="text-gray-900 font-bold">免费</span>
              </div>
            </div>
          </div>

          <div className="bg-white p-10 rounded-2xl border border-gray-200 hover:border-black transition-all duration-300">
            <h3 className="text-2xl font-bold text-gray-900 mb-6">Subscription</h3>
            <p className="text-xl text-gray-500 mb-8">
              适合频繁使用的专业设计师
            </p>
            <div className="space-y-6">
              <div className="flex justify-between items-center">
                <span className="text-gray-900">无限生成</span>
                <span className="text-gray-900 font-bold">$99/月</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-900">无限渲染</span>
                <span className="text-gray-900 font-bold">包含在订阅中</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-900">优先支持</span>
                <span className="text-gray-900 font-bold">包含在订阅中</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

// 7. Footer Section
const Footer = () => {
  return (
    <footer className="py-16 bg-white border-t border-gray-200">
      <div className="max-w-[1400px] mx-auto px-6 lg:px-12">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-12">
          <div>
            <div className="flex items-center gap-3 mb-6">
              <div className="w-10 h-10 bg-black text-white flex items-center justify-center font-bold text-xl">
                P
              </div>
              <span className="font-bold text-2xl tracking-tighter text-gray-900">PalX</span>
            </div>
            <p className="text-gray-500 mb-6">
              让设计像语言一样自由流动
            </p>
            <div className="flex items-center gap-4">
              <a href="#" className="text-gray-500 hover:text-black transition-colors">
                <Globe size={20} />
              </a>
              <a href="#" className="text-gray-500 hover:text-black transition-colors">
                <Twitter size={20} />
              </a>
              <a href="#" className="text-gray-500 hover:text-black transition-colors">
                <Github size={20} />
              </a>
            </div>
          </div>

          <div>
            <h3 className="text-sm font-bold text-gray-900 mb-6 uppercase tracking-wider">Product</h3>
            <ul className="space-y-4">
              <li><a href="#" className="text-gray-500 hover:text-black transition-colors">Features</a></li>
              <li><a href="#" className="text-gray-500 hover:text-black transition-colors">Pricing</a></li>
              <li><a href="#" className="text-gray-500 hover:text-black transition-colors">Documentation</a></li>
              <li><a href="#" className="text-gray-500 hover:text-black transition-colors">API</a></li>
            </ul>
          </div>

          <div>
            <h3 className="text-sm font-bold text-gray-900 mb-6 uppercase tracking-wider">Company</h3>
            <ul className="space-y-4">
              <li><a href="#" className="text-gray-500 hover:text-black transition-colors">About</a></li>
              <li><a href="#" className="text-gray-500 hover:text-black transition-colors">Blog</a></li>
              <li><a href="#" className="text-gray-500 hover:text-black transition-colors">Careers</a></li>
              <li><a href="#" className="text-gray-500 hover:text-black transition-colors">Contact</a></li>
            </ul>
          </div>

          <div>
            <h3 className="text-sm font-bold text-gray-900 mb-6 uppercase tracking-wider">Legal</h3>
            <ul className="space-y-4">
              <li><a href="#" className="text-gray-500 hover:text-black transition-colors">Privacy</a></li>
              <li><a href="#" className="text-gray-500 hover:text-black transition-colors">Terms</a></li>
              <li><a href="#" className="text-gray-500 hover:text-black transition-colors">Cookies</a></li>
              <li><a href="#" className="text-gray-500 hover:text-black transition-colors">Licenses</a></li>
            </ul>
          </div>
        </div>

        <div className="mt-16 pt-8 border-t border-gray-200 flex flex-col md:flex-row justify-between items-center">
          <p className="text-gray-500 mb-6 md:mb-0">
            © 2026 PalX. All rights reserved.
          </p>
          <div className="flex items-center gap-6">
            <a href="#" className="text-gray-500 hover:text-black transition-colors text-sm">Privacy</a>
            <a href="#" className="text-gray-500 hover:text-black transition-colors text-sm">Terms</a>
            <a href="#" className="text-gray-500 hover:text-black transition-colors text-sm">Cookies</a>
          </div>
        </div>
      </div>
    </footer>
  );
};

// 8. Landing Page Container
const LandingPage = ({ onStartCreate, onOpenIrosgaze }) => {
  return (
    <motion.div 
      initial={{ opacity: 0 }} 
      animate={{ opacity: 1 }} 
      exit={{ opacity: 0 }}
      className="w-full"
    >
      <Hero onStartCreate={onStartCreate} onOpenIrosgaze={onOpenIrosgaze} />
      <Workflow />
      <TechStack />
      <Products />
      <Pricing />
      <Footer />
    </motion.div>
  );
};

// 9. Creator App Component (Powered by Gemini)
const CreatorApp = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [selectedImage, setSelectedImage] = useState(null); // 上传图片状态
  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);
  
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages]);

  // Handle fake image upload
  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      // Create a fake URL for preview
      const reader = new FileReader();
      reader.onload = (e) => {
        setSelectedImage(e.target.result);
      };
      reader.readAsDataURL(file);
    }
  };

  // --- Gemini API Function (STRICT TERMINAL MODE with PRE-DESIGNED FALLBACK) ---
  const generateContent = async (userPrompt, hasImage) => {
    
    // ----------------------------------------------------
    // PRE-DESIGNED SCENARIOS (预设剧本)
    // ----------------------------------------------------
    const promptLower = userPrompt.toLowerCase();
    
    // Scenario 1: 童话床 (Fairy Tale Bed)
    if (promptLower.includes("童话") || promptLower.includes("床") || promptLower.includes("bed") || promptLower.includes("fairy")) {
        // Return highly detailed pre-designed data
        return {
            response: `SYSTEM: BLUEPRINT GENERATED [Project: Fairy_Castle_Bed_v1]
            
            技术参数 (TECHNICAL SPECS):
            --------------------------------
            总尺寸 (Dimensions): 2215(L) x 1351(W) x 1000(H) mm
            主体材料 (Material): 欧洲进口橡木 (European Oak Wood)
            表面处理 (Finish): 环保水性清漆
            
            结构指标 (STRUCTURAL):
            --------------------------------
            结构节点 (Joints): 36个 榫卯+金属连接件 (SC6 Sart Joint)
            静态承重 (Load Cap): 350 kg
            装配部件 (Parts): 护栏、床板、树形装饰侧板
            
            生产数据 (PRODUCTION):
            --------------------------------
            加工耗时 (Machining): 5.5 小时 (5轴 CNC)
            紧固件 (Fasteners): M6x105mm, M8x163mm 螺栓
            碳足迹 (Carbon): -18.2 kgCO2e
            安全倒角 (Safety): 全边倒角处理 (Child Safe)`,
            gcode_preview: `G21 ; Millimeters
G90 ; Absolute positioning
; --- TOOLPATH: BED_SIDE_PANEL_TREE_PATTERN ---
G00 Z50.00
G00 X0 Y0
M03 S15000 ; Spindle ON
G01 Z-18.00 F800 ; Cut Depth 18mm
G01 X108.38 Y200.50
G02 X150.00 Y300.00 R200.00
; --- DRILL: M6 HOLES ---`,
            blueprint_name: "Fairy_Bed_Assembly.dxf",
            // 使用蓝图样式的占位图 (深蓝背景白线)
            blueprint_image: "https://placehold.co/600x400/003366/ffffff/png?text=Fairy+Tale+Bed+Blueprint"
        };
    }

    // Scenario 2: 非遗灯 (Heritage Lamp) - Triggered by image or keywords
    if (hasImage || promptLower.includes("非遗") || promptLower.includes("灯") || promptLower.includes("lamp") || promptLower.includes("heritage")) {
         return {
             response: `SYSTEM: BLUEPRINT GENERATED [Project: Heritage_Lamp_v2]
             
             技术参数 (TECHNICAL SPECS):
             --------------------------------
             总尺寸 (Dimensions): 320(L) x 320(W) x 650(H) mm
             主体材料 (Material): 非遗竹编 + 亚克力板
             光源类型 (Light Source): 24V 12W LED 暖光
             
             结构指标 (STRUCTURAL):
             --------------------------------
             编织密度 (Weave Density): 12根/cm² 非遗工艺
             透光率 (Transmittance): 78% (均匀漫射)
             防护等级 (IP Rating): IP20 (室内使用)
             
             生产数据 (PRODUCTION):
             --------------------------------
             手工耗时 (Handwork): 8.5 小时 (非遗传承人)
             能耗 (Power): 12W (节能模式)
             碳足迹 (Carbon): -32.7 kgCO2e
             生命周期 (Lifespan): 50,000 小时`,
             gcode_preview: `G21 ; Millimeters
G90 ; Absolute positioning
; --- TOOLPATH: LAMP_BASE_CUTOUT ---
G00 Z50.00
G00 X0 Y0
M03 S12000 ; Spindle ON
G01 Z-15.00 F600 ; Cut Depth 15mm
G01 X160.00 Y0
G03 X160.00 Y0 I-160.00 J0
; --- DRILL: LED MOUNT HOLES ---`,
             blueprint_name: "Heritage_Lamp_Assembly.dxf",
             blueprint_image: "https://placehold.co/600x400/003366/ffffff/png?text=Heritage+Lamp+Blueprint"
         };
    }

    // Scenario 3: 科幻椅 (Sci-fi Chair)
    if (promptLower.includes("科幻") || promptLower.includes("椅") || promptLower.includes("chair") || promptLower.includes("sci-fi")) {
        return {
            response: `SYSTEM: BLUEPRINT GENERATED [Project: Sci-fi_Chair_v3]
            
            技术参数 (TECHNICAL SPECS):
            --------------------------------
            总尺寸 (Dimensions): 850(L) x 720(W) x 1100(H) mm
            主体材料 (Material): 碳纤维增强塑料 (CFRP)
            表面处理 (Finish): 哑光黑 + 金属拉丝
            
            结构指标 (STRUCTURAL):
            --------------------------------
            承重能力 (Load Cap): 250 kg
            调节角度 (Adjustment): 0-135° 无级调节
            扶手高度 (Armrest): 720-820 mm (电动升降)
            
            生产数据 (PRODUCTION):
            --------------------------------
            成型工艺 (Molding): 高压树脂传递模塑 (RTM)
            组装时间 (Assembly): 12 分钟/台
            碳足迹 (Carbon): 18.5 kgCO2e
            认证标准 (Certification): BIFMA X5.1`,
            gcode_preview: `G21 ; Millimeters
G90 ; Absolute positioning
; --- TOOLPATH: CHAIR_FRAME_CUT ---
G00 Z50.00
G00 X0 Y0
M03 S18000 ; Spindle ON
G01 Z-20.00 F1000 ; Cut Depth 20mm
G01 X425.00 Y0
G02 X425.00 Y0 I-425.00 J0
; --- DRILL: ADJUSTMENT MECHANISM ---`,
            blueprint_name: "Sci-fi_Chair_Assembly.dxf",
            blueprint_image: "https://placehold.co/600x400/003366/ffffff/png?text=Sci-fi+Chair+Blueprint"
        };
    }

    // Scenario 4: 通用场景 (General Case) - Return minimal valid response
    return {
        response: `SYSTEM: DESIGN GENERATED [Project: Generic_Design_v1]
        
        技术参数 (TECHNICAL SPECS):
        --------------------------------
        状态 (Status): 设计已生成
        类型 (Type): ${hasImage ? "图像驱动" : "文本驱动"}
        复杂度 (Complexity): 中等
        
        生产建议 (PRODUCTION):
        --------------------------------
        材料 (Material): 根据设计自动匹配
        工艺 (Process): CNC + 手工组装
        成本 (Cost): 中等范围
        碳足迹 (Carbon): 优化中`,
        gcode_preview: `G21 ; Millimeters
G90 ; Absolute positioning
; --- TOOLPATH: GENERIC_CUT ---
G00 Z50.00
G00 X0 Y0
M03 S15000 ; Spindle ON
G01 Z-10.00 F800 ; Cut Depth 10mm
G01 X100.00 Y0
G01 X100.00 Y100.00
G01 X0 Y100.00
G01 X0 Y0`,
        blueprint_name: "Generic_Design.dxf",
        blueprint_image: "https://placehold.co/600x400/003366/ffffff/png?text=Design+Blueprint"
    };
  };

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() && !selectedImage) return;

    // Add user message to chat
    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: input,
      timestamp: new Date(),
      image: selectedImage, // 包含图片预览
    };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsTyping(true);
    setSelectedImage(null); // 上传后清空

    try {
      // Simulate API call delay
      await new Promise(resolve => setTimeout(resolve, 1500));

      // Get pre-designed response based on prompt
      const response = await generateContent(input, !!selectedImage);

      // Add assistant message to chat
      const assistantMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: response.response,
        timestamp: new Date(),
        gcode: response.gcode_preview,
        blueprint: {
          name: response.blueprint_name,
          image: response.blueprint_image
        }
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error generating content:', error);
    } finally {
      setIsTyping(false);
    }
  };

  return (
    <div className="h-screen flex flex-col bg-gray-50">
      <div className="flex-1 overflow-hidden">
        {/* Messages Container */}
        <div className="h-full overflow-y-auto p-6 space-y-6">
          {messages.map((message) => (
            <div key={message.id} className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-[80%] rounded-lg px-4 py-3 ${message.role === 'user' ? 'bg-black text-white' : 'bg-white border border-gray-200'}`}>
                {/* Display image if present (only for user messages) */}
                {message.image && (
                  <div className="mb-3 rounded overflow-hidden">
                    <img src={message.image} alt="Uploaded" className="max-w-full h-auto" />
                  </div>
                )}
                <p className="whitespace-pre-wrap">{message.content}</p>
                <span className="text-xs opacity-40 mt-2 block">{new Date(message.timestamp).toLocaleTimeString()}</span>
                
                {/* Display G-code if available (assistant messages only) */}
                {message.gcode && (
                  <div className="mt-4 p-3 bg-gray-100 text-gray-900 rounded text-xs font-mono whitespace-pre overflow-x-auto">
                    {message.gcode}
                  </div>
                )}
                
                {/* Display blueprint if available (assistant messages only) */}
                {message.blueprint && (
                  <div className="mt-4">
                    <p className="text-xs font-bold mb-2">{message.blueprint.name}</p>
                    <div className="rounded overflow-hidden border border-gray-200">
                      <img src={message.blueprint.image} alt={message.blueprint.name} className="w-full h-auto" />
                    </div>
                  </div>
                )}
              </div>
            </div>
          ))}
          
          {/* Typing indicator */}
          {isTyping && (
            <div className="flex justify-start">
              <div className="bg-white border border-gray-200 rounded-lg px-4 py-3">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                </div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <div className="border-t border-gray-200 p-4 bg-white">
        {/* Image Preview (if selected) */}
        {selectedImage && (
          <div className="mb-3 p-2 bg-gray-100 rounded-lg flex items-center gap-3">
            <img src={selectedImage} alt="Preview" className="w-12 h-12 object-cover rounded" />
            <span className="text-sm text-gray-900">图片已上传，可直接发送</span>
            <button 
              onClick={() => setSelectedImage(null)}
              className="ml-auto text-gray-500 hover:text-red-500"
            >
              <X size={20} />
            </button>
          </div>
        )}
        
        <form onSubmit={handleSubmit} className="flex gap-2">
          {/* Hidden file input */}
          <input
            ref={fileInputRef}
            type="file"
            onChange={handleImageUpload}
            className="hidden"
            accept="image/*"
          />
          
          {/* Image upload button */}
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="p-2 rounded-lg border border-gray-300 hover:bg-gray-100 transition-colors"
          >
            <ImageIcon size={20} />
          </button>
          
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="描述你的设计需求..."
            className="flex-1 px-4 py-2 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-black"
          />
          <button
            type="submit"
            disabled={!input.trim() && !selectedImage}
            className="px-4 py-2 bg-black text-white rounded-lg hover:bg-gray-800 transition-colors disabled:bg-gray-300"
          >
            <Send size={20} />
          </button>
        </form>
      </div>
    </div>
  );
};

// 10. IROS Gaze App Component
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

// Main App Component
const App = () => {
  const [view, setView] = useState('landing'); 

  return (
    <div className="font-sans antialiased text-gray-900 bg-white selection:bg-black selection:text-white overflow-x-hidden min-h-screen w-full">
      <Navbar 
        onStartCreate={() => setView('creator')} 
        onBack={() => setView('landing')}
        isCreatorMode={view === 'creator'}
      />
      
      <AnimatePresence mode="wait">
        {view === 'landing' ? (
          <LandingPage 
            key="landing" 
            onStartCreate={() => setView('creator')}
            onOpenIrosgaze={() => setView('irosgaze')}
          />
        ) : view === 'creator' ? (
          <CreatorApp key="creator" />
        ) : (
          <IrosgazeApp key="irosgaze" onBack={() => setView('landing')} />
        )}
      </AnimatePresence>
    </div>
  );
};

export default App;