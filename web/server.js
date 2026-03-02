/**
 * IROS Gaze Backend Server
 * Handles API requests for image processing and gaze visualization
 * Integrated with Qwen API for LLM capabilities
 */

import express from 'express';
import cors from 'cors';
import multer from 'multer';
import { v4 as uuidv4 } from 'uuid';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import fs from 'fs';
import dotenv from 'dotenv';
import { spawn } from 'child_process';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Load environment variables from parent directory
const envPath = join(__dirname, '../.env');
dotenv.config({ path: envPath });

const app = express();
const PORT = process.env.PORT || 5000;

// Qwen API Configuration
const QWEN_API_KEY = process.env.QWEN_API_KEY || '';
const QWEN_BASE_URL = process.env.QWEN_BASE_URL || 'https://dashscope.aliyuncs.com/compatible-mode/v1';
const LLM_MODEL = process.env.LLM_MODEL || 'qwen-plus';
const VLM_MODEL = process.env.VLM_MODEL || 'qwen-vl-max-latest';

console.log(`[+] Qwen API configured: ${QWEN_BASE_URL}`);
console.log(`[+] LLM Model: ${LLM_MODEL}`);
console.log(`[+] VLM Model: ${VLM_MODEL}`);

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Static files
app.use('/outputs', express.static(join(__dirname, '../data/outputs')));
app.use('/uploads', express.static(join(__dirname, '../data/uploads')));

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = join(__dirname, 'data/uploads');
    fs.mkdirSync(uploadDir, { recursive: true });
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    const uniqueId = uuidv4();
    const ext = file.originalname.split('.').pop();
    cb(null, `${uniqueId}.${ext}`);
  },
});

const upload = multer({
  storage,
  limits: { fileSize: 50 * 1024 * 1024 }, // 50MB
  fileFilter: (req, file, cb) => {
    if (file.mimetype.startsWith('image/')) {
      cb(null, true);
    } else {
      cb(new Error('Only image files are allowed'));
    }
  },
});

// ==================== Qwen API Functions ====================

/**
 * Call Qwen API for chat completion
 */
async function callQwenAPI(messages, model = LLM_MODEL) {
  try {
    const response = await fetch(`${QWEN_BASE_URL}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${QWEN_API_KEY}`,
      },
      body: JSON.stringify({
        model,
        messages,
        temperature: parseFloat(process.env.LLM_TEMPERATURE || '0.7'),
        max_tokens: parseInt(process.env.LLM_MAX_TOKENS || '2000'),
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Qwen API error: ${response.status} - ${error}`);
    }

    const data = await response.json();
    return data.choices[0].message.content;
  } catch (error) {
    console.error('Qwen API call failed:', error);
    throw error;
  }
}

/**
 * Call Qwen Vision API for image analysis
 */
async function callQwenVisionAPI(imageBase64, prompt) {
  try {
    const response = await fetch(`${QWEN_BASE_URL}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${QWEN_API_KEY}`,
      },
      body: JSON.stringify({
        model: VLM_MODEL,
        messages: [
          {
            role: 'system',
            content: 'You are an eye-tracking analysis expert. Analyze the given image and provide insights about gaze patterns, attention hotspots, and visual flow.',
          },
          {
            role: 'user',
            content: [
              { type: 'text', text: prompt },
              { type: 'image_url', image_url: { url: imageBase64 } },
            ],
          },
        ],
        temperature: 0.5,
        max_tokens: 1500,
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Qwen Vision API error: ${response.status} - ${error}`);
    }

    const data = await response.json();
    return data.choices[0].message.content;
  } catch (error) {
    console.error('Qwen Vision API call failed:', error);
    throw error;
  }
}

/**
 * Call Python agent for gaze prediction
 */
function callPythonAgent(script, args = []) {
  return new Promise((resolve, reject) => {
    const pythonProcess = spawn('python', [join(__dirname, '..', script), ...args], {
      cwd: join(__dirname, '..'),
    });

    let stdout = '';
    let stderr = '';

    pythonProcess.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    pythonProcess.on('close', (code) => {
      if (code === 0) {
        resolve(stdout);
      } else {
        reject(new Error(`Python script failed: ${stderr}`));
      }
    });

    pythonProcess.on('error', (err) => {
      reject(err);
    });
  });
}

// ==================== Routes ====================

// Health check
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    timestamp: new Date().toISOString(),
    qwen_configured: !!QWEN_API_KEY,
    model: LLM_MODEL,
  });
});

// Get system info
app.get('/api/info', (req, res) => {
  res.json({
    system: 'IROS Gaze',
    version: '1.0.0',
    models: {
      llm: LLM_MODEL,
      vlm: VLM_MODEL,
    },
    capabilities: ['chat', 'heatmap', 'trajectory', 'prediction'],
  });
});

// Chat endpoint with Qwen API
app.post('/api/chat', async (req, res) => {
  try {
    const { message, history } = req.body;

    // Build messages array for Qwen
    const messages = [
      {
        role: 'system',
        content: `You are IROS Gaze Assistant, an expert in eye-tracking analysis and visualization. You help users understand gaze patterns, heatmaps, and trajectory visualizations.

Key capabilities:
- Analyze eye-tracking heatmaps to identify attention hotspots
- Interpret gaze trajectory patterns
- Provide insights on visual attention distribution
- Answer questions about museum/exhibition visitor behavior

Be helpful, concise, and technically accurate.`,
      },
    ];

    // Add history if provided
    if (history && Array.isArray(history)) {
      messages.push(...history.slice(-10)); // Keep last 10 messages for context
    }

    // Add current message
    messages.push({ role: 'user', content: message });

    // Call Qwen API
    const response = await callQwenAPI(messages);

    res.json({
      role: 'assistant',
      content: response,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error('Chat error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Analyze image with Qwen Vision API
app.post('/api/analyze-image', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No image uploaded' });
    }

    const { prompt = 'Analyze this image and describe the likely gaze patterns and attention hotspots. Where would viewers look first? What areas would attract the most attention?' } = req.body;

    // Convert image to base64
    const imageBuffer = fs.readFileSync(req.file.path);
    const imageBase64 = `data:${req.file.mimetype};base64,${imageBuffer.toString('base64')}`;

    // Call Qwen Vision API
    const analysis = await callQwenVisionAPI(imageBase64, prompt);

    res.json({
      success: true,
      analysis,
      imageInfo: {
        filename: req.file.filename,
        size: req.file.size,
        mimetype: req.file.mimetype,
      },
    });
  } catch (error) {
    console.error('Image analysis error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Generate heatmap visualization (calls Python backend)
app.post('/api/heatmap', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No image uploaded' });
    }

    const { gazeData } = req.body;

    // Create output directory
    const outputPath = join(__dirname, 'data/outputs/heatmaps');
    fs.mkdirSync(outputPath, { recursive: true });

    const resultId = uuidv4();
    const outputFile = join(outputPath, `${resultId}.png`);

    // For demo: copy the uploaded file
    // In production, you would call the Python visualization script
    fs.copyFileSync(req.file.path, outputFile);

    res.json({
      success: true,
      id: resultId,
      heatmapUrl: `/outputs/heatmaps/${resultId}.png`,
      originalUrl: `/uploads/${req.file.filename}`,
      message: 'Heatmap generated successfully.',
    });
  } catch (error) {
    console.error('Heatmap error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Generate trajectory visualization
app.post('/api/trajectory', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No image uploaded' });
    }

    const { gazePoints } = req.body;

    const outputPath = join(__dirname, 'data/outputs/trajectories');
    fs.mkdirSync(outputPath, { recursive: true });

    const resultId = uuidv4();
    const outputFile = join(outputPath, `${resultId}.png`);

    // For demo: copy the uploaded file
    fs.copyFileSync(req.file.path, outputFile);

    res.json({
      success: true,
      id: resultId,
      trajectoryUrl: `/outputs/trajectories/${resultId}.png`,
      originalUrl: `/uploads/${req.file.filename}`,
      message: 'Trajectory generated successfully.',
    });
  } catch (error) {
    console.error('Trajectory error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Generate both heatmap and trajectory
app.post('/api/analyze', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No image uploaded' });
    }

    const { options } = req.body;

    // Create output directories
    const heatmapDir = join(__dirname, 'data/outputs/heatmaps');
    const trajectoryDir = join(__dirname, 'data/outputs/trajectories');
    fs.mkdirSync(heatmapDir, { recursive: true });
    fs.mkdirSync(trajectoryDir, { recursive: true });

    const resultId = uuidv4();
    const heatmapFile = join(heatmapDir, `${resultId}_heatmap.png`);
    const trajectoryFile = join(trajectoryDir, `${resultId}_trajectory.png`);

    // For demo: copy the uploaded file
    fs.copyFileSync(req.file.path, heatmapFile);
    fs.copyFileSync(req.file.path, trajectoryFile);

    res.json({
      success: true,
      id: resultId,
      heatmapUrl: `/outputs/heatmaps/${resultId}_heatmap.png`,
      trajectoryUrl: `/outputs/trajectories/${resultId}_trajectory.png`,
      originalUrl: `/uploads/${req.file.filename}`,
      message: 'Analysis complete.',
    });
  } catch (error) {
    console.error('Analysis error:', error);
    res.status(500).json({ error: error.message });
  }
});

// ==================== IROS Gaze Processing API ====================

// Full IROS Gaze analysis with SAM2 + VLM + Heatmap + Trajectory
app.post('/api/iros-gaze/analyze', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No image uploaded' });
    }

    const { 
      maskPrompt = 'auto',
      vlmPrompt = 'auto',
      options = {}
    } = req.body;

    // Create output directory
    const resultId = uuidv4();
    const outputDir = join(__dirname, `../data/outputs/iros-gaze/${resultId}`);
    fs.mkdirSync(outputDir, { recursive: true });

    console.log(`[IROS Gaze] Processing image: ${req.file.path}`);
    console.log(`[IROS Gaze] Output directory: ${outputDir}`);

    // Call Python IROS Gaze Processor
    const pythonScript = join(__dirname, '../scripts/iros_gaze_processor.py');
    const configPath = join(__dirname, '../config.json');
    
    const args = [
      pythonScript,
      '--image', req.file.path,
      '--output', outputDir,
      '--mask-prompt', maskPrompt,
      '--vlm-prompt', vlmPrompt,
    ];

    // Add config if exists
    if (fs.existsSync(configPath)) {
      args.push('--config', configPath);
    }

    console.log(`[IROS Gaze] Running: python ${args.join(' ')}`);

    const result = await callPythonScript(args);
    
    // Parse the JSON result from stdout
    let analysisResult;
    try {
      // Extract JSON from output (look for the last JSON block)
      const jsonMatch = result.stdout.match(/\{[\s\S]*"outputs"[\s\S]*\}/);
      if (jsonMatch) {
        analysisResult = JSON.parse(jsonMatch[0]);
      } else {
        throw new Error('Could not parse Python output');
      }
    } catch (parseError) {
      console.error('Failed to parse Python output:', parseError);
      console.log('Raw output:', result.stdout);
      throw new Error('Failed to parse analysis result');
    }

    // Convert file paths to URLs
    const response = {
      success: true,
      id: resultId,
      segmentation: analysisResult.segmentation,
      topology: analysisResult.topology,
      outputs: {
        heatmap: analysisResult.outputs.heatmap 
          ? `/outputs/iros-gaze/${resultId}/heatmap.png`
          : null,
        trajectory: analysisResult.outputs.trajectory
          ? `/outputs/iros-gaze/${resultId}/trajectory.png`
          : null,
        original: `/uploads/${req.file.filename}`
      },
      message: 'IROS Gaze analysis completed successfully'
    };

    res.json(response);
  } catch (error) {
    console.error('[IROS Gaze] Analysis error:', error);
    res.status(500).json({ 
      error: error.message,
      details: error.stack
    });
  }
});

/**
 * Helper function to call Python scripts
 */
function callPythonScript(args) {
  return new Promise((resolve, reject) => {
    const pythonProcess = spawn('python', args, {
      cwd: join(__dirname, '..'),
      env: { ...process.env, PYTHONUTF8: '1' }
    });

    let stdout = '';
    let stderr = '';

    pythonProcess.stdout.on('data', (data) => {
      stdout += data.toString();
      console.log('[Python]', data.toString().trim());
    });

    pythonProcess.stderr.on('data', (data) => {
      stderr += data.toString();
      console.error('[Python Error]', data.toString().trim());
    });

    pythonProcess.on('close', (code) => {
      resolve({ stdout, stderr, code });
    });

    pythonProcess.on('error', (err) => {
      reject(err);
    });
  });
}

// Get prediction from gaze history (calls Python agent)
app.post('/api/predict', async (req, res) => {
  try {
    const { gazeHistory, currentExhibit } = req.body;

    // Use Qwen API to predict next exhibit
    const prompt = `Given the following gaze history from a museum visitor, predict the next exhibit they are likely to visit.

Gaze History: ${JSON.stringify(gazeHistory || [])}
Current Exhibit: ${currentExhibit || 'Unknown'}

Exhibits in the museum:
- 入口: The entrance area
- 丁香花: A painting of white lilacs in a white round pot
- 金鱼兰: A plant with long slender leaves and goldfish-shaped flowers in a terracotta pot
- 牡丹花: Peony flowers with rich, saturated colors and delicate petal layers
- 说明文字-千岛湖: Information panel about Qiandao Lake from 1980s
- 玉兰花开: White magnolia flowers blooming on a tree, displayed on a black wall
- 人物-祝大年创作: Figure paintings by Zhu Danian depicting Dai people in Xishuangbanna
- 自序: Curator's autograph preface displayed on white wall
- 松竹海: Painting featuring pine and bamboo themes
- 西双版纳: Painting depicting Xishuangbanna tropical rainforest scene
- 千岛湖: Artwork with Qiandao Lake theme
- 山茶花: Camellia flower painting displayed on a column
- 耕织图: Ancient Chinese theme of farming and weaving

Respond with JSON in this format:
{
  "prediction_id": "exhibit_id",
  "prediction_name": "exhibit name",
  "reasoning": "brief explanation",
  "confidence": 0.0-1.0,
  "attention_level": "A/B/C/D/E"
}`;

    const messages = [
      {
        role: 'system',
        content: 'You are an expert in spatial gaze prediction and museum visitor behavior analysis. Predict the next exhibit based on gaze history and spatial topology.',
      },
      { role: 'user', content: prompt },
    ];

    const response = await callQwenAPI(messages);

    // Try to extract JSON from response
    let prediction;
    try {
      const jsonMatch = response.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        prediction = JSON.parse(jsonMatch[0]);
      } else {
        throw new Error('No JSON found in response');
      }
    } catch (e) {
      // Fallback: create a prediction from the text
      prediction = {
        prediction_id: 'unknown',
        prediction_name: 'Unknown',
        reasoning: response.substring(0, 200),
        confidence: 0.5,
        attention_level: 'C',
      };
    }

    res.json({
      success: true,
      prediction,
    });
  } catch (error) {
    console.error('Prediction error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Error handler
app.use((error, req, res, next) => {
  console.error('Server error:', error);
  res.status(500).json({ error: error.message || 'Internal server error' });
});

// Start server with error handling for port conflicts
const server = app.listen(PORT, () => {
  console.log(`\n${'='.repeat(60)}`);
  console.log(`🚀 IROS Gaze Server running on port ${PORT}`);
  console.log(`📡 API endpoint: http://localhost:${PORT}/api`);
  console.log(`🤖 Qwen API: ${QWEN_BASE_URL}`);
  console.log(`📁 Uploads directory: ${join(__dirname, 'data/uploads')}`);
  console.log(`📁 Outputs directory: ${join(__dirname, 'data/outputs')}`);
  console.log(`${'='.repeat(60)}\n`);
});

server.on('error', (err) => {
  if (err.code === 'EADDRINUSE') {
    console.error(`❌ Port ${PORT} is already in use!`);
    console.error(`   Please either:`);
    console.error(`   1. Stop the process using port ${PORT}`);
    console.error(`   2. Or change the PORT in .env file`);
    process.exit(1);
  }
  throw err;
});
