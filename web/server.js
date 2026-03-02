/**
 * IROS Gaze Backend Server
 * Handles API requests for image processing and gaze visualization
 */

import express from 'express';
import cors from 'cors';
import multer from 'multer';
import { v4 as uuidv4 } from 'uuid';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import fs from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Static files
app.use('/outputs', express.static(join(__dirname, '../data/outputs')));

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = join(__dirname, '../data/uploads');
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

// Import agent functions
let agentInstance = null;

function getAgent() {
  if (!agentInstance) {
    // Import agent dynamically to avoid issues if not available
    try {
      const agentPath = join(__dirname, '../agent.py');
      // We'll use subprocess to call Python scripts
      console.log('Agent module will be called via subprocess');
    } catch (e) {
      console.error('Agent module not available:', e.message);
    }
  }
  return agentInstance;
}

// Routes

// Health check
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Chat endpoint
app.post('/api/chat', async (req, res) => {
  try {
    const { message, history } = req.body;

    // Simple response for now
    const response = {
      role: 'assistant',
      content: `I received your message: "${message}". The chat functionality is ready to be connected to your AI backend.`,
      timestamp: new Date().toISOString(),
    };

    res.json(response);
  } catch (error) {
    console.error('Chat error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Upload image and generate heatmap
app.post('/api/heatmap', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No image uploaded' });
    }

    const { gazeData } = req.body;
    const imagePath = req.file.path;

    // For now, return a placeholder response
    // In production, this would call the Python backend
    const outputPath = join(__dirname, '../data/outputs/heatmaps');
    fs.mkdirSync(outputPath, { recursive: true });

    const resultId = uuidv4();
    const resultPath = join(outputPath, `${resultId}.png`);

    // Placeholder: Copy the uploaded file as a demo
    fs.copyFileSync(imagePath, resultPath);

    res.json({
      success: true,
      id: resultId,
      heatmapUrl: `/api/outputs/heatmaps/${resultId}.png`,
      originalUrl: `/api/uploads/${req.file.filename}`,
      message: 'Heatmap generation complete. Connect to Python backend for actual processing.',
    });
  } catch (error) {
    console.error('Heatmap error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Upload image and generate trajectory
app.post('/api/trajectory', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No image uploaded' });
    }

    const { gazePoints } = req.body;

    const outputPath = join(__dirname, '../data/outputs/trajectories');
    fs.mkdirSync(outputPath, { recursive: true });

    const resultId = uuidv4();
    const resultPath = join(outputPath, `${resultId}.png`);

    // Placeholder: Copy the uploaded file as a demo
    fs.copyFileSync(req.file.path, resultPath);

    res.json({
      success: true,
      id: resultId,
      trajectoryUrl: `/api/outputs/trajectories/${resultId}.png`,
      originalUrl: `/api/uploads/${req.file.filename}`,
      message: 'Trajectory generation complete. Connect to Python backend for actual processing.',
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
    const heatmapDir = join(__dirname, '../data/outputs/heatmaps');
    const trajectoryDir = join(__dirname, '../data/outputs/trajectories');
    fs.mkdirSync(heatmapDir, { recursive: true });
    fs.mkdirSync(trajectoryDir, { recursive: true });

    const resultId = uuidv4();
    const heatmapPath = join(heatmapDir, `${resultId}_heatmap.png`);
    const trajectoryPath = join(trajectoryDir, `${resultId}_trajectory.png`);

    // Placeholder: Copy the uploaded file as a demo
    fs.copyFileSync(req.file.path, heatmapPath);
    fs.copyFileSync(req.file.path, trajectoryPath);

    res.json({
      success: true,
      id: resultId,
      heatmapUrl: `/api/outputs/heatmaps/${resultId}_heatmap.png`,
      trajectoryUrl: `/api/outputs/trajectories/${resultId}_trajectory.png`,
      originalUrl: `/api/uploads/${req.file.filename}`,
      message: 'Analysis complete. Connect to Python backend for actual processing.',
    });
  } catch (error) {
    console.error('Analysis error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Get prediction from gaze history
app.post('/api/predict', async (req, res) => {
  try {
    const { gazeHistory } = req.body;

    // Placeholder response
    res.json({
      success: true,
      prediction: {
        id: 'TH-B03',
        name: 'Sample Exhibit',
        confidence: 0.85,
        reasoning: 'This is a placeholder. Connect to Python backend for actual predictions.',
      },
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

// Start server
app.listen(PORT, () => {
  console.log(`\n🚀 IROS Gaze Server running on port ${PORT}`);
  console.log(`📡 API endpoint: http://localhost:${PORT}/api`);
  console.log(`📁 Uploads directory: ${join(__dirname, '../data/uploads')}`);
  console.log(`📁 Outputs directory: ${join(__dirname, '../data/outputs')}\n`);
});
