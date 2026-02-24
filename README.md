# Eye-LLM: Spatial Intent Prediction System

> **IROS 2026 - Gaze Prediction System**

Eye-LLM is an advanced spatial intent prediction system that combines large language models (LLMs), computer vision, and topological reasoning to predict human gaze patterns and spatial intent in complex environments.

## 🌟 Key Features

- **Multi-Modal Reasoning**: Combines visual information with natural language understanding
- **Advanced Memory Management**: Short-term and long-term memory systems for context retention
- **Topological Graph Engine**: Spatial reasoning based on environmental topology
- **Real-Time Predictions**: Dynamic gaze trajectory and intent prediction
- **Comprehensive Visualization**: Heatmaps, trajectory plots, and network visualizations
- **Backward Compatible**: Fallback to original implementation for reliability

## 📦 Installation

### Prerequisites

- Python 3.11+
- uv (Python package manager)
- OpenAI API Key
- Replicate API Token (for SAM 2 segmentation)

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/supa-2/IROS_Gaze_2026.git
cd IROS_Gaze_2026
```

2. **Install dependencies**
```bash
uv sync
```

3. **Activate virtual environment**
```bash
source .venv/bin/activate
```

4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env file with your API keys
```

## 🚀 Usage

### Basic Usage

```python
from agent import EyeLLMAgent

# Initialize agent with new architecture
agent = EyeLLMAgent(map_name='TH', use_new_architecture=True)

# Add observations
agent.add_observation(exhibit_id='E1', attention_level='A')
agent.add_observation(exhibit_id='E3', attention_level='B')

# Predict next intent
prediction = agent.predict_next()
print(f"Next predicted exhibit: {prediction['next_exhibit']}")

# Visualize trajectory
agent.visualize_trajectory()
```

### Advanced Prediction

```python
# Predict sequence of next 5 steps
sequence = agent.predict_sequence(n_steps=5)
for i, step in enumerate(sequence):
    print(f"Step {i+1}: {step['exhibit_id']} (Confidence: {step['confidence']})")
```

### Pixel-Level Heatmap

```python
# Generate pixel-level heatmap on original image
heatmap_path = agent.visualize_pixel_heatmap(image_path='data/maps/th_hall.jpg')
print(f"Heatmap saved to: {heatmap_path}")
```

## 📁 Project Structure

```
iros_agent/
├── agent.py              # Main agent class
├── config.py             # Global configuration
├── dashboard.py          # Streamlit dashboard
├── main.py              # Entry point
├── pyproject.toml       # Dependencies
├── skills/
│   ├── memory/          # Memory management
│   ├── prediction/      # Prediction engine
│   ├── segmentation/    # Image segmentation
│   ├── topology/        # Topological graph engine
│   └── visualization/   # Visualization tools
├── tests/               # Test suite
└── scripts/             # Utility scripts
```

## 🧪 Testing

```bash
# Run all tests
uv run pytest

# Run specific test
uv run pytest tests/test_prediction.py

# Run with coverage
uv run coverage run -m pytest
uv run coverage report
```

## 📊 Configuration

### Environment Variables

| Variable               | Description                          | Default Value                  |
|------------------------|--------------------------------------|--------------------------------|
| OPENAI_API_KEY         | OpenAI API key                       | None                           |
| OPENAI_BASE_URL        | OpenAI API base URL                  | https://api.openai.com/v1      |
| REPLICATE_API_TOKEN    | Replicate API token                  | None                           |
| SHORT_TERM_SIZE        | Short-term memory size               | 5                              |
| LONG_TERM_MAX_SIZE     | Long-term memory maximum size        | 10000                          |

### Attention Levels

| Level | Duration | Description                          |
|-------|----------|--------------------------------------|
| A     | 120s     | Deep focus - Long, careful observation |
| B     | 60s      | Medium focus - Normal observation    |
| C     | 30s      | General focus - Browsing             |
| D     | 15s      | Quick glance - Brief stop            |
| E     | 5s       | Glimpse - Rapid scan                 |

## 🎨 Visualization Examples

### Trajectory Visualization
Shows historical gaze path through the environment with timestamps and attention levels.

### Heatmap Visualization
Displays spatial attention distribution across exhibits based on gaze data.

### Network Visualization
Visualizes topological graph with exhibit nodes and connection weights.

## 🔧 Development

### Adding New Features

1. Create new module in appropriate directory
2. Add tests in tests/ directory
3. Update documentation
4. Run tests and ensure coverage

### Contributing

1. Fork the repository
2. Create feature branch
3. Make changes and add tests
4. Submit pull request

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🤝 Acknowledgments

- OpenAI for GPT-4o model
- Meta for SAM 2 segmentation model
- LangChain for LLM integration
- IROS 2026 committee for research support

## 📧 Contact

For questions or collaboration, please contact:

- supa-2@github.com

---

**IROS 2026 - Gaze Prediction System** 👁️🔍