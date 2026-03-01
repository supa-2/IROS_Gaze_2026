# IROS_Gaze_2026: Spatial Intent Prediction System for Museum Gaze Behavior

A system for predicting visitor gaze behavior and spatial intent in museum environments using Large Language Models (LLMs) and Vision-Language Models (VLMs). This project is submitted to IROS 2026.

## Overview

This system aims to predict:
1. **Next Exhibit Prediction**: Which exhibit a visitor will view next based on their gaze trajectory
2. **Attention Level Estimation**: How long a visitor will spend viewing an exhibit (A=120s, B=60s, C=30s, D=15s, E=5s)
3. **Scan Path Planning**: Optimal viewing sequence for a set of exhibits
4. **Gaze Attribution**: Which exhibit a visitor is looking at from gaze coordinates

## Key Features

- **Multi-Task Learning**: Trained on four tasks (predict_next, attention_level, plan_scan_path, attribution)
- **ShareGPT Format**: Uses ShareGPT conversation format for fine-tuning
- **Candidate Selection**: Topology-aware candidate exhibit filtering
- **Ablation Studies**: Comprehensive ablation experiments to validate each component
- **Baseline Comparison**: System-level comparison with closed-source LLMs (GPT-4o, Claude, Gemini)

## Project Structure

```
IROS_Gaze_2026/
├── data/
│   ├── OS/                      # OS map data
│   ├── TH/                      # TH map data
│   ├── processed/               # Processed data
│   │   └── sharegpt_json/       # ShareGPT format training data
│   │       ├── train_sharegpt.jsonl
│   │       └── val_sharegpt.jsonl
│   └── outputs/                 # Experiment outputs
│       ├── vllm_ablation/       # Ablation study results
│       ├── test_predictions/    # Test set predictions
│       └── baselines/           # Baseline comparison results
├── scripts/
│   ├── experiments/             # Experiment scripts
│   │   ├── baseline_comparison.py   # System-level baseline comparison
│   │   └── ablation_study.py        # Ablation experiments
│   ├── inference_test.py        # Test set inference
│   ├── clean_sharegpt_dataset.py
│   └── ...
├── skills/                      # Core modules
│   ├── prediction/              # Prediction engine
│   ├── topology/                # Spatial topology
│   ├── memory/                  # Memory systems
│   ├── segmentation/            # SAM2 segmentation
│   ├── visualization/           # Visualization tools
│   └── experiments/             # Experiment utilities
├── agent.py                     # Main agent
├── config.py                    # Configuration
└── requirements.txt             # Dependencies
```

## Installation

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (for model inference)

### Setup

```bash
# Clone the repository
git clone https://github.com/supa-2/IROS_Gaze_2026.git
cd IROS_Gaze_2026

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Copy `.env.example` to `.env` and configure your API keys:

```bash
cp .env.example .env
```

Required environment variables:
- `OPENAI_API_KEY`: For GPT models
- `ANTHROPIC_API_KEY`: For Claude models
- `GOOGLE_API_KEY`: For Gemini models

## Data Format

### ShareGPT Format

The training data uses ShareGPT conversation format with JSON-encoded task data:

```json
{
  "conversations": [
    {
      "from": "human",
      "value": "```json\n{\"task\": \"predict_next\", \"exhibits\": [...]}\n```"
    },
    {
      "from": "gpt",
      "value": "```json\n{\"prediction\": {\"name\": \"...\", \"attention_level\": \"A\"}}\n```"
    }
  ]
}
```

### Task Types

1. **predict_next**: Predict the next exhibit and attention level
2. **plan_scan_path**: Plan optimal viewing sequence
3. **attribution**: Attribute gaze coordinates to exhibits

### Attention Levels

| Level | Duration | Description |
|-------|----------|-------------|
| A     | 120s     | Deep attention - long careful viewing |
| B     | 60s      | Medium attention - normal viewing |
| C     | 30s      | Normal attention - browsing |
| D     | 15s      | Quick glance - brief pause |
| E     | 5s       | Fleeting glance - quick scan |

## Usage

### Test Set Inference

```bash
python scripts/inference_test.py \
    --data data/processed/sharegpt_json/val_sharegpt.jsonl \
    --api-url http://localhost:8000/v1 \
    --model-name Qwen \
    --num-samples 5 \
    --output data/outputs/test_predictions/predictions_test.csv
```

### Ablation Study

```bash
python scripts/experiments/ablation_study.py \
    --test-data data/processed/sharegpt_json/val_sharegpt.jsonl \
    --output data/outputs/vllm_ablation/
```

### Baseline Comparison

```bash
python skills/experiments/baseline_comparison.py \
    --data data/processed/sharegpt_json/val_sharegpt.jsonl \
    --ablation data/outputs/vllm_ablation/ablation_results.json \
    --zero-shot GPT-4o Claude-Sonnet-4 Gemini-2.5-Pro
```

## Experiments

### Ablation Study

The ablation study evaluates the contribution of each component:

| Configuration | Description |
|---------------|-------------|
| Full          | Complete system with all components |
| No History    | Without visit history |
| No Topology   | Without topology constraint |
| No Reasoning  | Without reasoning explanation |
| Single Sample | Single prediction sampling |

### Baseline Comparison

System-level comparison with:
- **Statistical**: Markov Chain
- **Deep Learning**: LSTM/MLP
- **Zero-Shot LLMs**: GPT-4o, Claude-Sonnet-4, Gemini-2.5-Pro
- **Ours**: Fine-tuned model with full system

## Evaluation Metrics

- **Top-K Accuracy**: Correct prediction in top-K candidates
- **KL Divergence**: Distribution matching (lower is better)
- **JS Divergence**: Symmetric distribution distance (lower is better)
- **Correlation**: Prediction-Ground truth correlation (higher is better)

## Citation

If you use this code for your research, please cite:

```bibtex
@inproceedings{iros2026_gaze,
  title={Spatial Intent Prediction for Museum Gaze Behavior using Large Language Models},
  author={Your Name},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- SAM2 for image segmentation
- vLLM for efficient model inference
- ShareGPT for data format
