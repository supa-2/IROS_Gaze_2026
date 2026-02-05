# Predictions Output Directory

This directory stores prediction results exported by the Eye-LLM system.

## File Format

Prediction files are saved in JSON format with the following structure:

```json
{
  "timestamp": "2025-02-05T12:34:56",
  "map_name": "TH",
  "memory_statistics": {
    "short_term_count": 3,
    "long_term_count": 15,
    "unique_exhibits": 8
  },
  "predictions": [
    {
      "step_number": 1,
      "prediction_id": "TH-B02",
      "prediction_name": "何尊",
      "attention_level": "A",
      "estimated_duration": 120,
      "confidence": 0.85,
      "reasoning": "..."
    }
  ]
}
```

## Usage

Predictions are automatically exported when calling:
```python
engine.export_predictions(sequence, output_path="data/outputs/predictions/prediction_20250205.json")
```
