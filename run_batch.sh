#!/bin/bash
# Batch processing for images 1-5

source .venv/bin/activate

for i in {1..5}; do
    echo "========================================"
    echo "Processing data/${i}.jpg..."
    echo "========================================"
    python scripts/unified_gaze_viz.py --image data/${i}.jpg --use-sam2
    
    if [ $? -eq 0 ]; then
        echo "[SUCCESS] data/${i}.jpg completed"
    else
        echo "[FAILED] data/${i}.jpg failed"
    fi
    echo ""
done

echo "========================================"
echo "All done! Results in data/outputs/"
echo "========================================"
