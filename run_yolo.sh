#!/bin/bash
# Easy wrapper for YOLOv8 processing

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if model exists
if [ ! -f "best.pt" ]; then
    echo "ERROR: YOLOv8 model 'best.pt' not found!"
    echo ""
    echo "Please download your trained model from Roboflow:"
    echo "  1. Go to your Roboflow project"
    echo "  2. Click 'Deploy' → 'YOLOv8'"
    echo "  3. Download 'best.pt'"
    echo "  4. Copy it to: $SCRIPT_DIR/best.pt"
    echo ""
    echo "See YOLO_SETUP.md for detailed instructions"
    exit 1
fi

# Activate virtual environment and run
source venv/bin/activate

# Run with optional folder argument
if [ $# -eq 0 ]; then
    python process_bills_yolo.py best.pt
else
    python process_bills_yolo.py best.pt "$1"
fi
