#!/bin/bash
# ============================================================
# Example: Training DualPoseGazeNet with Dummy Data
# ============================================================
# This script demonstrates how to run DualPoseGazeNet
# with the built-in dummy dataset for testing purposes.

set -e  # Exit on error

echo "========================================"
echo "DualPoseGazeNet Training Example"
echo "========================================"
echo ""

# Check if we're in the right directory
if [ ! -f "model.py" ]; then
    echo "Error: model.py not found!"
    echo "Please run this script from the gaze/ directory"
    exit 1
fi

# Run a quick test first
echo "Step 1: Running model tests..."
echo "----------------------------------------"
python test_model.py

if [ $? -ne 0 ]; then
    echo "Error: Model tests failed!"
    echo "Please check your PyTorch installation"
    exit 1
fi

echo ""
echo "Step 2: Training DualPoseGazeNet with dummy data..."
echo "----------------------------------------"
echo "Configuration:"
echo "  - Backbone: resnet18 (no pretrained weights for speed)"
echo "  - Epochs: 3 (quick test)"
echo "  - Batch size: 4 (small for testing)"
echo "  - Dataset: Dummy data (1000 train, 200 val, 200 test)"
echo ""

python train_dualpose.py \
    --backbone resnet18 \
    --epochs 3 \
    --batch_size 4 \
    --num_workers 2 \
    --lr 1e-4

echo ""
echo "========================================"
echo "Training completed successfully!"
echo "========================================"
echo ""
echo "Check outputs/ directory for:"
echo "  - Checkpoints (.pth files)"
echo "  - Training logs (training_log.json)"
echo "  - Test predictions (test_results.csv if --save_csv was used)"
echo ""
echo "To train with your own dataset, modify train_dualpose.py"
echo "and replace DummyGazeDataset with your data loader."
echo ""
