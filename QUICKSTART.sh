#!/bin/bash
# ============================================================
# Quick Start Guide for DualPoseGazeNet Training
# ============================================================

# This script shows various ways to run the DualPoseGazeNet model

# ============================================================
# Option 1: Test with Dummy Data (No real dataset needed)
# ============================================================
echo "=== Option 1: Testing with Dummy Data ==="
echo "This will create synthetic data and verify the model works"
echo ""
echo "Command:"
echo "  python train_dualpose.py --epochs 5 --batch_size 8"
echo ""

# ============================================================
# Option 2: Training with Real Dataset
# ============================================================
echo "=== Option 2: Training with Real Dataset ==="
echo "If you have a dataset with the following structure:"
echo ""
echo "  data_occluded_0/"
echo "    ├── train_occlusion_0.txt"
echo "    ├── val_occlusion_0.txt"
echo "    ├── test_occlusion_0.txt"
echo "    └── images/"
echo ""
echo "Command:"
echo "  python train_dualpose.py \\"
echo "    --data_path ./data_occluded_0 \\"
echo "    --batch_size 32 \\"
echo "    --epochs 50 \\"
echo "    --lr 1e-4 \\"
echo "    --backbone resnet18 \\"
echo "    --save_csv"
echo ""

# ============================================================
# Option 3: Using SwitchGazeNet (Original Model)
# ============================================================
echo "=== Option 3: Using SwitchGazeNet ==="
echo "If you want to use the original SwitchGazeNet model:"
echo ""
echo "Command:"
echo "  python run.py \\"
echo "    --data_path ./data_occluded_0 \\"
echo "    --train_list train_occlusion_0.txt \\"
echo "    --val_list val_occlusion_0.txt \\"
echo "    --test_list test_occlusion_0.txt \\"
echo "    --batch_size 64 \\"
echo "    --epochs 40 \\"
echo "    --lr 1e-4 \\"
echo "    --backbone resnet18"
echo ""

# ============================================================
# Option 4: Resume Training from Checkpoint
# ============================================================
echo "=== Option 4: Resume Training ==="
echo "To continue training from a saved checkpoint:"
echo ""
echo "Command:"
echo "  python train_dualpose.py \\"
echo "    --resume ./outputs/DualPoseGazeNet_xxx/checkpoints/best_model.pth \\"
echo "    --epochs 100"
echo ""

# ============================================================
# Option 5: Evaluation Only
# ============================================================
echo "=== Option 5: Evaluation Only ==="
echo "To evaluate a trained model without training:"
echo ""
echo "Command:"
echo "  python train_dualpose.py \\"
echo "    --eval_only \\"
echo "    --checkpoint ./outputs/DualPoseGazeNet_xxx/checkpoints/best_model.pth \\"
echo "    --save_csv"
echo ""

# ============================================================
# Data Preparation Guide
# ============================================================
echo "=== Data Preparation ==="
echo ""
echo "Your dataset should provide:"
echo "  1. head_seq: (7, 3, 224, 224) - 7-frame head crop sequence"
echo "  2. body_seq: (7, 3, 224, 224) - 7-frame body crop sequence"
echo "  3. conf: (2,) - [head_confidence, body_confidence]"
echo "  4. head_vec: (15,) - 5 head keypoints × (x, y, confidence)"
echo "  5. upper_vec: (24,) - 8 body keypoints × (x, y, confidence)"
echo "  6. gaze: (2,) - ground truth (azimuth, elevation)"
echo ""
echo "Currently, train_dualpose.py uses DummyGazeDataset for testing."
echo "Replace it with your actual dataset loader."
echo ""

# ============================================================
# Quick Test Command
# ============================================================
echo "=== Quick Test (Recommended First Step) ==="
echo ""
echo "To verify everything works, run:"
echo "  python test_model.py"
echo ""
echo "This will test all model components without needing a dataset."
echo ""
