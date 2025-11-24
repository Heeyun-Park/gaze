#!/usr/bin/env python3
"""
Simple usage example for DualPoseGazeNet.

This script demonstrates how to:
1. Create and initialize the model
2. Prepare input data
3. Run inference
4. Interpret the results
"""

import torch
from model import DualPoseGazeNet
import math


def example_1_basic_usage():
    """Example 1: Basic model usage with dummy data."""
    print("\n" + "="*60)
    print("Example 1: Basic Model Usage")
    print("="*60 + "\n")

    # Step 1: Create model
    print("Step 1: Creating DualPoseGazeNet...")
    model = DualPoseGazeNet(backbone="resnet18", pretrained=False)
    model.eval()
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Step 2: Prepare dummy input data
    print("\nStep 2: Preparing input data...")
    batch_size = 4

    # Image sequences (7 frames, 224x224 RGB)
    head_seq = torch.randn(batch_size, 7, 3, 224, 224)
    body_seq = torch.randn(batch_size, 7, 3, 224, 224)

    # Confidence scores (both head and body visible)
    conf = torch.ones(batch_size, 2)  # [head_conf, body_conf]

    # Skeleton keypoints
    head_vec = torch.randn(batch_size, 15)   # 5 head keypoints × 3
    upper_vec = torch.randn(batch_size, 24)  # 8 body keypoints × 3

    print(f"✓ Input shapes:")
    print(f"  - head_seq: {head_seq.shape}")
    print(f"  - body_seq: {body_seq.shape}")
    print(f"  - conf: {conf.shape}")
    print(f"  - head_vec: {head_vec.shape}")
    print(f"  - upper_vec: {upper_vec.shape}")

    # Step 3: Run inference
    print("\nStep 3: Running inference...")
    with torch.no_grad():
        gaze_angle, variance = model(head_seq, body_seq, conf, head_vec, upper_vec)

    print(f"✓ Inference complete!")
    print(f"  - gaze_angle shape: {gaze_angle.shape}")
    print(f"  - variance shape: {variance.shape}")

    # Step 4: Interpret results
    print("\nStep 4: Interpreting results...")
    for i in range(batch_size):
        azimuth = gaze_angle[i, 0].item()
        elevation = gaze_angle[i, 1].item()
        az_deg = azimuth * 180 / math.pi
        el_deg = elevation * 180 / math.pi

        print(f"\nSample {i+1}:")
        print(f"  Azimuth: {az_deg:6.1f}° ({azimuth:6.3f} rad)")
        print(f"  Elevation: {el_deg:6.1f}° ({elevation:6.3f} rad)")
        print(f"  Uncertainty: [{variance[i, 0]:.3f}, {variance[i, 1]:.3f}]")


def example_2_occlusion_scenarios():
    """Example 2: Different occlusion scenarios."""
    print("\n" + "="*60)
    print("Example 2: Handling Different Occlusion Scenarios")
    print("="*60 + "\n")

    model = DualPoseGazeNet(backbone="resnet18", pretrained=False)
    model.eval()

    # Prepare dummy data
    head_seq = torch.randn(5, 7, 3, 224, 224)
    body_seq = torch.randn(5, 7, 3, 224, 224)
    head_vec = torch.randn(5, 15)
    upper_vec = torch.randn(5, 24)

    # Different confidence configurations
    scenarios = [
        ("Both head and body visible", torch.tensor([[1.0, 1.0]])),
        ("Head occluded, body visible", torch.tensor([[0.1, 1.0]])),
        ("Body occluded, head visible", torch.tensor([[1.0, 0.1]])),
        ("Head dominant (70/30)", torch.tensor([[0.7, 0.3]])),
        ("Body dominant (30/70)", torch.tensor([[0.3, 0.7]])),
    ]

    print("Running inference with different occlusion scenarios:\n")

    with torch.no_grad():
        for name, conf in scenarios:
            # Expand conf to match batch size
            conf_batch = conf.expand(5, 2)

            # Run inference
            gaze, var = model(head_seq, body_seq, conf_batch, head_vec, upper_vec)

            # Print results for first sample
            az_deg = gaze[0, 0].item() * 180 / math.pi
            el_deg = gaze[0, 1].item() * 180 / math.pi

            print(f"{name}:")
            print(f"  Confidence: head={conf[0,0]:.1f}, body={conf[0,1]:.1f}")
            print(f"  Predicted gaze: azimuth={az_deg:6.1f}°, elevation={el_deg:6.1f}°")
            print(f"  Uncertainty: [{var[0,0]:.3f}, {var[0,1]:.3f}]")
            print()


def example_3_batch_processing():
    """Example 3: Efficient batch processing."""
    print("\n" + "="*60)
    print("Example 3: Efficient Batch Processing")
    print("="*60 + "\n")

    model = DualPoseGazeNet(backbone="resnet18", pretrained=False)
    model.eval()

    # Simulate processing multiple samples
    num_batches = 5
    batch_size = 8
    total_samples = num_batches * batch_size

    print(f"Processing {total_samples} samples in {num_batches} batches...\n")

    all_predictions = []

    with torch.no_grad():
        for batch_idx in range(num_batches):
            # Create dummy batch
            head_seq = torch.randn(batch_size, 7, 3, 224, 224)
            body_seq = torch.randn(batch_size, 7, 3, 224, 224)
            conf = torch.rand(batch_size, 2)  # Random confidence
            head_vec = torch.randn(batch_size, 15)
            upper_vec = torch.randn(batch_size, 24)

            # Run inference
            gaze, var = model(head_seq, body_seq, conf, head_vec, upper_vec)

            # Store predictions
            all_predictions.append(gaze.cpu())

            print(f"Batch {batch_idx+1}/{num_batches}: Processed {batch_size} samples")

    # Concatenate all predictions
    all_predictions = torch.cat(all_predictions, dim=0)
    print(f"\n✓ Total predictions: {all_predictions.shape}")
    print(f"  Mean azimuth: {all_predictions[:, 0].mean():.3f} rad")
    print(f"  Mean elevation: {all_predictions[:, 1].mean():.3f} rad")


def example_4_gpu_usage():
    """Example 4: Using GPU if available."""
    print("\n" + "="*60)
    print("Example 4: GPU Usage")
    print("="*60 + "\n")

    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create model and move to device
    model = DualPoseGazeNet(backbone="resnet18", pretrained=False)
    model = model.to(device)
    model.eval()

    # Create dummy data and move to device
    batch_size = 16
    head_seq = torch.randn(batch_size, 7, 3, 224, 224).to(device)
    body_seq = torch.randn(batch_size, 7, 3, 224, 224).to(device)
    conf = torch.ones(batch_size, 2).to(device)
    head_vec = torch.randn(batch_size, 15).to(device)
    upper_vec = torch.randn(batch_size, 24).to(device)

    print(f"\nRunning inference on {device}...")

    with torch.no_grad():
        gaze, var = model(head_seq, body_seq, conf, head_vec, upper_vec)

    print(f"✓ Inference complete!")
    print(f"  Output device: {gaze.device}")
    print(f"  Output shape: {gaze.shape}")


def example_5_save_load_model():
    """Example 5: Saving and loading model checkpoints."""
    print("\n" + "="*60)
    print("Example 5: Saving and Loading Model")
    print("="*60 + "\n")

    # Create and save model
    print("Step 1: Creating and saving model...")
    model = DualPoseGazeNet(backbone="resnet18", pretrained=False)

    checkpoint = {
        "state_dict": model.state_dict(),
        "backbone": "resnet18",
        "pretrained": False,
    }

    save_path = "dualpose_checkpoint.pth"
    torch.save(checkpoint, save_path)
    print(f"✓ Model saved to: {save_path}")

    # Load model
    print("\nStep 2: Loading model from checkpoint...")
    checkpoint = torch.load(save_path, map_location="cpu")

    model_loaded = DualPoseGazeNet(
        backbone=checkpoint["backbone"],
        pretrained=checkpoint["pretrained"]
    )
    model_loaded.load_state_dict(checkpoint["state_dict"])
    model_loaded.eval()

    print(f"✓ Model loaded successfully!")

    # Verify it works
    print("\nStep 3: Verifying loaded model...")
    with torch.no_grad():
        head_seq = torch.randn(2, 7, 3, 224, 224)
        body_seq = torch.randn(2, 7, 3, 224, 224)
        conf = torch.ones(2, 2)
        head_vec = torch.randn(2, 15)
        upper_vec = torch.randn(2, 24)

        gaze, var = model_loaded(head_seq, body_seq, conf, head_vec, upper_vec)

    print(f"✓ Loaded model works correctly!")
    print(f"  Output shape: {gaze.shape}")

    # Clean up
    import os
    os.remove(save_path)
    print(f"\n✓ Checkpoint file removed")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("DualPoseGazeNet Usage Examples")
    print("="*60)

    examples = [
        ("Basic Usage", example_1_basic_usage),
        ("Occlusion Scenarios", example_2_occlusion_scenarios),
        ("Batch Processing", example_3_batch_processing),
        ("GPU Usage", example_4_gpu_usage),
        ("Save/Load Model", example_5_save_load_model),
    ]

    for i, (name, func) in enumerate(examples, 1):
        try:
            func()
        except Exception as e:
            print(f"\n✗ Example {i} ({name}) failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
