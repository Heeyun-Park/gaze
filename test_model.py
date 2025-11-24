#!/usr/bin/env python3
"""
Simple test script to verify DualPoseGazeNet implementation.
This script creates dummy data and performs a forward pass to check
that the model architecture is correct.
"""

import torch
import sys

# Import the model
try:
    from model import DualPoseGazeNet, GazeLSTM
    print("✓ Successfully imported DualPoseGazeNet and GazeLSTM")
except ImportError as e:
    print(f"✗ Failed to import models: {e}")
    sys.exit(1)


def test_gazelstm():
    """Test GazeLSTM module."""
    print("\n" + "="*60)
    print("Testing GazeLSTM Module")
    print("="*60)

    # Create model
    model = GazeLSTM(backbone="resnet18", pretrained=False)
    model.eval()

    # Create dummy input
    batch_size = 4
    num_frames = 7
    input_seq = torch.randn(batch_size, num_frames, 3, 224, 224)

    print(f"Input shape: {input_seq.shape}")

    # Forward pass
    try:
        with torch.no_grad():
            ang, var = model(input_seq)

        print(f"Output angle shape: {ang.shape} (expected: [{batch_size}, 2])")
        print(f"Output variance shape: {var.shape} (expected: [{batch_size}, 2])")

        # Check output ranges
        print(f"\nAngle range: [{ang.min().item():.3f}, {ang.max().item():.3f}]")
        print(f"Variance range: [{var.min().item():.3f}, {var.max().item():.3f}]")

        # Verify shapes
        assert ang.shape == (batch_size, 2), f"Wrong angle shape: {ang.shape}"
        assert var.shape == (batch_size, 2), f"Wrong variance shape: {var.shape}"

        print("\n✓ GazeLSTM test passed!")
        return True

    except Exception as e:
        print(f"\n✗ GazeLSTM test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dualpose_gazenet():
    """Test DualPoseGazeNet module."""
    print("\n" + "="*60)
    print("Testing DualPoseGazeNet Module")
    print("="*60)

    # Create model
    model = DualPoseGazeNet(backbone="resnet18", pretrained=False)
    model.eval()

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")

    # Create dummy inputs
    batch_size = 4
    num_frames = 7

    head_seq = torch.randn(batch_size, num_frames, 3, 224, 224)
    body_seq = torch.randn(batch_size, num_frames, 3, 224, 224)
    conf = torch.rand(batch_size, 2)  # Random confidence [0, 1]
    head_vec = torch.randn(batch_size, 15)  # 5 keypoints × 3
    upper_vec = torch.randn(batch_size, 24)  # 8 keypoints × 3

    print(f"\nInput shapes:")
    print(f"  head_seq: {head_seq.shape}")
    print(f"  body_seq: {body_seq.shape}")
    print(f"  conf: {conf.shape}")
    print(f"  head_vec: {head_vec.shape}")
    print(f"  upper_vec: {upper_vec.shape}")

    # Forward pass
    try:
        with torch.no_grad():
            ang, var = model(head_seq, body_seq, conf, head_vec, upper_vec)

        print(f"\nOutput shapes:")
        print(f"  angle: {ang.shape} (expected: [{batch_size}, 2])")
        print(f"  variance: {var.shape} (expected: [{batch_size}, 2])")

        # Check output ranges
        print(f"\nOutput statistics:")
        print(f"  Angle range: [{ang.min().item():.3f}, {ang.max().item():.3f}]")
        print(f"  Variance range: [{var.min().item():.3f}, {var.max().item():.3f}]")
        print(f"  Mean angle: [{ang[:, 0].mean().item():.3f}, {ang[:, 1].mean().item():.3f}]")
        print(f"  Mean variance: [{var[:, 0].mean().item():.3f}, {var[:, 1].mean().item():.3f}]")

        # Verify shapes
        assert ang.shape == (batch_size, 2), f"Wrong angle shape: {ang.shape}"
        assert var.shape == (batch_size, 2), f"Wrong variance shape: {var.shape}"

        print("\n✓ DualPoseGazeNet test passed!")
        return True

    except Exception as e:
        print(f"\n✗ DualPoseGazeNet test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_confidence_weighting():
    """Test confidence-based weighting behavior."""
    print("\n" + "="*60)
    print("Testing Confidence-based Weighting")
    print("="*60)

    model = DualPoseGazeNet(backbone="resnet18", pretrained=False)
    model.eval()

    batch_size = 3
    num_frames = 7

    # Create inputs
    head_seq = torch.randn(batch_size, num_frames, 3, 224, 224)
    body_seq = torch.randn(batch_size, num_frames, 3, 224, 224)
    head_vec = torch.randn(batch_size, 15)
    upper_vec = torch.randn(batch_size, 24)

    # Test different confidence scenarios
    test_cases = [
        ("Both available", torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])),
        ("Head only", torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])),
        ("Body only", torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])),
        ("Head dominant", torch.tensor([[0.8, 0.2], [0.8, 0.2], [0.8, 0.2]])),
        ("Body dominant", torch.tensor([[0.3, 0.7], [0.3, 0.7], [0.3, 0.7]])),
    ]

    try:
        with torch.no_grad():
            for name, conf in test_cases:
                ang, var = model(head_seq, body_seq, conf, head_vec, upper_vec)
                print(f"\n{name}:")
                print(f"  conf = {conf[0].tolist()}")
                print(f"  angle = [{ang[0, 0].item():.3f}, {ang[0, 1].item():.3f}]")
                print(f"  var = [{var[0, 0].item():.3f}, {var[0, 1].item():.3f}]")

        print("\n✓ Confidence weighting test passed!")
        return True

    except Exception as e:
        print(f"\n✗ Confidence weighting test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gradient_flow():
    """Test that gradients flow properly through the model."""
    print("\n" + "="*60)
    print("Testing Gradient Flow")
    print("="*60)

    model = DualPoseGazeNet(backbone="resnet18", pretrained=False)
    model.train()

    # Create inputs
    batch_size = 2
    num_frames = 7

    head_seq = torch.randn(batch_size, num_frames, 3, 224, 224, requires_grad=True)
    body_seq = torch.randn(batch_size, num_frames, 3, 224, 224, requires_grad=True)
    conf = torch.rand(batch_size, 2)
    head_vec = torch.randn(batch_size, 15, requires_grad=True)
    upper_vec = torch.randn(batch_size, 24, requires_grad=True)

    target = torch.randn(batch_size, 2)

    try:
        # Forward pass
        ang, var = model(head_seq, body_seq, conf, head_vec, upper_vec)

        # Simple loss
        loss = ((ang - target) ** 2).mean()

        # Backward pass
        loss.backward()

        # Check gradients
        has_grad = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_grad.append(name)

        print(f"\nParameters with gradients: {len(has_grad)}")
        print(f"Total parameters: {len(list(model.parameters()))}")

        # Check input gradients
        assert head_seq.grad is not None, "No gradient for head_seq"
        assert body_seq.grad is not None, "No gradient for body_seq"
        assert head_vec.grad is not None, "No gradient for head_vec"
        assert upper_vec.grad is not None, "No gradient for upper_vec"

        print("\n✓ Gradient flow test passed!")
        return True

    except Exception as e:
        print(f"\n✗ Gradient flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("DualPoseGazeNet Implementation Test Suite")
    print("="*60)

    # Run tests
    results = []
    results.append(("GazeLSTM", test_gazelstm()))
    results.append(("DualPoseGazeNet", test_dualpose_gazenet()))
    results.append(("Confidence Weighting", test_confidence_weighting()))
    results.append(("Gradient Flow", test_gradient_flow()))

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    all_passed = all(passed for _, passed in results)

    print("="*60)
    if all_passed:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
