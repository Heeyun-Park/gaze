#!/usr/bin/env python3
# ============================================================
# train_dualpose.py - Training Script for DualPoseGazeNet
#
# Occlusion-robust gaze estimation using dual-stream architecture
# combining head and upper-body information with skeleton keypoints.
# ============================================================

import os
import math
import argparse
import torch
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import json

from model import DualPoseGazeNet


# ============================================================
# Loss Function: PinBall Loss (Quantile Regression)
# ============================================================
class PinBallLoss(torch.nn.Module):
    """
    PinBall Loss for uncertainty-aware gaze estimation.
    Combines standard angular error with variance prediction.
    """
    def __init__(self, quantile=0.1):
        super().__init__()
        self.quantile = quantile

    def forward(self, pred, target, var):
        """
        Args:
            pred: (B, 2) - Predicted angles
            target: (B, 2) - Ground truth angles
            var: (B, 2) - Predicted variance
        """
        # Angular error
        error = pred - target

        # PinBall loss: penalize underestimation more than overestimation
        loss = torch.where(
            error >= 0,
            self.quantile * error,
            (self.quantile - 1) * error
        )

        # Weight by inverse variance (higher variance → lower weight)
        weighted_loss = loss / (var + 1e-6)

        # Regularization: prevent variance from becoming too large
        var_reg = torch.log(var + 1e-6)

        total_loss = weighted_loss.abs().mean() + 0.5 * var_reg.mean()

        return total_loss


# ============================================================
# Metrics: Angular Error
# ============================================================
def compute_angular_error(pred, target):
    """
    Compute mean angular error in degrees.

    Args:
        pred: (B, 2) - Predicted angles in radians [azimuth, elevation]
        target: (B, 2) - Ground truth angles in radians

    Returns:
        Mean angular error in degrees
    """
    # Convert to unit vectors
    pred_az, pred_el = pred[:, 0], pred[:, 1]
    target_az, target_el = target[:, 0], target[:, 1]

    # Spherical to Cartesian
    pred_x = torch.cos(pred_el) * torch.cos(pred_az)
    pred_y = torch.cos(pred_el) * torch.sin(pred_az)
    pred_z = torch.sin(pred_el)
    pred_vec = torch.stack([pred_x, pred_y, pred_z], dim=1)

    target_x = torch.cos(target_el) * torch.cos(target_az)
    target_y = torch.cos(target_el) * torch.sin(target_az)
    target_z = torch.sin(target_el)
    target_vec = torch.stack([target_x, target_y, target_z], dim=1)

    # Cosine similarity
    cos_sim = (pred_vec * target_vec).sum(dim=1).clamp(-1, 1)

    # Angular error in radians → degrees
    angular_error = torch.acos(cos_sim) * 180.0 / math.pi

    return angular_error.mean().item()


# ============================================================
# Average Meter (for tracking metrics)
# ============================================================
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ============================================================
# Dummy Data Loader (for testing)
# ============================================================
class DummyGazeDataset(torch.utils.data.Dataset):
    """
    Dummy dataset for testing. Replace with your actual dataset.
    """
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random data (replace with actual data loading)
        head_seq = torch.randn(7, 3, 224, 224)
        body_seq = torch.randn(7, 3, 224, 224)
        conf = torch.rand(2)  # Random confidence
        head_vec = torch.randn(15)
        upper_vec = torch.randn(24)

        # Random gaze target
        gaze = torch.rand(2) * 2 - 1  # Random angles
        gaze[0] *= math.pi  # azimuth: [-π, π]
        gaze[1] *= math.pi / 2  # elevation: [-π/2, π/2]

        return head_seq, body_seq, conf, gaze, head_vec, upper_vec


# ============================================================
# Training Functions
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    loss_meter = AverageMeter()
    err_meter = AverageMeter()

    pbar = tqdm(loader, desc="[Train]", ncols=100)

    for head, body, conf, target, head_vec, upper_vec in pbar:
        head = head.to(device)
        body = body.to(device)
        conf = conf.to(device)
        target = target.to(device)
        head_vec = head_vec.to(device)
        upper_vec = upper_vec.to(device)

        optimizer.zero_grad()

        # Forward pass
        pred, var = model(head, body, conf, head_vec, upper_vec)

        # Compute loss
        loss = criterion(pred, target, var)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Compute metrics
        err = compute_angular_error(pred, target)

        loss_meter.update(loss.item())
        err_meter.update(err)

        pbar.set_postfix({
            "loss": f"{loss_meter.avg:.4f}",
            "err": f"{err_meter.avg:.2f}°"
        })

    return loss_meter.avg, err_meter.avg


def validate(model, loader, device):
    """Validate the model."""
    model.eval()
    meter = AverageMeter()

    pbar = tqdm(loader, desc="[Val]", ncols=100)

    with torch.no_grad():
        for head, body, conf, target, head_vec, upper_vec in pbar:
            head = head.to(device)
            body = body.to(device)
            conf = conf.to(device)
            target = target.to(device)
            head_vec = head_vec.to(device)
            upper_vec = upper_vec.to(device)

            # Forward pass
            pred, _ = model(head, body, conf, head_vec, upper_vec)

            # Compute metrics
            err = compute_angular_error(pred, target)
            meter.update(err)

            pbar.set_postfix({"err": f"{meter.avg:.2f}°"})

    return meter.avg


def evaluate(model, loader, device, save_path=None):
    """Evaluate the model and optionally save predictions."""
    model.eval()
    errors = []

    pbar = tqdm(loader, desc="[Test]", ncols=100)

    with torch.no_grad():
        for head, body, conf, target, head_vec, upper_vec in pbar:
            head = head.to(device)
            body = body.to(device)
            conf = conf.to(device)
            target = target.to(device)
            head_vec = head_vec.to(device)
            upper_vec = upper_vec.to(device)

            # Forward pass
            pred, _ = model(head, body, conf, head_vec, upper_vec)

            # Compute metrics
            err = compute_angular_error(pred, target)
            errors.append(err)

            pbar.set_postfix({"err": f"{np.mean(errors):.2f}°"})

    mean_err = float(np.mean(errors))
    print(f"[Test] Mean angular error = {mean_err:.3f}° ({len(errors)} samples)")

    if save_path:
        import csv
        with open(save_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_idx", "angular_error"])
            for i, e in enumerate(errors):
                writer.writerow([i, e])
        print(f"[Saved] {save_path}")

    return mean_err


# ============================================================
# Main Training Loop
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Train DualPoseGazeNet for occlusion-robust gaze estimation"
    )

    # Data arguments
    parser.add_argument("--data_path", type=str, default="./data",
                        help="Path to dataset")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")

    # Model arguments
    parser.add_argument("--backbone", type=str, default="resnet18",
                        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
                        help="Backbone architecture")
    parser.add_argument("--pretrained", action="store_true", default=True,
                        help="Use pretrained backbone")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay")

    # Checkpoint arguments
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--eval_only", action="store_true",
                        help="Only evaluate (no training)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint for evaluation")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Directory for saving outputs")
    parser.add_argument("--save_csv", action="store_true",
                        help="Save prediction results to CSV")

    args = parser.parse_args()

    # --------------------------------------------------------
    # Setup experiment directory
    # --------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"DualPoseGazeNet_{args.backbone}_{timestamp}"
    exp_dir = os.path.join(args.output_dir, exp_name)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    pred_dir = os.path.join(exp_dir, "predictions")

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    print(f"[Experiment] {exp_dir}")

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # --------------------------------------------------------
    # Data transforms
    # --------------------------------------------------------
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    transform_eval = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    # --------------------------------------------------------
    # Datasets (using dummy data for now)
    # TODO: Replace with your actual dataset
    # --------------------------------------------------------
    print("[Info] Using dummy dataset for testing")
    print("[TODO] Replace DummyGazeDataset with your actual dataset")

    train_dataset = DummyGazeDataset(num_samples=1000)
    val_dataset = DummyGazeDataset(num_samples=200)
    test_dataset = DummyGazeDataset(num_samples=200)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"[Data] Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    model = DualPoseGazeNet(backbone=args.backbone, pretrained=args.pretrained)
    model = torch.nn.DataParallel(model).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] DualPoseGazeNet with {num_params:,} trainable parameters")

    # --------------------------------------------------------
    # Loss & Optimizer
    # --------------------------------------------------------
    criterion = PinBallLoss().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )

    # --------------------------------------------------------
    # Resume from checkpoint
    # --------------------------------------------------------
    start_epoch = 0
    best_val_err = float("inf")

    if args.resume:
        print(f"[Resume] Loading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)

        # Load model state
        state_dict = ckpt.get("state_dict", ckpt)
        if list(state_dict.keys())[0].startswith("module."):
            # Remove 'module.' prefix if present
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.module.load_state_dict(state_dict)

        # Load optimizer state
        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except:
                print("[Resume] Warning: Could not load optimizer state")

        start_epoch = ckpt.get("epoch", 0)
        best_val_err = ckpt.get("best_val_err", float("inf"))

        print(f"[Resume] Epoch={start_epoch}, Best Val Error={best_val_err:.3f}°")

    # --------------------------------------------------------
    # Evaluation-only mode
    # --------------------------------------------------------
    if args.eval_only:
        if args.checkpoint is None:
            raise ValueError("--checkpoint required for --eval_only")

        print(f"[Eval] Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt.get("state_dict", ckpt)
        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.module.load_state_dict(state_dict)

        save_path = os.path.join(pred_dir, "test_results.csv") if args.save_csv else None
        test_err = evaluate(model, test_loader, device, save_path)
        print(f"[Eval] Test Error = {test_err:.3f}°")
        return

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Starting training for {args.epochs} epochs")
    print(f"{'='*60}\n")

    best_ckpt_path = os.path.join(ckpt_dir, "best_model.pth")

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch [{epoch+1}/{args.epochs}]")
        print("-" * 60)

        # Train
        train_loss, train_err = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_err = validate(model, val_loader, device)

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Print summary
        print(f"Train Loss: {train_loss:.4f}, Train Error: {train_err:.2f}°")
        print(f"Val Error: {val_err:.2f}°, LR: {current_lr:.6f}")

        # Save best model
        if val_err < best_val_err:
            best_val_err = val_err
            torch.save({
                "epoch": epoch + 1,
                "state_dict": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_err": best_val_err,
                "args": vars(args)
            }, best_ckpt_path)
            print(f"✓ Best model saved (val_err={best_val_err:.3f}°)")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch+1}.pth")
            torch.save({
                "epoch": epoch + 1,
                "state_dict": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_err": val_err,
                "args": vars(args)
            }, ckpt_path)

    # --------------------------------------------------------
    # Final evaluation on test set
    # --------------------------------------------------------
    print(f"\n{'='*60}")
    print("Training completed. Evaluating on test set...")
    print(f"{'='*60}\n")

    # Load best model
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.module.load_state_dict(ckpt["state_dict"])

    save_path = os.path.join(pred_dir, "test_results.csv") if args.save_csv else None
    test_err = evaluate(model, test_loader, device, save_path)

    # --------------------------------------------------------
    # Save training log
    # --------------------------------------------------------
    log = {
        "model": "DualPoseGazeNet",
        "backbone": args.backbone,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        "best_val_err": round(best_val_err, 3),
        "test_err": round(test_err, 3),
        "num_parameters": num_params,
        "timestamp": timestamp
    }

    log_path = os.path.join(exp_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=4)
    print(f"\n[Saved] Training log: {log_path}")

    print(f"\n{'='*60}")
    print(f"Final Results:")
    print(f"  Best Val Error: {best_val_err:.3f}°")
    print(f"  Test Error: {test_err:.3f}°")
    print(f"  Checkpoint: {best_ckpt_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
