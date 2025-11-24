# ============================================================
# run_occlusion.py (SwitchGazeNet + Resume Support)
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

# --- add project root to sys.path ---
import os, sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# ------------------------------------

from core.data_loader import OcclusionLoader
from core.losses import PinBallLoss
from core.metrics import compute_angular_error
from core.utils import AverageMeter
from models.switchgaze.model import SwitchGazeNet


# ============================================================
# main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--train_list", type=str, required=True)
    parser.add_argument("--val_list", type=str, default=None)
    parser.add_argument("--test_list", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--backbone", type=str, default="resnet18",
                        choices=["resnet18", "resnet34", "resnet50", "resnet101"])
    parser.add_argument("--img_feature_dim", type=int, default=512,
                        help="Feature dimension for LSTM (default: 512)")  # ★ added

    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)

    parser.add_argument("--resume", type=str, default=None, help="path to checkpoint for continuing training")  # ★ added
    parser.add_argument("--save_csv", action="store_true")

    # ★ Training enhancements
    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="Number of warmup epochs (default: 5)")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping max norm (default: 1.0)")

    args = parser.parse_args()

    # --------------------------------------------------------
    # Experiment directory
    # --------------------------------------------------------
    occ_tag = args.train_list.replace("train_", "").replace(".txt", "")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_name = "SwitchGaze"
    exp_dir = os.path.join("outputs", f"{model_name}_{occ_tag}_{timestamp}")
    ckpt_dir = os.path.join(exp_dir, "checkpoint")
    pred_dir = os.path.join(exp_dir, "predictions")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------------
    # Transforms (★ Enhanced with Data Augmentation)
    # --------------------------------------------------------
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # ★
        transforms.RandomRotation(degrees=10),  # ★ Small rotation
        transforms.RandomHorizontalFlip(p=0.5),  # ★ Note: requires azimuth flip in dataset
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.1)),  # ★ Random occlusion
        normalize
    ])
    transform_eval = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------
    val_list = args.val_list or args.train_list.replace("train_", "val_")
    test_list = args.test_list or args.train_list.replace("train_", "test_")

    train_loader = DataLoader(
        OcclusionLoader(args.data_path, os.path.join(args.data_path, args.train_list), transform_train),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        OcclusionLoader(args.data_path, os.path.join(args.data_path, val_list), transform_eval),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    # --------------------------------------------------------
    # Model / Optimizer (★ with img_feature_dim parameter)
    # --------------------------------------------------------
    model = SwitchGazeNet(backbone=args.backbone, pretrained=True,
                          img_feature_dim=args.img_feature_dim)
    model = torch.nn.DataParallel(model).to(device)

    criterion = PinBallLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # ★ Cosine Annealing with warmup (manual implementation)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)

    print(f"[ExpDir] {exp_dir}")
    print(f"[Info] Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}")

    # ============================================================
    # Resume Feature
    # ============================================================
    start_epoch = 0
    best_val = float("inf")

    if args.resume:
        print(f"[Resume] Loading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)

        # Fix DataParallel key mismatch
        state_dict = ckpt["state_dict"]
        new_state = {}
        for k, v in state_dict.items():
            new_k = k.replace("module.", "") if k.startswith("module.") else k
            new_state[new_k] = v

        model.module.load_state_dict(new_state)

        # Restore optimizer
        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except:
                print("[Resume] Optimizer mismatch, ignoring optimizer resume")

        start_epoch = ckpt.get("epoch", 0)
        best_val = ckpt.get("best_error", float("inf"))

        print(f"[Resume] start_epoch={start_epoch}, best_val={best_val:.3f}")

    # --------------------------------------------------------
    # Eval-only mode
    # --------------------------------------------------------
    if args.eval_only:
        if args.checkpoint is None:
            raise ValueError("--checkpoint required for eval_only")

        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)

        test_loader = DataLoader(
            OcclusionLoader(args.data_path, os.path.join(args.data_path, test_list), transform_eval),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True
        )
        err = evaluate(model, test_loader, device, pred_dir, save_csv=True)
        print(f"[Eval-only] test_err={err:.3f}")
        return

    # --------------------------------------------------------
    # Training loop (with resume + warmup + gradient clipping)
    # --------------------------------------------------------
    best_ckpt_path = os.path.join(ckpt_dir, "best_model.pth.tar")

    for epoch in range(start_epoch, args.epochs):
        # ★ Learning rate warmup
        if epoch < args.warmup_epochs:
            warmup_lr = args.lr * (epoch + 1) / args.warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
            print(f"[Warmup] Epoch {epoch+1}/{args.warmup_epochs}, lr={warmup_lr:.6f}")

        train_loss, train_err = train_one_epoch(model, train_loader, criterion, optimizer, device, args.grad_clip)
        val_err = validate(model, val_loader, device)

        # ★ Apply scheduler after warmup
        if epoch >= args.warmup_epochs:
            scheduler.step()

        if val_err < best_val:
            best_val = val_err
            torch.save({
                "epoch": epoch + 1,
                "state_dict": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_error": best_val
            }, best_ckpt_path)
            print(f"[Best Updated @ Epoch {epoch+1}] val_err={best_val:.3f}")

        print(f"[Epoch {epoch+1}/{args.epochs}] "
              f"train_loss={train_loss:.4f}, val_err={val_err:.3f}, best={best_val:.3f}")

    # --------------------------------------------------------
    # Final test
    # --------------------------------------------------------
    test_loader = DataLoader(
        OcclusionLoader(args.data_path, os.path.join(args.data_path, test_list), transform_eval),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    test_err = evaluate(model, test_loader, device, pred_dir, args.save_csv)

    # --------------------------------------------------------
    # Log
    # --------------------------------------------------------
    log = {
        "model": "SwitchGazeNet",
        "train_list": args.train_list,
        "backbone": args.backbone,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "train_size": len(train_loader.dataset),
        "val_size": len(val_loader.dataset),
        "test_size": len(test_loader.dataset),
        "best_val_err": round(best_val, 3),
        "test_err": round(test_err, 3),
        "checkpoint": best_ckpt_path,
        "timestamp": timestamp
    }

    with open(os.path.join(exp_dir, "log.json"), "w") as f:
        json.dump(log, f, indent=4)
    print(f"[Saved] {exp_dir}/log.json")


# ============================================================
# Train / Validate / Eval (★ with gradient clipping)
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip=1.0):
    model.train()
    loss_meter, err_meter = AverageMeter(), AverageMeter()
    pbar = tqdm(loader, desc="[Train]", ncols=100)

    for head, body, conf, target, head_vec, upper_vec in pbar:
        head, body = head.to(device), body.to(device)
        conf, target = conf.to(device), target.to(device)
        head_vec, upper_vec = head_vec.to(device), upper_vec.to(device)

        optimizer.zero_grad()

        pred, var = model(head, body, conf, head_vec, upper_vec)

        loss = criterion(pred, target, var)
        loss.backward()

        # ★ Gradient clipping to prevent exploding gradients
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        err = compute_angular_error(pred, target)
        loss_meter.update(loss.item())
        err_meter.update(err)

        pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}",
                          "err": f"{err_meter.avg:.2f}°"})

    return loss_meter.avg, err_meter.avg


def validate(model, loader, device):
    model.eval()
    meter = AverageMeter()
    pbar = tqdm(loader, desc="[Val]", ncols=100)

    with torch.no_grad():
        for head, body, conf, target, head_vec, upper_vec in pbar:
            head, body = head.to(device), body.to(device)
            conf, target = conf.to(device), target.to(device)
            head_vec, upper_vec = head_vec.to(device), upper_vec.to(device)

            pred, _ = model(head, body, conf, head_vec, upper_vec)
            err = compute_angular_error(pred, target)
            meter.update(err)
            pbar.set_postfix({"err": f"{meter.avg:.2f}°"})

    return meter.avg


def evaluate(model, loader, device, pred_dir, save_csv=False):
    model.eval()
    errors = []
    pbar = tqdm(loader, desc="[Test]", ncols=100)

    with torch.no_grad():
        for head, body, conf, target, head_vec, upper_vec in pbar:
            head, body = head.to(device), body.to(device)
            conf, target = conf.to(device), target.to(device)
            head_vec, upper_vec = head_vec.to(device), upper_vec.to(device)

            pred, _ = model(head, body, conf, head_vec, upper_vec)
            err = compute_angular_error(pred, target)
            errors.append(err)
            pbar.set_postfix({"err": f"{np.mean(errors):.2f}°"})

    mean_err = float(np.mean(errors))
    print(f"[Test] Mean angular error = {mean_err:.3f}°  ({len(errors)} samples)")

    if save_csv:
        import csv
        path = os.path.join(pred_dir, "test_results.csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["idx", "angular_error"])
            for i, e in enumerate(errors):
                writer.writerow([i, e])
        print(f"[Saved] {path}")

    return mean_err


# ============================================================
if __name__ == "__main__":
    main()
# ============================================================
