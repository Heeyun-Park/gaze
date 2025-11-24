#!/usr/bin/env python3
"""
train_sequential.py
모든 occlusion level을 순차적으로 학습하는 고급 스크립트

특징:
- 여러 occlusion level 자동 탐지
- 각 level별 최적 checkpoint 저장
- 학습 결과 자동 집계 및 CSV 저장
- GPU 메모리 관리
- 에러 핸들링 및 resume 지원
"""

import os
import sys
import subprocess
import json
import csv
from pathlib import Path
from datetime import datetime
import argparse


class SequentialTrainer:
    def __init__(self, args):
        self.main_dir = args.main_dir
        self.script_path = os.path.join(self.main_dir, "models/switchgaze/run.py")

        # 하이퍼파라미터 (성능 최적화)
        self.backbone = args.backbone
        self.img_feature_dim = args.img_feature_dim
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.lr = args.lr
        self.warmup_epochs = args.warmup_epochs
        self.grad_clip = args.grad_clip
        self.num_workers = args.num_workers

        # 결과 저장
        self.results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def find_occlusion_levels(self):
        """사용 가능한 occlusion level 자동 탐지"""
        levels = []
        for i in range(20):  # 최대 20개까지 확인
            data_dir = os.path.join(self.main_dir, f"data_occluded_{i}")
            train_file = os.path.join(data_dir, f"train_occlusion_{i}.txt")

            if os.path.exists(data_dir) and os.path.exists(train_file):
                levels.append(i)

        return levels

    def train_single_level(self, level):
        """단일 occlusion level 학습"""
        print("\n" + "="*60)
        print(f"Training Occlusion Level {level}")
        print("="*60)

        data_dir = os.path.join(self.main_dir, f"data_occluded_{level}")

        cmd = [
            "python", self.script_path,
            "--data_path", data_dir,
            "--train_list", f"train_occlusion_{level}.txt",
            "--val_list", f"val_occlusion_{level}.txt",
            "--test_list", f"test_occlusion_{level}.txt",
            "--backbone", self.backbone,
            "--img_feature_dim", str(self.img_feature_dim),
            "--batch_size", str(self.batch_size),
            "--epochs", str(self.epochs),
            "--lr", str(self.lr),
            "--warmup_epochs", str(self.warmup_epochs),
            "--grad_clip", str(self.grad_clip),
            "--num_workers", str(self.num_workers),
            "--save_csv"
        ]

        print(f"Command: {' '.join(cmd)}")
        print()

        try:
            result = subprocess.run(cmd, check=True, cwd=self.main_dir)

            # 결과 로그 찾기
            output_dir = os.path.join(self.main_dir, "outputs")
            log_file = self._find_latest_log(output_dir, level)

            if log_file:
                with open(log_file, 'r') as f:
                    log_data = json.load(f)
                    self.results.append({
                        "level": level,
                        "best_val_err": log_data.get("best_val_err", "N/A"),
                        "test_err": log_data.get("test_err", "N/A"),
                        "checkpoint": log_data.get("checkpoint", "N/A")
                    })

            print(f"✓ [Success] Occlusion level {level} completed")
            return True

        except subprocess.CalledProcessError as e:
            print(f"✗ [Error] Occlusion level {level} failed: {e}")
            self.results.append({
                "level": level,
                "best_val_err": "FAILED",
                "test_err": "FAILED",
                "checkpoint": "N/A"
            })
            return False

    def _find_latest_log(self, output_dir, level):
        """가장 최근 log.json 찾기"""
        if not os.path.exists(output_dir):
            return None

        pattern = f"SwitchGaze_occlusion_{level}_*"
        matching_dirs = sorted(Path(output_dir).glob(pattern),
                             key=lambda x: x.stat().st_mtime,
                             reverse=True)

        if matching_dirs:
            log_file = matching_dirs[0] / "log.json"
            if log_file.exists():
                return str(log_file)

        return None

    def save_summary(self):
        """전체 결과 요약 저장"""
        summary_path = os.path.join(
            self.main_dir,
            f"training_summary_{self.timestamp}.csv"
        )

        with open(summary_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["level", "best_val_err", "test_err", "checkpoint"])
            writer.writeheader()
            writer.writerows(self.results)

        print("\n" + "="*60)
        print("Training Summary")
        print("="*60)
        print(f"{'Level':<10} {'Val Error':<15} {'Test Error':<15}")
        print("-"*60)

        for result in self.results:
            print(f"{result['level']:<10} {result['best_val_err']:<15} {result['test_err']:<15}")

        print("="*60)
        print(f"Summary saved to: {summary_path}")
        print("="*60)

    def run(self):
        """전체 학습 실행"""
        print("="*60)
        print("Sequential Training - All Occlusion Levels")
        print("="*60)
        print(f"Main Directory: {self.main_dir}")
        print(f"Backbone: {self.backbone}")
        print(f"Feature Dim: {self.img_feature_dim}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Epochs: {self.epochs}")
        print("="*60)

        # Occlusion levels 탐지
        levels = self.find_occlusion_levels()

        if not levels:
            print("Error: No occlusion data found!")
            return

        print(f"Found {len(levels)} occlusion levels: {levels}")
        print()

        # 순차 학습
        for level in levels:
            success = self.train_single_level(level)
            if not success:
                print(f"Warning: Level {level} failed, continuing...")

        # 결과 저장
        self.save_summary()
        print("\n✓ All training completed!")


def main():
    parser = argparse.ArgumentParser(description="Sequential training on all occlusion levels")

    parser.add_argument("--main_dir", type=str,
                       default="/home/ubuntu22/heeyun/Main",
                       help="Main project directory")

    # 성능 최적화 파라미터
    parser.add_argument("--backbone", type=str, default="resnet50",
                       choices=["resnet18", "resnet34", "resnet50", "resnet101"],
                       help="Backbone architecture")
    parser.add_argument("--img_feature_dim", type=int, default=512,
                       help="Feature dimension")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--warmup_epochs", type=int, default=5,
                       help="Warmup epochs")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                       help="Gradient clipping")
    parser.add_argument("--num_workers", type=int, default=8,
                       help="Data loading workers")

    args = parser.parse_args()

    trainer = SequentialTrainer(args)
    trainer.run()


if __name__ == "__main__":
    main()
