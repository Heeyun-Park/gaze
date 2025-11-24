# 순차 학습 가이드 (Sequential Training Guide)

모든 occlusion level을 자동으로 순차 학습하는 스크립트입니다.

## 🚀 빠른 시작

### 방법 1: Bash 스크립트 (간단)

```bash
cd /home/ubuntu22/heeyun/Main
bash train_all_occlusions.sh
```

### 방법 2: Python 스크립트 (고급, 권장)

```bash
cd /home/ubuntu22/heeyun/Main
python train_sequential.py
```

---

## 📋 주요 특징

### Bash 스크립트 (`train_all_occlusions.sh`)
- ✅ 간단하고 빠름
- ✅ 모든 occlusion level 자동 탐지
- ✅ 각 level별 독립 학습
- ✅ 에러 발생 시 자동 종료

### Python 스크립트 (`train_sequential.py`)
- ✅ 자동 데이터 탐지
- ✅ 학습 결과 자동 집계
- ✅ CSV 요약 보고서 생성
- ✅ 에러 발생해도 계속 진행
- ✅ 커스터마이징 가능

---

## ⚙️ 설정 커스터마이징

### Python 스크립트 옵션

```bash
python train_sequential.py \
  --main_dir /home/ubuntu22/heeyun/Main \
  --backbone resnet50 \
  --img_feature_dim 512 \
  --batch_size 32 \
  --epochs 100 \
  --lr 1e-4 \
  --warmup_epochs 5 \
  --grad_clip 1.0 \
  --num_workers 8
```

### 옵션 설명

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--backbone` | resnet50 | 백본 네트워크 (resnet18/34/50/101) |
| `--img_feature_dim` | 512 | Feature dimension (256/512/1024) |
| `--batch_size` | 32 | 배치 크기 (GPU 메모리에 맞게) |
| `--epochs` | 100 | 학습 epoch 수 |
| `--lr` | 1e-4 | Learning rate |
| `--warmup_epochs` | 5 | Warmup epoch 수 |
| `--grad_clip` | 1.0 | Gradient clipping |
| `--num_workers` | 8 | Data loader workers |

---

## 🎯 성능 최적화 팁

### 1. **최고 성능 (GPU 메모리 충분)**
```bash
python train_sequential.py \
  --backbone resnet101 \
  --img_feature_dim 1024 \
  --batch_size 16 \
  --epochs 150 \
  --lr 5e-5
```

### 2. **균형 잡힌 설정 (권장)**
```bash
python train_sequential.py \
  --backbone resnet50 \
  --img_feature_dim 512 \
  --batch_size 32 \
  --epochs 100
```

### 3. **빠른 학습 (테스트용)**
```bash
python train_sequential.py \
  --backbone resnet18 \
  --img_feature_dim 256 \
  --batch_size 64 \
  --epochs 50
```

---

## 📊 학습 결과 확인

### 1. 개별 결과
각 occlusion level의 결과는 다음 위치에 저장됩니다:
```
outputs/SwitchGaze_occlusion_X_YYYYMMDD_HHMM/
├── checkpoint/best_model.pth.tar  # 최고 성능 모델
├── predictions/test_results.csv   # 테스트 결과
└── log.json                        # 학습 로그
```

### 2. 전체 요약 (Python 스크립트)
```
training_summary_YYYYMMDD_HHMMSS.csv
```

예시:
```csv
level,best_val_err,test_err,checkpoint
0,12.34,12.56,outputs/.../best_model.pth.tar
1,13.21,13.45,outputs/.../best_model.pth.tar
2,14.12,14.38,outputs/.../best_model.pth.tar
```

---

## 🔧 Bash 스크립트 수정하기

설정을 바꾸려면 `train_all_occlusions.sh` 파일의 상단 부분을 수정하세요:

```bash
# 학습 하이퍼파라미터 수정
BACKBONE="resnet101"          # 더 강력한 모델
IMG_FEATURE_DIM=1024          # 더 큰 feature
BATCH_SIZE=16                 # GPU 메모리 적게
EPOCHS=150                    # 더 오래 학습
```

---

## 🛠️ 문제 해결

### GPU 메모리 부족
```bash
# batch_size 줄이기
python train_sequential.py --batch_size 16

# 또는 더 작은 모델 사용
python train_sequential.py --backbone resnet18 --img_feature_dim 256
```

### 특정 level만 학습
단일 level 학습:
```bash
cd /home/ubuntu22/heeyun/Main
python models/switchgaze/run.py \
  --data_path data_occluded_0 \
  --train_list train_occlusion_0.txt \
  --backbone resnet50 \
  --batch_size 32 \
  --epochs 100
```

### 중단된 학습 재개
```bash
python models/switchgaze/run.py \
  --resume outputs/SwitchGaze_xxx/checkpoint/best_model.pth.tar \
  --data_path data_occluded_0 \
  --train_list train_occlusion_0.txt \
  ... (other args)
```

---

## 📈 예상 성능

개선된 모델 (Temporal Attention + Deeper Networks):
- **Baseline (gaze360)**: ~15-18° angular error
- **개선 모델**: ~12-15° angular error
- **예상 개선**: 3-5° 각도 오차 감소

---

## 💡 추가 기능

### 병렬 학습 (여러 GPU)
GPU가 여러 개라면 동시에 여러 level을 학습할 수 있습니다:

```bash
# Terminal 1 (GPU 0)
CUDA_VISIBLE_DEVICES=0 python models/switchgaze/run.py --data_path data_occluded_0 ...

# Terminal 2 (GPU 1)
CUDA_VISIBLE_DEVICES=1 python models/switchgaze/run.py --data_path data_occluded_1 ...
```

### 학습 모니터링
```bash
# tensorboard 사용 (추가 구현 필요)
# 또는 실시간 로그 확인
tail -f outputs/SwitchGaze_xxx/train.log
```

---

## 📞 도움말

문제가 발생하면:
1. GPU 메모리 확인: `nvidia-smi`
2. 데이터 경로 확인: `ls data_occluded_*`
3. 로그 확인: `cat outputs/latest/log.json`

Good luck! 🚀
