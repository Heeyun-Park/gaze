#!/bin/bash
# ============================================================
# train_all_occlusions.sh
# 모든 occlusion level을 순차적으로 학습하는 스크립트
# ============================================================

# 설정
MAIN_DIR="/home/ubuntu22/heeyun/Main"
SCRIPT_PATH="${MAIN_DIR}/models/switchgaze/run.py"

# 학습 하이퍼파라미터 (성능 최적화)
BACKBONE="resnet50"           # resnet50 사용 (더 강력함)
IMG_FEATURE_DIM=512           # 512 feature dimension
BATCH_SIZE=32                 # GPU 메모리에 맞게 조절
EPOCHS=100                    # 충분한 학습
LR=1e-4                       # Learning rate
WARMUP_EPOCHS=5               # Warmup epochs
GRAD_CLIP=1.0                 # Gradient clipping
NUM_WORKERS=8                 # Data loading workers

echo "========================================="
echo "Starting Sequential Training on All Occlusion Levels"
echo "========================================="
echo "Backbone: ${BACKBONE}"
echo "Feature Dim: ${IMG_FEATURE_DIM}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Epochs: ${EPOCHS}"
echo "========================================="

# occlusion level 0부터 순차적으로 학습
for i in 0 1 2 3 4 5 6 7 8 9; do
  DATA_DIR="${MAIN_DIR}/data_occluded_${i}"

  # 데이터 디렉토리 존재 확인
  if [ ! -d "${DATA_DIR}" ]; then
    echo "[Skip] data_occluded_${i} does not exist"
    continue
  fi

  # train 파일 존재 확인
  TRAIN_LIST="${DATA_DIR}/train_occlusion_${i}.txt"
  if [ ! -f "${TRAIN_LIST}" ]; then
    echo "[Skip] ${TRAIN_LIST} does not exist"
    continue
  fi

  echo ""
  echo "========================================="
  echo "Training on Occlusion Level ${i}"
  echo "Data Path: ${DATA_DIR}"
  echo "========================================="

  cd "${MAIN_DIR}"

  python "${SCRIPT_PATH}" \
    --data_path "${DATA_DIR}" \
    --train_list "train_occlusion_${i}.txt" \
    --val_list "val_occlusion_${i}.txt" \
    --test_list "test_occlusion_${i}.txt" \
    --backbone "${BACKBONE}" \
    --img_feature_dim ${IMG_FEATURE_DIM} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --warmup_epochs ${WARMUP_EPOCHS} \
    --grad_clip ${GRAD_CLIP} \
    --num_workers ${NUM_WORKERS} \
    --save_csv

  # 학습 결과 확인
  if [ $? -eq 0 ]; then
    echo "[Success] Occlusion level ${i} training completed"
  else
    echo "[Error] Occlusion level ${i} training failed"
    exit 1
  fi

  echo "========================================="
done

echo ""
echo "========================================="
echo "All Training Completed Successfully!"
echo "========================================="
