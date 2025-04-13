#!/bin/bash

# Set up environment variables and paths
export PYTHONPATH=/home/bsd/cospeech/DiffGesture:$PYTHONPATH

# Paths
MODEL_DIR=/home/bsd/cospeech/DiffGesture/model
LOG_DIR=/home/bsd/cospeech/DiffGesture/log

# Create directories if they don't exist
mkdir -p "$MODEL_DIR"
mkdir -p "$LOG_DIR"

# Training parameters
BATCH_SIZE=16
EPOCHS=200
LEARNING_RATE=1e-4
RANDOM_SEED=42
POSE_DIM=141  # Change this if your direction vectors have a different dimension

# Run the training script with LMDB dataset
python /home/bsd/cospeech/DiffGesture/scripts/train_trinity.py \
    --name DiffGesture_Trinity_LMDB \
    --gpu_id 0 \
    --pose_dim $POSE_DIM \
    --model pose_diffusion \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --learning_rate $LEARNING_RATE \
    --random_seed $RANDOM_SEED \
    --model_save_path "$MODEL_DIR" \
    --train_data_path /home/bsd/cospeech/DiffGesture/data/trinity/allRec \
    --val_data_path /home/bsd/cospeech/DiffGesture/data/trinity/allTestMotion \
    --test_data_path /home/bsd/cospeech/DiffGesture/data/trinity/allTestMotion \
    --verbose

echo "Training completed!" 