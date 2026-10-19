#!/usr/bin/env bash
# ==============================================================================
# SeaDino: ViT-B/16 Pretraining Pipeline on 1.3M Unlabelled Marine Image Shards
# ==============================================================================
#
# Pretrains a Vision Transformer Base (ViT-B/16, ~86M parameters) on the full
# 1.3 Million unlabelled BenthicNet seafloor imagery corpus via WebDataset streaming,
# equipped with the complete 5-component marine optical physics augmentation suite.
#
# Usage:
#   bash scripts/train_vitb_1.3m.sh [SHARD_DIR] [OUTPUT_DIR] [EXTRA_ARGS...]
#
# Examples:
#   # Default launch using external drive:
#   bash scripts/train_vitb_1.3m.sh /mnt/g/SeaDino_Data/unlabelled_shards ./checkpoints/seadino_vitb16_1.3m
#
#   # Multi-GPU / Distributed launch (e.g. torchrun on 2 GPUs):
#   torchrun --nproc_per_node=2 ssl_train.py \
#       --model_id facebook/dinov3-vitb16-pretrain-lvd1689m \
#       --use_webdataset --shard_dir /mnt/g/SeaDino_Data/unlabelled_shards \
#       --batch_size 16 --grad_accum_steps 4 --use_bf16 ...
# ==============================================================================

set -euo pipefail

# Auto-detect default directories
DEFAULT_SHARD_DIR="/mnt/g/SeaDino_Data/unlabelled_shards"
if [ ! -d "$DEFAULT_SHARD_DIR" ]; then
    if [ -d "/mnt/g/data/unlabelled_shards" ]; then
        DEFAULT_SHARD_DIR="/mnt/g/data/unlabelled_shards"
    elif [ -d "/mnt/g/data/benchmarks/unlabelled_shards" ]; then
        DEFAULT_SHARD_DIR="/mnt/g/data/benchmarks/unlabelled_shards"
    fi
fi

SHARD_DIR="${1:-$DEFAULT_SHARD_DIR}"
OUTPUT_DIR="${2:-./checkpoints/seadino_vitb16_1.3m}"
shift 2 || true
EXTRA_ARGS=("$@")

mkdir -p "$OUTPUT_DIR"

echo "=============================================================================="
echo "SeaDino ViT-B/16 Pretraining on 1.3M Unlabelled Shards"
echo "=============================================================================="
echo "  • Shard Directory:    $SHARD_DIR"
echo "  • Checkpoint Output:  $OUTPUT_DIR"
echo "  • Model Architecture: facebook/dinov3-vitb16-pretrain-lvd1689m (ViT-Base)"
echo "  • Micro Batch Size:   16 per GPU"
echo "  • Grad Accum Steps:   4 (Effective Batch Size = 64)"
echo "  • Precision:          bfloat16 (BF16)"
echo "  • Physics Augs:       Enabled (Attenuation, Turbidity, Vignetting, Snow, Rotation)"
echo "=============================================================================="

# Check if shard directory exists
if [ ! -d "$SHARD_DIR" ]; then
    echo "Warning: Shard directory '$SHARD_DIR' does not exist yet."
    echo "Please ensure your 1.3M WebDataset shards (.tar files) are located in that directory"
    echo "or pass the path as the first argument: bash scripts/train_vitb_1.3m.sh /path/to/shards"
fi

export OMP_NUM_THREADS=8
export PYTHONPATH=.

python ssl_train.py \
    --model_id facebook/dinov3-vitb16-pretrain-lvd1689m \
    --use_webdataset \
    --shard_dir "$SHARD_DIR" \
    --shard_pattern "*.tar" \
    --webdataset_num_samples 1300000 \
    --webdataset_shuffle_buffer 5000 \
    --output_dir "$OUTPUT_DIR" \
    --epochs 30 \
    --batch_size 16 \
    --grad_accum_steps 4 \
    --learning_rate 2e-4 \
    --head_lr_mult 5.0 \
    --warmup_epochs 3 \
    --lr_schedule cosine_warmup \
    --use_bf16 \
    --marine_aug \
    --global_crop_size 224 \
    --local_crop_size 96 \
    --num_global_crops 2 \
    --num_local_crops 8 \
    --use_ibot \
    --use_koleo \
    --koleo_loss_weight 0.1 \
    --teacher_ema_tau 0.994 \
    --save_interval 1 \
    --log_every_n_steps 10 \
    --num_workers 8 \
    --prefetch_factor 2 \
    "${EXTRA_ARGS[@]}"
