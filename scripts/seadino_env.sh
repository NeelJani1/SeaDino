#!/usr/bin/env bash
# ==============================================================================
# SeaDino External Storage Environment Configuration
# ==============================================================================
# Source this file to route SeaDino to your external checkpoint vault:
#   source scripts/seadino_env.sh
# ==============================================================================

# Default target directory on external drive (WSL /mnt/d or local path)
TARGET_VAULT="${SEADINO_EXTERNAL_VAULT:-/home/njan320/Neel/Checkpoints_wsl}"

if [ ! -d "$TARGET_VAULT" ] && [ -d "/mnt/d/Neel/Checkpoints_wsl" ]; then
    TARGET_VAULT="/mnt/d/Neel/Checkpoints_wsl"
fi

export SEADINO_CHECKPOINTS_DIR="${TARGET_VAULT}/checkpoints"
export SEADINO_LOGS_DIR="${TARGET_VAULT}/logs"
export SEADINO_RESULTS_DIR="${TARGET_VAULT}/results"

echo "================================================================================"
echo "[SeaDino] External Storage Paths Activated:"
echo "  • Vault Base:   $TARGET_VAULT"
echo "  • Checkpoints:  $SEADINO_CHECKPOINTS_DIR"
echo "  • Logs:         $SEADINO_LOGS_DIR"
echo "  • Results:      $SEADINO_RESULTS_DIR"
echo "================================================================================"
