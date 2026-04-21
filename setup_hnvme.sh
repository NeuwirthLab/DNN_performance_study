#!/bin/bash
# =============================================================================
# setup_hnvme.sh — Run ONCE on the Helma LOGIN node
#
# Copies project source code + ZIPs from NFS (/home/atuin) to the
# hnvme workspace (/hnvme/workspace/m102cd10-amrit) which IS accessible
# from Helma compute nodes.
#
# IMPORTANT — hnvme inode (file count) quota:
#   Soft: 61K files   Hard: 81K files
#   tiny-imagenet-200/ extracted = 120K files → DOES NOT FIT
#   So we copy tiny-imagenet-200.ZIP (1 file) and extract in $TMPDIR at runtime.
#   ImageNet ZIP (150GB, 1 file) is too large to copy here; read from NFS or
#   $HPCVAULT instead.
#
# What gets copied:
#   - Source code (~60 files, ~3MB)
#   - tiny-imagenet-200.zip (1 file, 248MB) — extracted in SLURM $TMPDIR
#   - tests/ directory
#   - SLURM scripts, helper scripts, docs
#
# What does NOT get copied:
#   - tiny-imagenet-200/ (extracted) — 120K files, exceeds 81K inode limit
#   - imagenet ZIP (150GB) — stays on NFS or $HPCVAULT
#   - build/, .git/, Windows binaries, checkpoints
#
# Usage:
#   chmod +x setup_hnvme.sh
#   bash setup_hnvme.sh
# =============================================================================

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
NFS_RESNET="/home/atuin/m102cd/m102cd10/resnet_multi_gpu/resnet50"
NFS_IMAGENET_ZIP="/home/atuin/m102cd/m102cd10/resnet_multi_gpu/imagenet/imagenet-object-localization-challenge.zip"

HNVME_BASE="/hnvme/workspace/m102cd10-amrit"
HNVME_RESNET="$HNVME_BASE/resnet50"

# ── Preflight checks ──────────────────────────────────────────────────────────
echo "=== hnvme Setup Script ==="
echo ""

if [ ! -d "$NFS_RESNET" ]; then
    echo "ERROR: Source project not found: $NFS_RESNET"
    exit 1
fi

if [ ! -d "$HNVME_BASE" ]; then
    echo "ERROR: hnvme workspace not found: $HNVME_BASE"
    echo ""
    echo "Create it first:"
    echo "  ws_allocate -F hnvme m102cd10-amrit 90"
    exit 1
fi

# ── Show current quota ───────────────────────────────────────────────────────
echo "Current hnvme quota:"
lfs quota -u "$USER" /hnvme 2>/dev/null || echo "(Could not read quota)"
echo ""
echo "Current hnvme usage:"
du -sh "$HNVME_BASE"/* 2>/dev/null || echo "(Workspace empty)"
echo ""

# ── Step 1: Copy source code + ZIPs (NOT extracted dirs) ─────────────────────
echo "--- Step 1/2: Copying project source code + ZIPs ---"
echo "  From: $NFS_RESNET"
echo "  To:   $HNVME_RESNET"
echo ""

mkdir -p "$HNVME_RESNET"

rsync -av --progress \
    --exclude='build/' \
    --exclude='*.err' \
    --exclude='*.out' \
    --exclude='build*/' \
    --exclude='__pycache__/' \
    --exclude='*.exe' \
    --exclude='*.exp' \
    --exclude='*.lib' \
    --exclude='*.obj' \
    --exclude='*.o' \
    --exclude='tiny-imagenet-200/' \
    --exclude='checkpoints/' \
    --exclude='.git/' \
    --exclude='.claude/' \
    --exclude='figures/' \
    --exclude='plots/' \
    --exclude='profile_results/' \
    --exclude='tests/build*/' \
    "$NFS_RESNET/" \
    "$HNVME_RESNET/"

echo ""
echo "Code copy complete."
echo ""

# ── Step 2: Verify tiny-imagenet-200.zip was included ────────────────────────
echo "--- Step 2/2: Checking data ZIPs ---"
if [ -f "$HNVME_RESNET/tiny-imagenet-200.zip" ]; then
    TINY_SIZE=$(du -h "$HNVME_RESNET/tiny-imagenet-200.zip" | awk '{print $1}')
    echo "  tiny-imagenet-200.zip: $TINY_SIZE  ✓ (will extract in \$TMPDIR at job time)"
else
    echo "  WARNING: tiny-imagenet-200.zip not found on hnvme"
    if [ -f "$NFS_RESNET/tiny-imagenet-200.zip" ]; then
        echo "  Copying from NFS..."
        rsync -av --progress "$NFS_RESNET/tiny-imagenet-200.zip" "$HNVME_RESNET/"
    else
        echo "  Not found on NFS either. Download with: bash download_data.sh"
    fi
fi
echo ""

# Full ImageNet status
if [ -f "$NFS_IMAGENET_ZIP" ]; then
    IMG_SIZE=$(du -h "$NFS_IMAGENET_ZIP" | awk '{print $1}')
    echo "  ImageNet ZIP ($IMG_SIZE): stays on NFS (too large for hnvme copy)"
    echo "    Path: $NFS_IMAGENET_ZIP"
    echo "    SLURM script will read directly from NFS or \$HPCVAULT"
else
    echo "  WARNING: ImageNet ZIP not found at: $NFS_IMAGENET_ZIP"
fi
echo ""

# ── Show final state ─────────────────────────────────────────────────────────
echo "=== Final State ==="
echo ""
echo "Files on hnvme ($HNVME_RESNET):"
ls "$HNVME_RESNET/"
echo ""
echo "File count:"
FILE_COUNT=$(find "$HNVME_RESNET" -type f | wc -l)
DIR_COUNT=$(find "$HNVME_RESNET" -type d | wc -l)
echo "  $FILE_COUNT files, $DIR_COUNT directories (inode limit: 81K)"
echo ""
echo "Updated hnvme quota:"
lfs quota -u "$USER" /hnvme 2>/dev/null || echo "(Could not read quota)"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "=== Setup complete ==="
echo ""
echo "Your hnvme workspace:"
echo "  Code: $HNVME_RESNET"
echo ""
echo "Next steps:"
echo ""
echo "  1. Smoke test (Tiny ImageNet, 2 GPUs, ~20 min):"
echo "     cd $HNVME_RESNET && sbatch smoke_test_helma.slurm"
echo ""
echo "  2. Full ImageNet training (2 x H100, ~10 hours):"
echo "     Update IMAGENET_ZIP in run_multigpu_tmpdir_v2.slurm to point to"
echo "     wherever your 150GB ZIP lives (NFS or \$HPCVAULT), then:"
echo "     cd $HNVME_RESNET && sbatch run_multigpu_tmpdir_v2.slurm"
