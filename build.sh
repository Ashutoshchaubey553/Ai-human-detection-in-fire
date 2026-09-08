#!/usr/bin/env bash
# build.sh — Render build script for Fire Detection Flask App
# Installs Python dependencies and downloads large model files from Google Drive.
# Run automatically by Render via the buildCommand in render.yaml.

set -euo pipefail

echo "=== [1/3] Installing Python dependencies ==="
pip install -r requirements.txt

echo "=== [2/3] Downloading model files ==="
mkdir -p models

# ──────────────────────────────────────────────────────────────────
# Google Drive direct-download helper
# Usage: gdrive_download <FILE_ID> <OUTPUT_PATH>
# ──────────────────────────────────────────────────────────────────
gdrive_download() {
    local FILE_ID="$1"
    local OUTPUT="$2"
    local CONFIRM_URL="https://drive.google.com/uc?export=download&id=${FILE_ID}"
    echo "  Downloading: ${OUTPUT}"
    # First request — get the confirm token (for large files)
    local RESPONSE
    RESPONSE=$(curl -sc /tmp/gdrive_cookie "${CONFIRM_URL}" -L)
    local TOKEN
    TOKEN=$(grep -o 'confirm=[a-zA-Z0-9_-]*' <<< "${RESPONSE}" | sed 's/confirm=//' | head -1)
    if [ -n "${TOKEN}" ]; then
        curl -Lb /tmp/gdrive_cookie \
             "https://drive.google.com/uc?export=download&confirm=${TOKEN}&id=${FILE_ID}" \
             -o "${OUTPUT}"
    else
        # Small file or token not needed — direct download
        curl -L "${CONFIRM_URL}" -o "${OUTPUT}"
    fi
    echo "  ✓ Saved: ${OUTPUT} ($(du -sh "${OUTPUT}" | cut -f1))"
}

# ──────────────────────────────────────────────────────────────────
# REPLACE the FILE_ID values below with your actual Google Drive IDs
# after you upload and share the files (see README / deployment guide).
#
# How to get FILE_ID from a Google Drive share link:
#   https://drive.google.com/file/d/FILE_ID_HERE/view?usp=sharing
#                                    ^^^^^^^^^^^^
# ──────────────────────────────────────────────────────────────────

FIRE_MODEL_ID="REPLACE_WITH_FIRE_MODEL_FILE_ID"       # fire_detection_model.h5  (~7 MB)
YOLO_WEIGHTS_ID="REPLACE_WITH_YOLO_WEIGHTS_FILE_ID"   # yolov3.weights           (~237 MB)
YOLO_CFG_ID="REPLACE_WITH_YOLO_CFG_FILE_ID"           # yolov3.cfg               (~8 KB)
COCO_NAMES_ID="REPLACE_WITH_COCO_NAMES_FILE_ID"       # coco.names               (~1 KB)

# Download fire detection model (~7 MB)
if [ ! -f "models/fire_detection_model.h5" ]; then
    gdrive_download "${FIRE_MODEL_ID}" "models/fire_detection_model.h5"
else
    echo "  ✓ models/fire_detection_model.h5 already present — skipping."
fi

# Download YOLOv3 weights (~237 MB)
if [ ! -f "models/yolov3.weights" ]; then
    gdrive_download "${YOLO_WEIGHTS_ID}" "models/yolov3.weights"
else
    echo "  ✓ models/yolov3.weights already present — skipping."
fi

# Download YOLOv3 config (~8 KB)
if [ ! -f "models/yolov3.cfg" ]; then
    gdrive_download "${YOLO_CFG_ID}" "models/yolov3.cfg"
else
    echo "  ✓ models/yolov3.cfg already present — skipping."
fi

# Download COCO class names (~1 KB)
if [ ! -f "models/coco.names" ]; then
    gdrive_download "${COCO_NAMES_ID}" "models/coco.names"
else
    echo "  ✓ models/coco.names already present — skipping."
fi

echo "=== [3/3] Build complete ==="
