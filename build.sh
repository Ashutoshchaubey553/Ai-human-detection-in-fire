#!/usr/bin/env bash
# build.sh — Render build script for Fire Detection Flask App
# Installs Python dependencies and ensures YOLO and CNN model files are downloaded and verified.
# Run automatically by Render via the buildCommand in render.yaml.

set -euo pipefail

echo "=== [1/2] Installing Python dependencies ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== [2/2] Verifying and downloading model weights ==="
python download_models.py

echo "=== Build complete ==="
