#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/models"

# COCO model: person, chair, backpack, and bottle are all present. Keep model
# files outside Git; use a license appropriate to the intended deployment.
curl --fail --location --retry 3 \
  --output "$ROOT/models/yolov8n.onnx" \
  https://huggingface.co/Kalray/yolov8/resolve/main/yolov8n.onnx
echo "Saved $ROOT/models/yolov8n.onnx"
