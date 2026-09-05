#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ONNX_PATH="${1:-$ROOT/models/yolov8n.onnx}"
ENGINE_PATH="${2:-$ROOT/models/yolov8n_fp16_orin.engine}"

test -s "$ONNX_PATH"
mkdir -p "$(dirname "$ENGINE_PATH")"
rm -f "$ENGINE_PATH"

# This engine is deliberately built on the target Jetson. Rebuild after changes
# to the ONNX model, TensorRT, CUDA, or the Jetson hardware/software release.
trtexec --onnx="$ONNX_PATH" --saveEngine="$ENGINE_PATH" --fp16 \
  --memPoolSize=workspace:1024 --skipInference
test -s "$ENGINE_PATH"
sha256sum "$ONNX_PATH" "$ENGINE_PATH"
