#!/bin/bash
# Run final parity test comparing all three implementations
#
# Requirements:
# - .venv: has silero-stress 1.4 installed + torch
# - .onnx_venv: clean env with numpy + onnxruntime (no silero-stress)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

VENV_PYTHON=".venv/bin/python"
ONNX_VENV_PYTHON=".onnx_venv/bin/python"
SCRIPT="scripts/final_parity_test.py"

echo "========================================"
echo "FINAL PARITY TEST - silero-stress"
echo "========================================"
echo ""
echo "Project: $PROJECT_DIR"
echo "1.4 venv: $VENV_PYTHON"
echo "ONNX venv: $ONNX_VENV_PYTHON"
echo ""

# Step 1: Generate reference from 1.4
echo "========================================"
echo "STEP 1: Generate reference from 1.4"
echo "========================================"
$VENV_PYTHON $SCRIPT --mode=reference --output=reference_1.4.json
echo ""

# Step 2: Test torch wrapper
echo "========================================"
echo "STEP 2: Test torch wrapper (from src/)"
echo "========================================"
PYTHONPATH=src $VENV_PYTHON $SCRIPT --mode=torch --reference=reference_1.4.json || true
echo ""

# Step 3: Test numpy/onnx
echo "========================================"
echo "STEP 3: Test numpy/onnx (from src/)"
echo "========================================"
PYTHONPATH=src $ONNX_VENV_PYTHON $SCRIPT --mode=numpy --reference=reference_1.4.json || true
echo ""

echo "========================================"
echo "DONE"
echo "========================================"
echo ""
echo "Results saved to:"
echo "  - reference_1.4.json (original 1.4 outputs)"
echo "  - parity_torch.json (torch wrapper comparison)"
echo "  - parity_numpy.json (numpy/onnx comparison)"
