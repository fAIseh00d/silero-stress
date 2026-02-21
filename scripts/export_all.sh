#!/bin/bash
# Export all model weights and prepare for distribution
# Each Python script handles ONNX merging and gzipping internally
# Usage: ./scripts/export_all.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "============================================================"
echo "Silero-Stress Full Export"
echo "============================================================"

# Check Python
if ! command -v python &> /dev/null; then
    echo "Error: Python not found"
    exit 1
fi

# Step 1: Export Ukrainian and Belarusian
echo ""
echo "[1/3] Exporting Ukrainian and Belarusian models..."
PYTHONPATH=src python scripts/extract_and_export.py

# Step 2: Export Russian N-gram components
echo ""
echo "[2/3] Exporting Russian N-gram components..."
PYTHONPATH=src python scripts/export_ru.py

# Step 3: Export Russian BERT
echo ""
echo "[3/3] Exporting Russian BERT to ONNX..."
PYTHONPATH=src python scripts/export_bert_direct.py --all

echo ""
echo "============================================================"
echo "Export Complete!"
echo "============================================================"
echo ""
echo "Run parity tests with:"
echo "  PYTHONPATH=src python -m pytest tests/test_parity.py -v"
