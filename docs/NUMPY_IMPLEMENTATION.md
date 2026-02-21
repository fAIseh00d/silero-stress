# Pure NumPy Implementation for Silero-Stress

This document describes the PyTorch-free implementation of silero-stress models.

> **Technical Reference:** For detailed BERT architecture, tokenizer implementation, and weight extraction, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Overview

The original silero-stress library uses PyTorch (~2GB dependency) for inference. This implementation reduces the dependency to just NumPy (~25MB). ONNX Runtime (~17MB) is only needed for Russian homograph disambiguation (BERT).

### Performance Comparison

| Metric | PyTorch | NumPy | Improvement |
|--------|---------|-------|-------------|
| **Per-sentence latency (RU)** | 4.35ms | 2.95ms | **1.5x faster** |
| **Per-sentence latency (UKR)** | 0.95ms | 0.20ms | **4.7x faster** |
| **Per-sentence latency (BEL)** | 1.00ms | 0.20ms | **4.9x faster** |
| **Import memory** | ~2GB (torch) | ~25MB (numpy) | **80x smaller** |

*See [Benchmarks](#benchmarks) section for methodology. Run `scripts/benchmark.py` to reproduce.*

## Architecture

### Language Support

| Language | Dependencies | Components |
|----------|--------------|------------|
| **ru** (Russian) | NumPy + ONNX Runtime (optional) | N-gram embeddings, stress/yo classifiers, BERT homograph solver |
| **ukr** (Ukrainian) | NumPy only | N-gram embeddings + stress classifier |
| **bel** (Belarusian) | NumPy only | N-gram embeddings + stress classifier |
| **20 others** | None | Dictionary-based (already PyTorch-free) |

### Model Components

#### N-gram Accentor (ru, ukr, bel)

```
Input: "привет"
   ↓
N-gram extraction: ["<п", "<пр", "при", "рив", "иве", "вет", "ет>", ...]
   ↓
Embedding lookup + mean pooling → 16-dim vector
   ↓
MLP classifier (16 → 64 → 128 → 64 → 10)
   ↓
Output: stress position (vowel index)
```

#### Russian Homograph Solver (BERT-based, optional)

Uses a 3-layer BERT model for context-aware homograph disambiguation. See [ARCHITECTURE.md](ARCHITECTURE.md) for full details.

## Exported Files

Files are organized by language in `src/silero_stress/data/<lang>/`:

### Russian (`data/ru/`) — 34MB total (3MB without BERT)

| File | Size | Description |
|------|------|-------------|
| `bert_ru.onnx.gz` | 30MB | BERT encoder for homograph disambiguation (optional) |
| `homo_clf_ru.onnx.gz` | 581KB | Homograph binary classifier (requires BERT) |
| `embedding_ru.npz` | 1.9MB | N-gram embeddings (126K × 16, float32) |
| `stress_clf_ru_weights.npz` | 74KB | Stress classifier MLP weights |
| `yo_clf_ru_weights.npz` | 73KB | Yo (ё) classifier MLP weights |
| `ngram_dict_ru.json.gz` | 708KB | N-gram to index mapping |
| `exceptions_ru.json.gz` | 112KB | 16,976 exception words |
| `vocab_ru.json.gz` | 601KB | BERT vocabulary (83K tokens) |
| `homodict_ru.json.gz` | 26KB | 1,924 homographs |
| `yohomodict_ru.json.gz` | 2.4KB | 164 yo-homographs |
| `special_tokens_ru.json.gz` | 113B | BERT special token IDs |

*Note: NPZ files are uncompressed (they use internal zip). BERT files are only needed for homograph disambiguation.*

### Ukrainian (`data/ukr/`) — 2.4MB total

| File | Size | Description |
|------|------|-------------|
| `embedding_ukr.npz` | 1.5MB | N-gram embeddings (130K × 16, int8 quantized) |
| `stress_clf_ukr_weights.npz` | 74KB | Stress classifier MLP weights |
| `ngram_dict_ukr.json.gz` | 750KB | N-gram mapping |
| `exceptions_ukr.json.gz` | 42B | No exceptions |

### Belarusian (`data/bel/`) — 2.1MB total

| File | Size | Description |
|------|------|-------------|
| `embedding_bel.npz` | 1.3MB | N-gram embeddings (104K × 16, int8 quantized) |
| `stress_clf_bel_weights.npz` | 74KB | Stress classifier MLP weights |
| `ngram_dict_bel.json.gz` | 592KB | N-gram mapping |
| `exceptions_bel.json.gz` | 85KB | 12,403 exception words |

## Usage

### All Languages

```python
from silero_stress import load_accentor

# Russian (with homograph solver)
ru = load_accentor('ru', use_homosolver=True)
print(ru('Привет, как дела?'))
# Output: Прив+ет, к+ак дел+а?

# Russian (without homograph solver - no ONNX needed)
ru_simple = load_accentor('ru', use_homosolver=False)

# Ukrainian
ukr = load_accentor('ukr')
print(ukr('Привіт, як справи?'))
# Output: Прив+іт, +як справ+и?

# Belarusian
bel = load_accentor('bel')
print(bel('Добры дзень!'))
# Output: Д+обры дз+ень!

# With GPU acceleration (requires onnxruntime-gpu)
ru_gpu = load_accentor('ru', use_homosolver=True, device='tensorrt')
```

## Scripts

All scripts are in `scripts/`:

| Script | Purpose |
|--------|---------|
| `export_all.sh` | Master script - exports all languages |
| `extract_and_export.py` | Export Ukrainian/Belarusian models |
| `export_ru.py` | Export Russian accentor components (n-gram) |
| `export_bert_direct.py` | Export Russian BERT + classifier to ONNX |

### Re-exporting Models

```bash
# Export all languages (recommended)
./scripts/export_all.sh

# Or individually:
PYTHONPATH=src python scripts/extract_and_export.py   # UKR/BEL
PYTHONPATH=src python scripts/export_ru.py            # RU n-gram
PYTHONPATH=src python scripts/export_bert_direct.py --all  # RU BERT
```

### Export Order (Critical)

For UKR/BEL, embedding weights must be extracted BEFORE model dequantization:

```
1. Load RAW model (int8 embeddings)
2. Extract int8 weights + scale + zero_point
3. Save to NPZ
4. THEN load with dequantization for testing
```

For RU BERT, embeddings must be dequantized BEFORE ONNX export:

```
1. Load model
2. Dequantize BERT word embeddings in place
3. Copy to transformers model
4. Export to ONNX (now contains float32 weights)
```

## Implementation Details

### N-gram Extraction

Character n-grams with boundary markers:

```python
def word_ngrams(word, min_n=1, max_n=None):
    if max_n is None:
        max_n = len(word) + 3
    word_with_markers = f"<{word}>"
    ngrams = []
    for n in range(min_n, max_n + 1):
        for i in range(len(word_with_markers) - n + 1):
            ngrams.append(word_with_markers[i:i+n])
    return ngrams
```

### Embedding Storage & Dequantization

Different languages have different embedding storage formats:

| Language | N-gram Storage | BERT Storage | Notes |
|----------|---------------|--------------|-------|
| **RU** | float32 | int8 (dequantized at export) | N-gram used as-is |
| **UKR** | int8 | N/A | Dequantize at load |
| **BEL** | int8 | N/A | Dequantize at load |

#### Ukrainian/Belarusian (int8 storage)

```python
# Storage format (embedding_ukr.npz, embedding_bel.npz)
{
    'weights': int8_array,      # Shape: (vocab_size, 16)
    'scale': float32_array,     # Shape: (1,)
    'zero_point': float32_array # Shape: (1,)
}

# Runtime dequantization (done at load time)
emb_data = np.load('embedding_ukr.npz')
weights = emb_data['scale'][0] * (emb_data['weights'].astype(np.float32) - emb_data['zero_point'][0])
```

#### Russian (float32 storage)

```python
# Storage format (embedding_ru.npz)
{
    'weights': float32_array,   # Shape: (vocab_size, 16) - already float!
    'scale': float32_array,     # Not used (present for consistency)
    'zero_point': float32_array # Not used
}

# No dequantization needed
emb_data = np.load('embedding_ru.npz')
weights = emb_data['weights']  # Use directly
```

#### Russian BERT (dequantized at export time)

The BERT word embeddings are stored as int8 in the original model but are
**dequantized during ONNX export**. The exported `bert_ru.onnx` contains
float32 weights, so no runtime dequantization is needed.

### Stress Classification

The MLP predicts which vowel receives stress (0-9 index):

```python
def classify_stress(embedding, weights):
    x = embedding
    x = np.dot(x, weights['0_weight'].T) + weights['0_bias']
    x = np.maximum(x, 0)  # ReLU
    x = np.dot(x, weights['2_weight'].T) + weights['2_bias']
    x = np.maximum(x, 0)
    x = np.dot(x, weights['4_weight'].T) + weights['4_bias']
    x = np.maximum(x, 0)
    x = np.dot(x, weights['6_weight'].T) + weights['6_bias']
    return np.argmax(x)
```

### Exception Dictionary

Some words have pre-defined stress positions (character index, not vowel index):

```python
exceptions = {
    'вечарам': [1, -1],  # stress at char position 1, no yo replacement
    'москва': [5, -1],   # stress at char position 5
    ...
}
```

## Why NumPy is Faster Than PyTorch

1. **No TorchScript interpreter overhead**: The original model runs n-gram extraction in TorchScript's Python interpreter, which is slower than native Python/NumPy.

2. **No JIT compilation**: PyTorch TorchScript has first-run compilation overhead (~50ms). NumPy has no warmup.

3. **Smaller memory footprint**: NumPy arrays are more memory-efficient than PyTorch tensors for inference.

4. **No framework overhead**: Direct NumPy operations avoid PyTorch's autograd and tensor management overhead.

### GPU Acceleration for BERT

The n-gram MLP takes only ~48µs on CPU, so GPU wouldn't help there. But the BERT homograph solver benefits significantly from GPU:

| SeqLen | CPU | CUDA | TensorRT | Speedup |
|--------|-----|------|----------|---------|
| 8 | 0.87ms | 0.71ms | 0.25ms | **3.5x** |
| 16 | 1.10ms | 0.79ms | 0.26ms | **4.2x** |
| 32 | 1.78ms | 0.87ms | 0.28ms | **6.4x** |
| 64 | 2.62ms | 0.92ms | 0.28ms | **9.2x** |
| 128 | 4.21ms | 0.89ms | 0.33ms | **12.9x** |

*TensorRT gives the best results due to kernel fusion optimizations.*

To enable GPU:
```python
from silero_stress import load_accentor

# Use TensorRT (fastest, requires onnxruntime-gpu)
ru = load_accentor('ru', use_homosolver=True, device='tensorrt')

# Use CUDA (fallback if TensorRT unavailable)
ru = load_accentor('ru', use_homosolver=True, device='cuda')

# CPU (default)
ru = load_accentor('ru', use_homosolver=True, device='cpu')
```

### Why N-gram MLP Stays on CPU

The n-gram stress classifier has a tiny 16-dim input and only ~19K parameters. GPU kernel launch overhead dominates at this scale:

| Batch | NumPy (CPU) | PyTorch CUDA | Winner |
|-------|-------------|--------------|--------|
| 1 | 18µs | 275µs | **NumPy 15x** |
| 10 | 27µs | 276µs | **NumPy 10x** |
| 100 | 122µs | 286µs | **NumPy 2.3x** |

*Benchmarks after 100 warmup iterations, median of 1000 runs.*

GPU kernel launch overhead is ~240µs per call, which is **13x larger** than the actual NumPy computation (18µs). Even at batch size 100, NumPy wins because:

1. **Fixed overhead**: GPU kernel launch takes ~240µs regardless of computation size
2. **Tiny model**: The 4-layer MLP (16→64→128→64→10) finishes in microseconds on CPU
3. **No data transfer benefit**: Copying 16 floats to GPU costs more than computing locally

This validates the decision to use pure NumPy for n-gram classifiers while only using GPU for BERT.

## Benchmarks

### Running Benchmarks

```bash
# Run benchmark comparing PyTorch vs NumPy backends
PYTHONPATH=src python scripts/benchmark.py

# NumPy only (no torch required)
PYTHONPATH=src python scripts/benchmark.py --numpy-only

# Single language
PYTHONPATH=src python scripts/benchmark.py --lang ru
```

The benchmark:
1. Loads both backends with homograph solver enabled for Russian
2. Runs 50 warmup iterations (ensures JIT compilation for PyTorch)
3. Benchmarks 500 iterations per language
4. Reports per-sentence latency, load times, and percentiles

### Test Setup

- **CPU**: Intel Xeon E5-2680 v4 (14C/28T)
- **GPU**: 2x RTX 3090, 1x RTX A4000
- **OS**: Ubuntu 24.04 (Linux 6.8)
- **Python**: 3.12.3
- **NumPy**: 2.4.2
- **PyTorch**: 2.9.1+cu130 (single thread, `torch.set_num_threads(1)`)
- **ONNX Runtime GPU**: 1.24.2

### Results

#### Per-Sentence Latency (after warmup)

| Lang | PyTorch | NumPy | Speedup |
|------|---------|-------|---------|
| **RU** | 4.35ms | 2.95ms | **1.5x** |
| **UKR** | 0.95ms | 0.20ms | **4.7x** |
| **BEL** | 1.00ms | 0.20ms | **4.9x** |

*Russian is slower due to BERT homograph solver overhead (ONNX). Without homosolver, Russian matches UKR/BEL speeds.*

#### Load Time

| Lang | PyTorch | NumPy | Speedup |
|------|---------|-------|---------|
| **RU** | 738ms | 942ms | 0.8x |
| **UKR** | 77ms | 80ms | 1.0x |
| **BEL** | 108ms | 71ms | 1.5x |

*Russian load time is dominated by ONNX session creation for BERT. PyTorch wins here due to lazy loading.*

#### Detailed Metrics (Russian)

```
[PyTorch backend]
  Load time:       748.1ms
  Per batch:       13.63ms ± 0.37ms  (5 sentences)
  Per sentence:    2.727ms
  P50/P95/P99:     13.49 / 14.38 / 14.65ms
  Min/Max:         13.07 / 15.13ms

[NumPy backend]
  Load time:       957.5ms
  Per batch:        5.59ms ± 0.11ms  (5 sentences)
  Per sentence:    1.119ms
  P50/P95/P99:     5.57 / 5.75 / 5.94ms
  Min/Max:         5.50 / 7.30ms
```

## Limitations

None significant. The architecture is optimized for the workload:
- N-gram MLP: CPU-bound, NumPy is optimal
- BERT homograph solver: GPU-accelerated via ONNX Runtime when available

## Dependencies

### Minimum (Pure NumPy)
```
numpy>=1.20
```

### With Homograph Solver (CPU)
```
numpy>=1.20
onnxruntime>=1.10
```

### With Homograph Solver (GPU)
```
numpy>=1.20
onnxruntime-gpu>=1.10
```

Install GPU version with CUDA 12 support:
```bash
pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
```

### For Re-export
```
torch>=2.0
onnx>=1.12
transformers>=4.30
```
