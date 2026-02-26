# Pure NumPy Implementation for Silero-Stress

This document describes the PyTorch-free implementation of silero-stress models.

> **Technical Reference:** For detailed BERT architecture, tokenizer implementation, and weight extraction, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Overview

The original silero-stress library uses PyTorch (~2GB dependency) for inference. This implementation reduces the dependency to just NumPy (~25MB). ONNX Runtime (~17MB) is only needed for Russian homograph disambiguation (BERT).

### Performance Comparison

| Backend | 100 sentences (RU) | vs PyTorch CPU |
|---------|-------------------|----------------|
| PyTorch CPU | 460ms | 1.0x |
| PyTorch CUDA | 420ms | 1.1x |
| **ONNX CPU** | 232ms | **2.0x** |
| ONNX CUDA | 152ms | 3.0x |
| **ONNX TensorRT** | 119ms | **3.9x** |

| Metric | PyTorch | NumPy/ONNX | Improvement |
|--------|---------|------------|-------------|
| **Import memory** | ~2GB (torch) | ~25MB (numpy) | **80x smaller** |
| **Dependencies** | torch>=2.0 | numpy + onnxruntime | Much lighter |

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

### Tokenization

Sentences are split using: `r'([\s.,!?;:<>=()/\\]+)'`

#### Hyphenated Words (Russian)

Hyphenated words are split into parts, each processed separately:
- `что-то` → `['что-', 'то']`
- `кому-то` → `['кому-', 'то']`

**Special case**: The `-то` suffix (indefinite particle) is never stressed:
```
что-то  → чт+о-то   (stress on "что", not "то")
кому-то → ком+у-то  (stress on "кому", not "то")
```

### Language-Specific Details

| Language | Vowels | Alphabet Pattern |
|----------|--------|------------------|
| **RU** | `аоуыэиеяёю` | `[А-Яа-яёЁ]` |
| **UKR** | `аеєиіїоуюя` | `[а-яєіїґА-ЯЄІЇҐь']` |
| **BEL** | `аоуіэыяеёю` | `[а-яёіўА-ЯЁІЎь']` |

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
# Auto-detect all available backends and devices
PYTHONPATH=src python scripts/benchmark.py

# ONNX only (no torch required)
PYTHONPATH=src python scripts/benchmark.py --onnx-only

# PyTorch only
PYTHONPATH=src python scripts/benchmark.py --torch-only
```

The benchmark:
1. Auto-detects available ONNX providers (CPU, CUDA, TensorRT, ROCm, OpenVINO, CoreML, DirectML)
2. Auto-detects PyTorch CUDA availability
3. Runs 10 warmup + 50 benchmark iterations
4. Reports per-100-sentence throughput and single homograph-heavy sentence latency

### Test Setup

- **CPU**: Intel Xeon E5-2680 v4 (14C/28T)
- **GPU**: NVIDIA RTX 3090
- **OS**: Ubuntu 24.04 (Linux 6.8)
- **Python**: 3.12.3
- **NumPy**: 2.4.2
- **PyTorch**: 2.9.1+cu130 (single thread, `torch.set_num_threads(1)`)
- **ONNX Runtime GPU**: 1.24.2

### Results (Russian with Homograph Solver)

| Backend        | Load    | 100 sentences    | 1 sent (7 homo)  |
|----------------|---------|------------------|------------------|
| torch_cpu      |   672ms |  460.6ms (0.5x)  |  20.23ms (0.4x)  |
| torch_cuda     |   904ms |  419.5ms (0.6x)  |   8.89ms (0.9x)  |
| onnx_cpu       |   784ms |  231.5ms (1.0x)  |   8.24ms (1.0x)  |
| onnx_cuda      |   841ms |  151.9ms (1.5x)  |   3.04ms (2.7x)  |
| onnx_tensorrt  |  1924ms |  119.1ms (1.9x)  |   2.42ms (3.4x)  |

**Key findings:**
- ONNX CPU is **2.0x faster** than PyTorch CPU
- ONNX TensorRT is **1.9x faster** than ONNX CPU for throughput
- ONNX TensorRT is **3.4x faster** than ONNX CPU for homograph-heavy sentences
- TensorRT has higher load time due to engine compilation (cached after first run)

## Limitations

### Floating-Point Precision

Some words may show different stress predictions between NumPy and PyTorch backends due to floating-point precision differences:

```
замок (no context) → NumPy: "зам+ок" (lock), PyTorch: "з+амок" (castle)
```

This happens when model confidence is low and argmax can flip between predictions. Both outputs are valid interpretations. For exact PyTorch parity, use `backend='torch'`.

### Architecture Constraints

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
