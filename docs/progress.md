# Parity Investigation - Progress

## Task
Achieve 100% parity between the original PyTorch silero-stress 1.4 model and our reimplementations.

## Current Status

### Torch Wrapper: 100% Parity

Our `_torch_impl.py` wrapper matches original 1.4 exactly:

| Test | Result |
|------|--------|
| All 1,924 homographs | 1924/1924 (100%) |
| All 164 yo-homographs | 164/164 (100%) |
| 36 Russian sentences | 36/36 (100%) |
| 8 Ukrainian sentences | 8/8 (100%) |
| 7 Belarusian sentences | 7/7 (100%) |

**Total: 2,139/2,139 test cases pass**

See [TORCH_PARITY.md](TORCH_PARITY.md) for testing details.

### NumPy/ONNX Backend: Correct Output (fixes torch bugs)

Our `_impl.py` NumPy/ONNX implementation produces **correct output** for all cases:

| Test | Result |
|------|--------|
| Russian simple sentences | 100% |
| Russian yo sentences | 100% |
| Russian homograph sentences | 100% |
| All 1,924 Russian homographs | 1917/1924 match torch, 7 are torch bugs* |
| Ukrainian sentences | 100% |
| Belarusian sentences | 100% |
| Valid homodict variants | **1924/1924 (100%)** |

*7 cases where torch produces invalid forms (ё in unstressed position). NumPy produces the correct output for all cases. See [Known Torch Bugs](#known-torch-bugs) section.

See `tests/test_torch_wrapper_parity.py` and `tests/test_yo_parity.py` for test implementation.

---

## Processing Pipeline

### Full Accentor Pipeline (Russian)

```
Input: "Это стоит дорого"
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: HOMOSOLVER (if use_homosolver=True)                 │
│                                                             │
│  1a. Find homographs using regex pattern                    │
│      Pattern: (?=.*[а-яёА-ЯЁ])[а-яёА-ЯЁ+]+                  │
│      Found: "стоит" (in homodict)                           │
│                                                             │
│  1b. Mark homographs                                        │
│      "Это [HOMO] стоит [/HOMO] дорого"                      │
│                                                             │
│  1c. Tokenize with BERT tokenizer                           │
│      [CLS] это [HOMO] стоит [/HOMO] дорого [SEP]            │
│      → [2, 1234, 83828, 5678, 83829, 9012, 3]               │
│                                                             │
│  1d. Run BERT encoder                                       │
│      Input: token_ids [1, seq_len]                          │
│      Output: hidden_states [1, seq_len, 312]                │
│                                                             │
│  1e. Extract embeddings                                     │
│      marker_emb = hidden[homo_start_idx] → [312]            │
│      word_emb = mean(hidden[start+1:end]) → [312]           │
│      combined = concat([marker_emb, word_emb]) → [624]      │
│                                                             │
│  1f. Run classifier                                         │
│      logit = homo_clf(combined) → [1]                       │
│      pred = 1 if sigmoid(logit) > 0.5 else 0                │
│                                                             │
│  1g. Select variant                                         │
│      variants = sorted(homodict["стоит"])                   │
│      → ["сто+ит", "ст+оит"]                                 │
│      selected = variants[pred] → "сто+ит"                   │
│                                                             │
│  Output: "Это сто+ит дорого"                                │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: N-GRAM ACCENTOR                                     │
│                                                             │
│  For each word not already stressed:                        │
│                                                             │
│  2a. Check exceptions dictionary                            │
│      "дорого" not in exceptions                             │
│                                                             │
│  2b. Generate n-grams                                       │
│      word_ngrams("дорого") → ["<", "д", "о", ..., "дорого>"]│
│                                                             │
│  2c. Look up embedding indices                              │
│      indices = [ngram_dict.get(ng, 0) for ng in ngrams]     │
│                                                             │
│  2d. Compute word embedding                                 │
│      embedding = mean(emb_weights[indices]) → [16]          │
│                                                             │
│  2e. Run stress classifier                                  │
│      logits = stress_clf(embedding) → [10]                  │
│      stress_pos = argmax(logits)                            │
│                                                             │
│  2f. Apply stress mark                                      │
│      "дорого" → "д+орого"                                   │
│                                                             │
│  Output: "Это сто+ит д+орого" (with put_yo=False)           │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: YO REPLACEMENT (if put_yo=True)                     │
│                                                             │
│  For each word with "е" at stressed position:               │
│                                                             │
│  3a. Generate n-grams (same as stress)                      │
│                                                             │
│  3b. Compute word embedding                                 │
│                                                             │
│  3c. Run yo classifier                                      │
│      logits = yo_clf(embedding) → [7]                       │
│      yo_pred = argmax(logits)                               │
│                                                             │
│  3d. If yo_pred indicates ё at stressed position:           │
│      "е" → "ё"                                              │
│                                                             │
│  Output: "+Это сто+ит д+орого"                              │
└─────────────────────────────────────────────────────────────┘
```

### Ukrainian/Belarusian Pipeline (No Homosolver)

```
Input: "Привіт світ"
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ N-GRAM ACCENTOR ONLY                                        │
│                                                             │
│  For each word:                                             │
│  1. Check exceptions dictionary                             │
│  2. Generate n-grams with <word> markers                    │
│  3. Look up indices in ngram_dict                           │
│  4. Compute embedding = mean(emb_weights[indices])          │
│  5. Run stress_clf → get stress position                    │
│  6. Apply stress mark                                       │
│                                                             │
│  Output: "Прив+іт св+іт"                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Quantization / Dequantization

### Overview

| Component | Storage | At Load | At Inference |
|-----------|---------|---------|--------------|
| **RU BERT word_embeddings** | int8 | Dequantize | Use float32 |
| **RU N-gram embeddings** | float32 | Use as-is | Use float32 |
| **UKR N-gram embeddings** | int8 | Dequantize | Use float32 |
| **BEL N-gram embeddings** | int8 | Dequantize | Use float32 |

Dequantization formula: `float_weights = scale * (int8_weights - zero_point)`

### PyTorch (_torch_impl.py)

#### Russian (TorchAccentorRu)

```python
def __init__(self):
    # Load model
    importer = torch.package.PackageImporter(model_path)
    self._model = importer.load_pickle('accentor_models', 'accentor')

    # BERT embeddings: int8 → DEQUANTIZE
    bert = self._model.homosolver.model.bert
    qw = bert.embeddings.word_embeddings.weight.data.clone()
    rw = bert.scale * (qw - bert.zero_point)  # Dequantize!
    bert.embeddings.word_embeddings.weight.data = rw

    # N-gram embeddings: already float32, DO NOT touch
    # (The accentor.model.embedding is float32 in storage)
```

#### Ukrainian/Belarusian (TorchAccentorUkrBel)

```python
def __init__(self, lang):
    importer = torch.package.PackageImporter(model_path)
    self._model = importer.load_pickle('accentor_models', 'accentor')

    # N-gram embeddings: int8 → DEQUANTIZE
    ngram_model = self._model.accentor.model
    qw = ngram_model.embedding.weight.data.clone()
    rw = ngram_model.scale * (qw - ngram_model.zero_point)  # Dequantize!
    ngram_model.embedding.weight.data = rw
```

### NumPy/ONNX (_impl.py)

#### Russian (AccentorRu)

```python
def __init__(self):
    # N-gram embeddings: stored as float32, use directly
    emb_data = _load_gzip_npz('ru', 'embedding_ru.npz.gz')
    self.emb_weights = emb_data['weights']  # Already float32

    # BERT: uses separate ONNX model (bert_ru.onnx.gz)
    # The ONNX model has DEQUANTIZED weights embedded in it
```

#### Ukrainian/Belarusian (AccentorUkrBel)

```python
def __init__(self, lang):
    # N-gram embeddings: stored as int8, DEQUANTIZE at load
    emb_data = _load_gzip_npz(lang, f'embedding_{lang}.npz.gz')
    weights = emb_data['weights']       # int8
    scale = emb_data['scale'][0]        # float32
    zero_point = emb_data['zero_point'][0]  # float32

    self.emb_weights = scale * (weights.astype(np.float32) - zero_point)
```

---

## Export Scripts

### Russian Export (scripts/export_ru.py + scripts/export_bert_direct.py)

```
┌─────────────────────────────────────────────────────────────┐
│ export_ru.py: N-gram component                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Load raw model (before any dequantization)              │
│                                                             │
│  2. Extract N-gram embeddings                               │
│     → embeddings are ALREADY float32 in RU model            │
│     → save as-is to embedding_ru.npz                        │
│                                                             │
│  3. Export stress_clf to ONNX                               │
│     → stress_clf_ru.onnx                                    │
│                                                             │
│  4. Export yo_clf to ONNX                                   │
│     → yo_clf_ru.onnx                                        │
│                                                             │
│  5. Export dictionaries                                     │
│     → ngram_dict_ru.json, exceptions_ru.json                │
│     → homodict_ru.json, yohomodict_ru.json                  │
│     → vocab_ru.json, special_tokens_ru.json                 │
│                                                             │
│  6. Gzip all files                                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ export_bert_direct.py: BERT component                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Load raw model                                          │
│                                                             │
│  2. DEQUANTIZE BERT word embeddings in place                │
│     bert = acc.homosolver.model.bert                        │
│     qw = bert.embeddings.word_embeddings.weight.data        │
│     bert.embeddings.word_embeddings.weight.data =           │
│         bert.scale * (qw.float() - bert.zero_point)         │
│                                                             │
│  3. Create transformers BertModel with same config          │
│                                                             │
│  4. Copy DEQUANTIZED weights to transformers model          │
│     (weights are now float32 after step 2)                  │
│                                                             │
│  5. Export transformers model to ONNX                       │
│     → bert_ru.onnx (contains dequantized float32 weights)   │
│                                                             │
│  6. Export homo_clf to ONNX                                 │
│     → homo_clf_ru.onnx                                      │
│                                                             │
│  7. Merge external data, gzip                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Ukrainian/Belarusian Export (scripts/extract_and_export.py)

```
┌─────────────────────────────────────────────────────────────┐
│ extract_and_export.py                                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Load RAW model (BEFORE dequantization)                  │
│     raw_accentor = load_pickle(...)                         │
│     raw_model = raw_accentor.accentor.model                 │
│                                                             │
│  2. Extract int8 embeddings IMMEDIATELY                     │
│     weights = raw_model.embedding.weight.data.numpy()       │
│     scale = float(raw_model.scale)                          │
│     zero_point = float(raw_model.zero_point)                │
│                                                             │
│     → Save as embedding_{lang}.npz with int8 weights        │
│       (dequantization happens at load time in _impl.py)     │
│                                                             │
│  3. Load model WITH dequantization for testing              │
│     accentor = load_accentor(lang)  # This dequantizes      │
│                                                             │
│  4. Extract other components (already dequantized ok)       │
│     → ngram_dict_{lang}.json                                │
│     → exceptions_{lang}.json                                │
│                                                             │
│  5. Export stress_clf to ONNX                               │
│     → stress_clf_{lang}.onnx                                │
│                                                             │
│  6. Gzip all files                                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Critical Export Order

The order of operations matters:

```
WRONG (what caused bugs):
1. Load model → dequantizes embeddings in place
2. Extract weights → gets float32 (already dequantized)
3. Save weights with scale/zero_point
4. _impl.py loads and dequantizes AGAIN → garbage values

CORRECT (current implementation):
1. Load RAW model
2. Extract int8 weights IMMEDIATELY
3. Save weights with scale/zero_point
4. Then load with dequantization for testing
5. _impl.py loads and dequantizes ONCE → correct values
```

---

## Testing Commands

### NumPy/ONNX vs Torch Wrapper

```bash
# Full test suite (all languages)
PYTHONPATH=src python -m pytest tests/test_torch_wrapper_parity.py -v

# Just Russian
PYTHONPATH=src python -m pytest tests/test_torch_wrapper_parity.py::TestRussianAccentorParity -v
```

### Final Parity Test (all implementations)

```bash
# Requires two venvs:
# - .venv: silero-stress 1.4 + torch
# - .onnx_venv: numpy + onnxruntime (no silero-stress)
./scripts/run_final_parity.sh
```

### Export Scripts

```bash
# Export all
./scripts/export_all.sh

# Or individually:
PYTHONPATH=src python scripts/extract_and_export.py   # UKR/BEL
PYTHONPATH=src python scripts/export_ru.py            # RU n-gram
PYTHONPATH=src python scripts/export_bert_direct.py --all  # RU BERT
```

---

## Known Torch Bugs

### Yo-Replacement Bugs (Russian)

The torch implementation has 7 bugs where it adds ё to unstressed vowels, violating Russian phonology rules (ё MUST be stressed). NumPy produces correct output (valid homodict variants) for all cases.

| Word | Torch Output | NumPy Output | Correct | Issue |
|------|--------------|--------------|---------|-------|
| отменен | отм+енён | отм+енен | отм+енен | Torch adds ё to unstressed е |
| познаем | позн+аём | позн+аем | позн+аем | Torch adds ё to unstressed е |
| познаешь | позн+аёшь | позн+аешь | позн+аешь | Torch adds ё to unstressed е |
| покорен | пок+орён | пок+орен | пок+орен | Torch adds ё to unstressed е |
| запоем | зап+оём | зап+оем | зап+оем | Torch adds ё to unstressed е |
| мужичье | муж+ичьё | муж+ичье | муж+ичье | Torch adds ё to unstressed е |
| напоен | нап+оён | нап+оен | нап+оен | Torch adds ё to unstressed е |

**Root cause:** When torch processes homograph words (with pre-existing stress markers from the homosolver), it incorrectly applies the n-gram yo classifier, adding ё even when the stressed vowel is not 'е'.

**NumPy fix:** For pre-stressed words, check if the actual stressed vowel (after +) is 'е' before considering yo replacement. The ё letter must always be stressed in Russian.

These are documented in `tests/test_torch_wrapper_parity.py` as `known_torch_yo_bugs`.

---

## Files

### Source Files

| File | Description |
|------|-------------|
| `src/silero_stress/_torch_impl.py` | Torch wrapper (100% parity with 1.4) |
| `src/silero_stress/_impl.py` | NumPy/ONNX implementation (correct ё handling, fixes 7 torch bugs) |

### Test Files

| File | Description |
|------|-------------|
| `tests/test_torch_parity.py` | Torch wrapper vs original 1.4 |
| `tests/test_torch_wrapper_parity.py` | NumPy/ONNX vs torch wrapper (documents torch bugs) |
| `tests/test_yo_parity.py` | Dedicated ё replacement tests and correctness validation |

### Export Scripts

| Script | Description | Output Files |
|--------|-------------|--------------|
| `scripts/export_all.sh` | Master script - exports all languages | — |
| `scripts/extract_and_export.py` | UKR/BEL export | `embedding_{lang}.npz.gz`, `stress_clf_{lang}_weights.npz.gz`, `ngram_dict_{lang}.json.gz`, `exceptions_{lang}.json.gz` |
| `scripts/export_ru.py` | RU n-gram export | `embedding_ru.npz.gz`, `stress_clf_ru_weights.npz.gz`, `yo_clf_ru_weights.npz.gz`, `ngram_dict_ru.json.gz`, `exceptions_ru.json.gz`, `homodict_ru.json.gz`, `yohomodict_ru.json.gz`, `vocab_ru.json.gz`, `special_tokens_ru.json.gz` |
| `scripts/export_bert_direct.py` | RU BERT export | `bert_ru.onnx.gz`, `homo_clf_ru.onnx.gz` |

### Exported Data Files (per language)

**Russian (`data/ru/`):**
- ONNX: `bert_ru.onnx.gz` (30MB), `homo_clf_ru.onnx.gz` (581KB) - BERT homograph solver
- NPZ: `embedding_ru.npz.gz`, `stress_clf_ru_weights.npz.gz`, `yo_clf_ru_weights.npz.gz` - N-gram MLP
- JSON: `ngram_dict_ru.json.gz`, `exceptions_ru.json.gz`, `homodict_ru.json.gz`, `yohomodict_ru.json.gz`, `vocab_ru.json.gz`, `special_tokens_ru.json.gz`

**Ukrainian (`data/ukr/`):**
- NPZ: `embedding_ukr.npz.gz`, `stress_clf_ukr_weights.npz.gz`
- JSON: `ngram_dict_ukr.json.gz`, `exceptions_ukr.json.gz`

**Belarusian (`data/bel/`):**
- NPZ: `embedding_bel.npz.gz`, `stress_clf_bel_weights.npz.gz`
- JSON: `ngram_dict_bel.json.gz`, `exceptions_bel.json.gz`

---

## Future Improvements

### Auto-Batching for Postprocessing

Current implementation processes one sentence at a time. For postprocessing large text corpora, auto-batching could improve throughput.

**Proposed API:**

```python
# Current: one sentence at a time
results = [acc(sent) for sent in sentences]

# Future: batch processing
results = acc.batch(sentences)  # or acc(sentences) with auto-detection
```

**Pipeline batching:**

```
Input: ["Sentence 1", "Sentence 2", ..., "Sentence N"]
                    │
                    ▼
┌───────────────────────────────────────────────┐
│ 1. TOKENIZE ALL                               │
│    Split all sentences into words             │
│    → words[], sentence_indices[]              │
└───────────────────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────┐
│ 2. BATCH N-GRAM MLP                           │
│    Compute embeddings for all words at once   │
│    embeddings = emb_weights[ngram_indices]    │
│    logits = stress_clf(embeddings)  # batched │
│    → stress_positions[]                       │
└───────────────────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────┐
│ 3. BATCH BERT (homographs only)               │
│    Collect all homograph contexts             │
│    Pad to max_seq_len, create attention_mask  │
│    hidden = bert(padded_ids, attention_mask)  │
│    preds = homo_clf(hidden)  # batched        │
│    → homograph_variants[]                     │
└───────────────────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────┐
│ 4. REASSEMBLE                                 │
│    Apply stress marks and homograph variants  │
│    → ["Stressed 1", "Stressed 2", ...]        │
└───────────────────────────────────────────────┘
```

**Required changes:**

1. **N-gram MLP batching** (NumPy, stays on CPU):
   ```python
   # Current: one word at a time
   for word in words:
       emb = emb_weights[ngram_indices].mean(axis=0)
       logits = stress_clf(emb)

   # Batched: all words at once
   all_embeddings = np.stack([
       emb_weights[get_ngram_indices(w)].mean(axis=0) for w in words
   ])
   all_logits = stress_clf_batch(all_embeddings)  # (N, 10)
   ```

2. **BERT batching** (ONNX, GPU-accelerated):
   - Re-export with dynamic axes for batch dimension
   - Add `attention_mask` input for padding
   ```python
   torch.onnx.export(
       model, dummy_input, "bert_ru.onnx",
       input_names=["input_ids", "attention_mask"],
       dynamic_axes={
           "input_ids": {0: "batch", 1: "seq"},
           "attention_mask": {0: "batch", 1: "seq"},
           "last_hidden_state": {0: "batch", 1: "seq"},
       },
   )
   ```

3. **Classifier batching**:
   - Re-export `stress_clf`, `yo_clf`, `homo_clf` with dynamic batch axis
   - Or keep as NumPy matmul (already supports batching)

**Performance expectation:**

| Scenario | Current | Batched | Speedup |
|----------|---------|---------|---------|
| 1000 sentences (no homographs) | ~200ms | ~50ms | ~4x |
| 1000 sentences (with BERT) | ~3000ms | ~500ms | ~6x |

*Estimates based on reducing Python loop overhead and enabling GPU parallelism for BERT.*

**Note:** N-gram MLP benefits from batching even on CPU due to reduced Python overhead, but GPU won't help (kernel launch overhead dominates at this scale).
