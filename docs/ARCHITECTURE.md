# Silero Stress Architecture

Reverse-engineered from TorchScript code extracted from `accentor.pt`.

**Verified:** 100% parity with original PyTorch implementation (1924/1924 homographs).

## Table of Contents

- [Overview](#overview)
- [HomoSolver Pipeline](#homosolver-pipeline)
- [Model Architecture: HomoSolverClean](#model-architecture-homosolverclean)
- [BERT Model: BertModelJIT](#bert-model-bertmodeljit)
- [Tokenizer: SimpleBertTokenizer](#tokenizer-simpleberttokenizer)
- [Weight Shapes Summary](#weight-shapes-summary)
- [Weight Extraction & Export](#weight-extraction--export)
- [ONNX Implementation Notes](#onnx-implementation-notes)
- [Known Limitations & Edge Cases](#known-limitations--edge-cases)

## Overview

The system has two main components:
1. **AccentorNgram** - N-gram based stress/yo prediction
2. **HomoSolver** - BERT-based homograph disambiguation

## HomoSolver Pipeline

```
Input sentence
     ↓
[1] Find homographs using regex pattern: r'(?=.*[а-яё])[а-яё+]+' (IGNORECASE)
    - Must contain at least one Cyrillic letter (lookahead)
    - Can include '+' character (for already-stressed words)
     ↓
[2] For each homograph, create marked text:
    "word" → "... [HOMO] word [/HOMO] ..."
    Note: Spaces around markers are important!
     ↓
[3] Tokenize with SimpleBertTokenizer:
    - BasicTokenizer: split on whitespace, handle punctuation, preserve [HOMO]/[/HOMO]
    - WordpieceTokenizer: subword tokenization using vocab
    - Add [CLS] at start, [SEP] at end
     ↓
[4] Batch all homographs and run HomoSolverClean model
     ↓
[5] Select variant from SORTED(homodict[word])[prediction]
     ↓
[6] Post-process:
    - Apply case from original word to predicted word
    - Add stress mark if (n_vowels > 1 OR stress_single_vowel) AND put_stress
    - Replace ё→е if put_yo=False
```

## Model Architecture: HomoSolverClean

```python
class HomoSolverClean:
    bert: BertModelJIT
    homo_clf: Sequential(Linear(624,256), ReLU, Dropout, Linear(256,1))

    def forward(input_ids, homo_start_ids, homo_end_ids):
        # 1. Run BERT
        hidden = bert(input_ids)[0]  # Shape: [batch, seq_len, 312]

        # 2. For each homograph in batch:
        for i in range(len(homo_start_ids)):
            # marker_emb = hidden at [HOMO] position
            marker_emb = hidden[i, homo_start_ids[i], :]  # [312]

            # word_emb = MEAN of tokens between [HOMO] and [/HOMO]
            start = homo_start_ids[i] + 1
            end = homo_end_ids[i]
            word_emb = hidden[i, start:end, :].mean(dim=0)  # [312]

            # Concatenate: [624]
            combined = cat([marker_emb, word_emb])

        # 3. Run classifier
        logits = homo_clf(stack(combined_list))  # [batch, 1]
        return logits
```

## BERT Model: BertModelJIT

### Configuration

| Parameter | Value |
|-----------|-------|
| Hidden size | 312 |
| Attention heads | 12 |
| Head size | 312 / 12 = 26 |
| Encoder layers | 3 |
| Intermediate size | 600 |
| Vocab size | 83,830 |
| Max positions | 2,048 |
| Type vocab size | 2 |

### Buffers (registered, not parameters)

- `embeddings.position_ids`: [1, 2048] int64
- `embeddings.token_type_ids`: [1, 2048] int64
- `scale`: Tensor (for int8 dequantization - NOT USED in forward)
- `zero_point`: Tensor (for int8 dequantization - NOT USED in forward)

### Forward Pass

```python
def forward(input_ids, attention_mask=None, token_type_ids=None, ...):
    # 1. Handle attention_mask default
    if attention_mask is None:
        attention_mask = ones([batch, seq_len])

    # 2. Handle token_type_ids default (from buffer)
    if token_type_ids is None:
        token_type_ids = self.embeddings.token_type_ids[:, :seq_len].expand(batch, seq_len)

    # 3. Create extended attention mask
    extended_mask = get_extended_attention_mask(attention_mask, input_shape)
    # Shape: [batch, 1, 1, seq_len], values: 0 or dtype_min (-1e9)

    # 4. Embeddings
    embedding_output = embeddings(input_ids, token_type_ids, position_ids)

    # 5. Encoder (3 layers)
    encoder_output = encoder(embedding_output, extended_mask)

    # 6. Pooler (not used for homographs)
    pooled = pooler(encoder_output)

    return (encoder_output, pooled)
```

## BertEmbeddings

```python
class BertEmbeddings:
    word_embeddings: Embedding(83830, 312)      # int8 stored!
    position_embeddings: Embedding(2048, 312)   # float32
    token_type_embeddings: Embedding(2, 312)    # float32
    LayerNorm: LayerNorm(312, eps=1e-12)
    dropout: Dropout(0.1)
    position_embedding_type: "absolute"

    def forward(input_ids, token_type_ids, position_ids):
        word_emb = word_embeddings(input_ids)
        pos_emb = position_embeddings(position_ids)  # Defaults to [0, 1, 2, ...]
        type_emb = token_type_embeddings(token_type_ids)  # Defaults to zeros

        embeddings = word_emb + pos_emb + type_emb
        embeddings = LayerNorm(embeddings)
        embeddings = dropout(embeddings)
        return embeddings
```

### Critical: int8 Word Embeddings

The `word_embeddings.weight` is stored as **int8** and must be **DEQUANTIZED** before use.

```python
# Storage: int8 values like [-2, -2, -5, 127, -2]
# Dequantization: scale * (int8 - zero_point)
#   scale = 0.01497, zero_point = 1.1597
#   → [-0.047, -0.047, -0.092, 1.884, -0.047]

# This happens at MODEL LOAD TIME, not during forward pass
# After dequantization, weights are float32 and used normally
```

**Important:** The original PyTorch model (silero-stress 1.4) applies this dequantization
implicitly. Our torch wrapper (`_torch_impl.py`) and ONNX export must do it explicitly.

## BertEncoder (3 layers)

```python
class BertEncoder:
    layer: ModuleList([BertLayer, BertLayer, BertLayer])

    def forward(hidden, attention_mask):
        for layer in self.layer:
            hidden = layer(hidden, attention_mask)[0]
        return (hidden,)
```

## BertLayer

```python
class BertLayer:
    attention: BertAttention
    intermediate: BertIntermediate
    output: BertOutput

    def forward(hidden, attention_mask):
        attention_output = attention(hidden, attention_mask)[0]
        layer_output = feed_forward_chunk(attention_output)
        return (layer_output,)

    def feed_forward_chunk(attention_output):
        intermediate_output = intermediate(attention_output)
        layer_output = output(intermediate_output, attention_output)
        return layer_output
```

## BertAttention

### BertSelfAttention

```python
class BertSelfAttention:
    num_attention_heads: 12
    attention_head_size: 26
    all_head_size: 312
    query: Linear(312, 312)
    key: Linear(312, 312)
    value: Linear(312, 312)
    dropout: Dropout(0.1)

    def forward(hidden, attention_mask):
        Q = query(hidden)  # [batch, seq, 312]
        K = key(hidden)
        V = value(hidden)

        # Reshape for multi-head: [batch, heads, seq, head_size]
        Q = Q.view(batch, seq, 12, 26).permute(0, 2, 1, 3)
        K = K.view(batch, seq, 12, 26).permute(0, 2, 1, 3)
        V = V.view(batch, seq, 12, 26).permute(0, 2, 1, 3)

        # Attention scores
        scores = (Q @ K.transpose(-1, -2)) / sqrt(26)

        # Apply attention mask (add -inf for padding)
        if attention_mask is not None:
            scores = scores + attention_mask

        # Softmax and apply
        probs = softmax(scores, dim=-1)
        probs = dropout(probs)
        context = probs @ V

        # Reshape back: [batch, seq, 312]
        context = context.permute(0, 2, 1, 3).reshape(batch, seq, 312)
        return (context,)
```

### BertSelfOutput

```python
class BertSelfOutput:
    dense: Linear(312, 312)
    LayerNorm: LayerNorm(312)
    dropout: Dropout(0.1)

    def forward(hidden, input_tensor):
        hidden = dense(hidden)
        hidden = dropout(hidden)
        return LayerNorm(hidden + input_tensor)  # Residual connection
```

## BertIntermediate + BertOutput (FFN)

### BertIntermediate

```python
class BertIntermediate:
    dense: Linear(312, 600)
    intermediate_act_fn: GELUActivation  # torch.gelu

    def forward(hidden):
        return gelu(dense(hidden))
```

### BertOutput

```python
class BertOutput:
    dense: Linear(600, 312)
    LayerNorm: LayerNorm(312)
    dropout: Dropout(0.1)

    def forward(hidden, input_tensor):
        hidden = dense(hidden)
        hidden = dropout(hidden)
        return LayerNorm(hidden + input_tensor)  # Residual connection
```

## BertPooler

```python
class BertPooler:
    dense: Linear(312, 312)
    activation: Tanh

    def forward(hidden):
        first_token = hidden[:, 0, :]  # [CLS] token
        pooled = tanh(dense(first_token))
        return pooled
```

## Tokenizer: SimpleBertTokenizer

Two-stage tokenization matching HuggingFace BERT tokenizer.

### Stage 1: BasicTokenizer

```python
class BasicTokenizer:
    never_split = {'[HOMO]', '[/HOMO]', '[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]'}

    def tokenize(text):
        text = _clean_text(text)           # Remove control chars (Unicode cat "C")
        text = _tokenize_chinese_chars(text)  # Add spaces around CJK
        tokens = text.strip().split()      # Whitespace split

        split_tokens = []
        for token in tokens:
            if token in never_split:
                split_tokens.append(token)
            else:
                split_tokens.extend(_run_split_on_punc(token))

        return " ".join(split_tokens).split()

    def _is_punctuation(char):
        # ASCII punctuation ranges: 33-47, 58-64, 91-96, 123-126
        # OR Unicode category starts with "P"
        cp = ord(char)
        if (33 <= cp <= 47) or (58 <= cp <= 64) or \
           (91 <= cp <= 96) or (123 <= cp <= 126):
            return True
        return unicodedata.category(char).startswith("P")

    def _is_whitespace(char):
        if char in " \t\n\r":
            return True
        return unicodedata.category(char) == "Zs"

    def _is_control(char):
        if char in "\t\n\r":
            return False
        return unicodedata.category(char).startswith("C")
```

### Stage 2: WordpieceTokenizer

```python
class WordpieceTokenizer:
    max_input_chars_per_word = 100
    unk_token = "[UNK]"

    def tokenize(text):
        output_tokens = []
        for token in text.split():
            if len(token) > max_input_chars_per_word:
                output_tokens.append(unk_token)
                continue

            # Greedy longest-match-first
            start = 0
            sub_tokens = []
            while start < len(token):
                end = len(token)
                found = None
                while start < end:
                    substr = token[start:end]
                    if start > 0:
                        substr = "##" + substr
                    if substr in vocab:
                        found = substr
                        break
                    end -= 1

                if found is None:
                    return [unk_token]  # Whole word unknown

                sub_tokens.append(found)
                start = end

            output_tokens.extend(sub_tokens)

        return output_tokens
```

### Special Token IDs

| Token | ID |
|-------|-----|
| `[PAD]` | 0 |
| `[UNK]` | 1 |
| `[CLS]` | 2 |
| `[SEP]` | 3 |
| `[MASK]` | 4 |
| `[HOMO]` | 83828 |
| `[/HOMO]` | 83829 |

### Encoding

```python
def encode(text):
    tokens = basic_tokenizer.tokenize(text)
    tokens = wordpiece_tokenizer.tokenize(" ".join(tokens))
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    ids = [vocab[t] for t in tokens]
    return ids
```

## Weight Shapes Summary

### Embeddings
```
embeddings.word_embeddings.weight:      [83830, 312]  int8 → float32
embeddings.position_embeddings.weight:  [2048, 312]   float32
embeddings.token_type_embeddings.weight: [2, 312]     float32
embeddings.LayerNorm.weight:            [312]         float32
embeddings.LayerNorm.bias:              [312]         float32
```

### Encoder Layer (x3)
```
layer.{i}.attention.self.query.weight:    [312, 312]
layer.{i}.attention.self.query.bias:      [312]
layer.{i}.attention.self.key.weight:      [312, 312]
layer.{i}.attention.self.key.bias:        [312]
layer.{i}.attention.self.value.weight:    [312, 312]
layer.{i}.attention.self.value.bias:      [312]
layer.{i}.attention.output.dense.weight:  [312, 312]
layer.{i}.attention.output.dense.bias:    [312]
layer.{i}.attention.output.LayerNorm.weight: [312]
layer.{i}.attention.output.LayerNorm.bias:   [312]
layer.{i}.intermediate.dense.weight:      [600, 312]
layer.{i}.intermediate.dense.bias:        [600]
layer.{i}.output.dense.weight:            [312, 600]
layer.{i}.output.dense.bias:              [312]
layer.{i}.output.LayerNorm.weight:        [312]
layer.{i}.output.LayerNorm.bias:          [312]
```

### Pooler
```
pooler.dense.weight: [312, 312]
pooler.dense.bias:   [312]
```

### Classifier
```
homo_clf.0.weight: [256, 624]   # Linear
homo_clf.0.bias:   [256]
homo_clf.3.weight: [1, 256]     # Linear
homo_clf.3.bias:   [1]
```

## Source Files

Architecture was reverse-engineered from TorchScript code in `accentor.pt`.

Key components extracted:
- `BertModelJIT` - BERT encoder
- `HomoSolverClean` - Homograph classifier wrapper
- `BasicTokenizer`, `WordpieceTokenizer` - Tokenization
- `NewFastTextEmbeddingBag` - N-gram embeddings
- `BertEmbeddings`, `BertEncoder`, `BertLayer` - BERT layers

---

## Weight Extraction & Export

### Loading the TorchScript Model

```python
import torch

# Load TorchScript model
importer = torch.package.PackageImporter('src/silero_stress/data/accentor.pt')
acc = importer.load_pickle('accentor_models', 'accentor')
bert = acc.homosolver.model.bert
clf = acc.homosolver.model.homo_clf
```

### Export to ONNX

```python
from transformers import BertConfig, BertModel

# 1. Load model and DEQUANTIZE embeddings in place
acc = load_model()
bert = acc.homosolver.model.bert

scale = float(bert.scale)
zero_point = float(bert.zero_point)
qw = bert.embeddings.word_embeddings.weight.data.clone()
bert.embeddings.word_embeddings.weight.data = scale * (qw.float() - zero_point)

# 2. Create transformers BERT with same config
config = BertConfig(
    vocab_size=83830, hidden_size=312, num_hidden_layers=3,
    num_attention_heads=12, intermediate_size=600,
    max_position_embeddings=2048, type_vocab_size=2,
    hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0
)
tf_bert = BertModel(config)

# 3. Copy DEQUANTIZED weights to transformers model
state_dict = {}
for name, param in bert.named_parameters():
    data = param.data
    if data.dtype == torch.int8:
        # Already dequantized in step 1, just cast any remaining int8
        data = scale * (data.float() - zero_point)
    state_dict[name] = data

tf_bert.load_state_dict(state_dict, strict=False)

# 4. Export to ONNX (weights are now float32)
torch.onnx.export(tf_bert, dummy_input, 'bert_ru.onnx', ...)
```

### Generated Files

| File | Size | Description |
|------|------|-------------|
| `bert_ru.onnx` | 112MB | BERT encoder (all weights internal) |
| `homo_clf_ru.onnx` | 627KB | Homograph classifier |

### Export Command

```bash
python scripts/export_bert_direct.py --all
```

---

## ONNX Implementation Notes

### Exported Models

Two ONNX models are required for the homograph solver:

1. **bert_ru.onnx** - BERT encoder only
   - Input: `input_ids` [batch, seq_len] int64
   - Output: `hidden_states` [batch, seq_len, 312] float32

2. **homo_clf_ru.onnx** - Classifier head
   - Input: `embedding` [batch, 624] float32
   - Output: `logits` [batch, 1] float32

### Critical Implementation Details

#### 1. Word Embeddings: Dequantization Required

The BERT word embeddings are stored as int8 and MUST be dequantized:

```python
# CORRECT: Dequantize int8 to float32
# This happens at LOAD TIME, not inference time
scale = 0.01497
zero_point = 1.1597
word_emb = scale * (word_embeddings_int8.astype(np.float32) - zero_point)

# For ONNX: The exported model already has dequantized float32 weights
# embedded in it, so no runtime dequantization needed

# WRONG: Direct cast without dequantization
word_emb = word_embeddings_int8.astype(np.float32)  # Will give wrong results!
```

**Note:** Russian N-gram embeddings are already float32 and should NOT be dequantized.
Ukrainian/Belarusian N-gram embeddings ARE int8 and need dequantization.

#### 2. Homograph Detection Pattern

```python
# Original pattern (case-insensitive)
pattern = re.compile(r'(?=.*[а-яё])[а-яё+]+', re.IGNORECASE)

# Equivalent without IGNORECASE flag
pattern = re.compile(r'(?=.*[а-яёА-ЯЁ])[а-яёА-ЯЁ+]+')
```

The lookahead `(?=.*[а-яё])` ensures at least one Cyrillic letter exists.

#### 3. Homograph Marking

Spaces around markers are critical for correct tokenization:

```python
# CORRECT
marked = text[:start] + ' [HOMO] ' + word + ' [/HOMO] ' + text[end:]

# WRONG (tokens won't be preserved)
marked = text[:start] + '[HOMO]' + word + '[/HOMO]' + text[end:]
```

#### 4. Classifier Output → Prediction

```python
logit = clf_session.run(None, {'embedding': combined})[0]
pred = 1 if 1 / (1 + np.exp(-logit[0, 0])) > 0.5 else 0  # sigmoid > 0.5

# Select variant (MUST be sorted!)
variants = sorted(homodict[word.lower()])
word_pred = variants[pred]
```

#### 5. Case Preservation

```python
# Apply original word's case to predicted word
word_pred_cased = ''.join(
    c2.lower() if c1.islower() else c2.upper()
    for c1, c2 in zip(original_word, predicted_word_no_stress)
)
```

#### 6. stress_single_vowel Parameter

Controls whether single-vowel words get stress marks:

```python
n_vowels = sum(c in 'аоуыэиеяёю' for c in word.lower())

# Add stress only if:
if (n_vowels > 1 or stress_single_vowel) and put_stress:
    word = word[:stress_idx] + '+' + word[stress_idx:]
```

### Parity Test Results

```
Tokenizer encoding:     100% match
BERT hidden states:     max_diff < 1e-4
Classifier outputs:     max_diff < 1e-5
Homograph predictions:  1924/1924 (100%)
Full accentor output:   100% match
```

### Files

```
src/silero_stress/
├── _impl.py              # Pure NumPy/ONNX implementation
├── accentor.py           # Public API (load_accentor)
└── data/ru/
    ├── bert_ru.onnx.gz       # BERT encoder
    ├── homo_clf_ru.onnx.gz   # Classifier
    ├── vocab_ru.json.gz      # 83,830 tokens
    ├── special_tokens_ru.json.gz
    ├── homodict_ru.json.gz   # 1,924 homographs
    └── yohomodict_ru.json.gz # ё-homographs
```

---

## Known Limitations & Edge Cases

Based on [Habr article](https://habr.com/ru/articles/955130/) and GitHub issues.

### Model Statistics

| Metric | Value |
|--------|-------|
| Active homographs | 1,924 (from 2,157 candidates) |
| Total variants | ~5,630 |
| Training sentences | 122 million |
| Median sentences/homograph | 11,000 |
| Median sentences/variant | 3,200 |
| Dictionary size | 4.1M word forms |

### Performance Metrics

| Model | Precision | Recall | F1 | Word Acc |
|-------|-----------|--------|-----|----------|
| silero-stress (public) | 0.84 | 0.90 | 0.85 | 0.92 |
| Naive baseline (most frequent) | - | - | - | 0.85 |

**Important:** Naive baseline achieves 85% by always selecting the most frequent variant.
This is why per-class metrics matter more than overall accuracy.

### Known Behaviors (Inherited from Original)

These behaviors exist in the original PyTorch implementation:

1. **"все" → "всё" ignores put_yo flag in homosolver context**
   ```python
   acc("все хорошо", put_yo=False)  # Returns "вс+ё хорош+о" (ё still appears!)
   ```
   The homosolver applies ё from homodict regardless of `put_yo_homo` in some cases.

2. **Capital letters may prevent ё replacement**
   ```python
   acc("Все идет", put_yo=True)   # Returns "Вс+е ид+ёт" (Все stays as е)
   acc("все идет", put_yo=True)   # Returns "вс+ё ид+ёт" (все → всё)
   ```
   Homodict contains lowercase entries; case-sensitive lookup affects ё.

3. **Rare homographs excluded** - Variants requiring >95% precision were removed
   to avoid brittleness. ~233 candidates were excluded.

### Verified Working

These were reported as issues but work correctly in v1.4:

| Feature | Status | Example |
|---------|--------|---------|
| Newline preservation | ✓ | `"Привет\nМир"` → `"Прив+ет\nМ+ир"` |
| Preset stress marks | ✓ | `"Прив+ет мир"` → `"Прив+ет м+ир"` |
| Percentage symbols | ✓ | `"100% готов"` → `"100% гот+ов"` |
| Adjacent word stress | ✓ | Preset stress doesn't break neighbors |
| Multiple homographs | ✓ | All processed in single sentence |
| Mixed case | ✓ | `"ЗАМОК"` → `"З+АМОК"` |

### GitHub Issues (Historical)

| Issue | Description | Status |
|-------|-------------|--------|
| #6 | Preset stress breaking adjacent words | Fixed |
| #5 | Stress marking errors | Fixed |
| #4 | `.to()` only accepts strings | Fixed |
| #3 | Missing model name in docs | Fixed |

### Performance

- **0.5ms per word** (single-threaded CPU with AVX2)
- **~30ms for 400-char text** with 2 homographs
- Package size: ~50MB (30MB compressed)

---

## Testing

### Running Tests

```bash
# Run parity tests
pytest tests/test_parity.py -v

# Run specific test class
pytest tests/test_parity.py::TestHomographPredictionParity -v
```

### Test Coverage

| Test Class | Coverage |
|------------|----------|
| `TestTokenizerParity` | Tokenizer encoding matches PyTorch |
| `TestBertOutputParity` | BERT hidden states < 1e-4 diff |
| `TestClassifierParity` | Classifier outputs < 1e-5 diff |
| `TestHomographPredictionParity` | All 1924 homographs |
| `TestFullAccentorParity` | End-to-end output match |
