"""
Pure NumPy implementation of silero-stress accentor.
NO PyTorch required! ONNX Runtime is optional (only for Russian BERT homograph solver).
"""

import re
import io
import gzip
import json
import unicodedata
import numpy as np

try:
    import importlib_resources as impresources
except ImportError:
    from importlib import resources as impresources

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    ort = None
    HAS_ONNX = False


def _get_data_path(lang, filename):
    """Get path to data file using importlib.resources."""
    package_path = f"silero_stress.data.{lang}"
    if hasattr(impresources, 'files'):
        return str(impresources.files(package_path).joinpath(filename))
    else:
        with impresources.path(package_path, filename) as f:
            return str(f)


def _load_gzip_json(lang, filename):
    """Load gzipped JSON file."""
    path = _get_data_path(lang, filename)
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return json.load(f)


def _load_npz(lang, filename):
    """Load NPZ file (uncompressed - NPZ uses internal zip compression)."""
    path = _get_data_path(lang, filename)
    return dict(np.load(path))


def _load_gzip_onnx(lang, filename):
    """Load gzipped ONNX model."""
    if not HAS_ONNX:
        raise ImportError("onnxruntime is required for BERT homograph solver")
    path = _get_data_path(lang, filename)
    with gzip.open(path, 'rb') as f:
        model_bytes = f.read()
    return model_bytes


def word_ngrams(word, min_n=1, max_n=None):
    """Generate character n-grams with boundary markers."""
    if max_n is None:
        max_n = len(word) + 3
    word_with_markers = f"<{word}>"
    ngrams = []
    for n in range(min_n, max_n + 1):
        for i in range(len(word_with_markers) - n + 1):
            ngrams.append(word_with_markers[i:i+n])
    return ngrams


def _get_onnx_providers(device='cpu'):
    """Get ONNX Runtime providers based on device preference."""
    if not HAS_ONNX:
        return []
    available = ort.get_available_providers()
    if device == 'tensorrt' and 'TensorrtExecutionProvider' in available:
        return ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
    elif device in ('cuda', 'tensorrt') and 'CUDAExecutionProvider' in available:
        return ['CUDAExecutionProvider', 'CPUExecutionProvider']
    return ['CPUExecutionProvider']


class BasicTokenizer:
    """Basic tokenizer matching original HuggingFace implementation."""

    def __init__(self, never_split=None):
        self.never_split = set(never_split) if never_split else set()

    def tokenize(self, text, never_split=None):
        """Tokenize text with punctuation splitting."""
        never_split = self.never_split.union(set(never_split or []))
        text = self._clean_text(text)
        text = self._tokenize_chinese_chars(text)

        orig_tokens = text.strip().split()
        split_tokens = []
        for token in orig_tokens:
            if token in never_split:
                split_tokens.append(token)
            else:
                split_tokens.extend(self._run_split_on_punc(token, never_split))

        return " ".join(split_tokens).split()

    def _clean_text(self, text):
        """Remove control characters and normalize whitespace."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or self._is_control(char):
                continue
            if self._is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_control(self, char):
        """Check if character is a control character."""
        if char in "\t\n\r":
            return False
        cat = unicodedata.category(char)
        return cat.startswith("C")

    def _is_whitespace(self, char):
        """Check if character is whitespace."""
        if char in " \t\n\r":
            return True
        return unicodedata.category(char) == "Zs"

    def _tokenize_chinese_chars(self, text):
        """Add whitespace around Chinese characters."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.extend([" ", char, " "])
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Check if character is Chinese."""
        return ((0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF) or
                (0x20000 <= cp <= 0x2A6DF) or (0x2A700 <= cp <= 0x2B73F) or
                (0x2B740 <= cp <= 0x2B81F) or (0x2B820 <= cp <= 0x2CEAF) or
                (0xF900 <= cp <= 0xFAFF) or (0x2F800 <= cp <= 0x2FA1F))

    def _run_split_on_punc(self, token, never_split=None):
        """Split token on punctuation."""
        if token in (never_split or set()):
            return [token]

        output = []
        start_new_word = True
        for char in token:
            if self._is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)

        return ["".join(x) for x in output]

    def _is_punctuation(self, char):
        """Check if character is punctuation."""
        cp = ord(char)
        if ((33 <= cp <= 47) or (58 <= cp <= 64) or
            (91 <= cp <= 96) or (123 <= cp <= 126)):
            return True
        return unicodedata.category(char).startswith("P")


class WordpieceTokenizer:
    """WordPiece tokenizer matching original implementation."""

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """Tokenize text using WordPiece algorithm."""
        output_tokens = []
        for token in text.split():
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1

                if cur_substr is None:
                    is_bad = True
                    break

                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)

        return output_tokens


class SimpleBertTokenizer:
    """BERT tokenizer with BasicTokenizer + WordPiece for Russian homograph solver."""

    def __init__(self, vocab, special_tokens):
        self.vocab = vocab
        self.ids_to_tokens = {v: k for k, v in vocab.items()}
        self.pad_token_id = special_tokens['pad_token_id']
        self.cls_token_id = special_tokens['cls_token_id']
        self.sep_token_id = special_tokens['sep_token_id']
        self.unk_token_id = special_tokens['unk_token_id']
        self.homo_start_id = special_tokens['homo_start_id']
        self.homo_end_id = special_tokens['homo_end_id']
        self.never_split = {'[HOMO]', '[/HOMO]', '[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]'}

        self.basic_tokenizer = BasicTokenizer(never_split=self.never_split)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=vocab, unk_token='[UNK]')

    def tokenize(self, text):
        """Tokenize text with BasicTokenizer + WordPiece."""
        tokens = self.basic_tokenizer.tokenize(text, never_split=self.never_split)
        wordpiece_tokens = self.wordpiece_tokenizer.tokenize(" ".join(tokens))
        return wordpiece_tokens

    def encode(self, text, add_special_tokens=True):
        """Encode text to token IDs."""
        tokens = self.tokenize(text)
        ids = [self.vocab.get(t, self.unk_token_id) for t in tokens]
        if add_special_tokens:
            ids = [self.cls_token_id] + ids + [self.sep_token_id]
        return ids


def _parse_device(device):
    """Parse device string to normalized form."""
    if hasattr(device, 'type'):  # torch.device
        device = str(device)
    device = device.lower().split(':')[0]  # 'cuda:0' -> 'cuda'
    if device in ('trt',):
        device = 'tensorrt'
    return device


class OnnxHomosolver:
    """ONNX-based Russian homograph solver."""

    def __init__(self, device='cpu'):
        self._device_str = _parse_device(device)

        # Load model bytes (cached for .to() calls)
        self._bert_bytes = _load_gzip_onnx('ru', 'bert_ru.onnx.gz')
        self._clf_bytes = _load_gzip_onnx('ru', 'homo_clf_ru.onnx.gz')

        # Create sessions
        self._create_sessions()

        vocab = _load_gzip_json('ru', 'vocab_ru.json.gz')
        special_tokens = _load_gzip_json('ru', 'special_tokens_ru.json.gz')
        self.tokenizer = SimpleBertTokenizer(vocab, special_tokens)

        self.homodict = _load_gzip_json('ru', 'homodict_ru.json.gz')
        self.yohomodict = _load_gzip_json('ru', 'yohomodict_ru.json.gz')

        # Pattern matches words containing at least one Cyrillic letter, may include '+'
        # Original: r'(?=.*[а-яё])[а-яё+]+' with re.IGNORECASE
        self.pattern = re.compile(r'(?=.*[а-яёА-ЯЁ])[а-яёА-ЯЁ+]+')
        self.vowels = 'аоуыэиеяёюАОУЫЭИЕЯЁЮ'

    def _create_sessions(self):
        """Create ONNX sessions with current device providers."""
        providers = _get_onnx_providers(self._device_str)
        self.bert_session = ort.InferenceSession(self._bert_bytes, providers=providers)
        self.clf_session = ort.InferenceSession(self._clf_bytes, providers=providers)

    def to(self, device):
        """Move model to device. Returns self for chaining."""
        new_device = _parse_device(device)
        if new_device != self._device_str:
            self._device_str = new_device
            self._create_sessions()
        return self

    @property
    def device(self):
        """Return current device string."""
        return self._device_str

    def __call__(self, text, put_stress_homo=True, put_yo_homo=True, stress_single_vowel=True):
        """Process text and resolve homographs.

        Args:
            text: Input text
            put_stress_homo: Add stress marks to homographs
            put_yo_homo: Replace е with ё in homographs
            stress_single_vowel: Add stress even for single-vowel words
        """
        if not (put_stress_homo or put_yo_homo):
            return text

        tagged = self._find_and_tag_homos(text, put_stress_homo, put_yo_homo)
        if not tagged:
            return text

        result = text
        offset = 0

        for start, end, word, marked_text in tagged:
            input_ids = self._encode_marked(marked_text)
            if input_ids is None:
                continue

            homo_start = np.where(input_ids[0] == self.tokenizer.homo_start_id)[0]
            homo_end = np.where(input_ids[0] == self.tokenizer.homo_end_id)[0]

            if len(homo_start) == 0 or len(homo_end) == 0:
                continue

            homo_start = homo_start[0]
            homo_end = homo_end[0]

            hidden_states = self.bert_session.run(None, {'input_ids': input_ids})[0]
            marker_emb = hidden_states[0, homo_start, :]
            word_tokens = hidden_states[0, homo_start + 1:homo_end, :]
            word_emb = word_tokens.mean(axis=0)
            combined = np.concatenate([marker_emb, word_emb]).reshape(1, -1).astype(np.float32)

            logit = self.clf_session.run(None, {'embedding': combined})[0]
            pred = 1 if 1 / (1 + np.exp(-logit[0, 0])) > 0.5 else 0

            word_lower = word.lower()
            if word_lower in self.homodict:
                variants = sorted(self.homodict[word_lower])
            elif word_lower in self.yohomodict:
                variants = sorted(self.yohomodict[word_lower])
            else:
                continue

            word_pred = variants[pred]

            # Handle yo replacement if disabled
            if not put_yo_homo:
                word_pred = word_pred.replace('ё', 'е')

            n_vowels = sum(c.lower() in 'аоуыэиеяёю' for c in word_pred)
            stress_idx = word_pred.index('+')
            word_pred_no_stress = word_pred.replace('+', '')

            # Apply case from original word to predicted word
            word_pred_cased = ''.join(
                c2.lower() if c1.islower() else c2.upper()
                for c1, c2 in zip(word, word_pred_no_stress)
            )

            # Add stress mark if (multiple vowels OR stress_single_vowel) AND put_stress_homo
            if (n_vowels > 1 or stress_single_vowel) and put_stress_homo:
                word_pred_cased = word_pred_cased[:stress_idx] + '+' + word_pred_cased[stress_idx:]
                extra_offset = 1
            else:
                extra_offset = 0

            actual_start = start + offset
            actual_end = end + offset
            result = result[:actual_start] + word_pred_cased + result[actual_end:]
            offset += extra_offset

        return result

    def _find_and_tag_homos(self, text, put_stress_homo, put_yo_homo):
        """Find homographs and create marked versions."""
        tagged = []
        for match in self.pattern.finditer(text):
            word = match.group()
            word_lower = word.lower()
            start, end = match.start(), match.end()

            is_stress_homo = word_lower in self.homodict and put_stress_homo
            is_yo_homo = word_lower in self.yohomodict and put_yo_homo

            if is_stress_homo or is_yo_homo:
                marked = text[:start] + ' [HOMO] ' + word + ' [/HOMO] ' + text[end:]
                tagged.append((start, end, word, marked))
        return tagged

    def _encode_marked(self, marked_text):
        """Encode text with [HOMO] markers."""
        ids = self.tokenizer.encode(marked_text)
        return np.array([ids], dtype=np.int64)


class _AccentorBase:
    """Base class for all accentors."""

    def __init__(self, lang):
        self.lang = lang
        self.stress_token = '+'

    def _get_ngram_indices(self, word):
        """Get n-gram indices for a word. Returns numpy array."""
        ngrams = word_ngrams(word, 1, len(word) + 3)
        indices = [self.ngram_dict[ng] for ng in ngrams if ng in self.ngram_dict]
        if not indices:
            return np.array([self.unk_idx], dtype=np.int32)
        return np.array(indices, dtype=np.int32)

    def _embed_word(self, word):
        """Compute embedding for a single word."""
        indices = self._get_ngram_indices(word)
        return np.mean(self.emb_weights[indices], axis=0)

    def _embed_words(self, words):
        """Compute embeddings for words using n-gram lookup."""
        if not words:
            return np.empty((0, self.emb_weights.shape[1]), dtype=np.float32)
        # Pre-allocate output array
        n_words = len(words)
        emb_dim = self.emb_weights.shape[1]
        embeddings = np.empty((n_words, emb_dim), dtype=np.float32)
        for i, word in enumerate(words):
            embeddings[i] = self._embed_word(word)
        return embeddings

    def _classify(self, embeddings, weights):
        """Run MLP classifier on embeddings."""
        x = embeddings
        x = np.dot(x, weights['0_weight'].T) + weights['0_bias']
        x = np.maximum(x, 0)
        x = np.dot(x, weights['2_weight'].T) + weights['2_bias']
        x = np.maximum(x, 0)
        x = np.dot(x, weights['4_weight'].T) + weights['4_bias']
        x = np.maximum(x, 0)
        x = np.dot(x, weights['6_weight'].T) + weights['6_bias']
        return x

    def _accentuate_word(self, word, stress_pos, is_char_pos=False):
        """Add stress marker to word at predicted position."""
        if is_char_pos:
            stress_idx = stress_pos
        else:
            vowel_ids = [i for i, c in enumerate(word.lower()) if c in self.vowels]
            if not vowel_ids:
                return word
            if stress_pos < len(vowel_ids):
                stress_idx = vowel_ids[stress_pos]
            elif len(vowel_ids) == 1:
                stress_idx = vowel_ids[0]
            else:
                stress_idx = vowel_ids[-1]

        return word[:stress_idx] + self.stress_token + word[stress_idx:]


class AccentorUkrBel(_AccentorBase):
    """Accentor for Ukrainian and Belarusian."""

    # Pre-compiled regex for tokenization
    _TOKEN_SPLIT_RE = re.compile(r'([\s.,!?;:<>=()/\\]+)')

    def __init__(self, lang):
        super().__init__(lang)
        self._device_str = 'cpu'

        self.ngram_dict = _load_gzip_json(lang, f'ngram_dict_{lang}.json.gz')
        self.unk_idx = self.ngram_dict.get('UNK', 0)

        emb_data = _load_npz(lang, f'embedding_{lang}.npz')
        self.emb_weights = emb_data['scale'][0] * (emb_data['weights'].astype(np.float32) - emb_data['zero_point'][0])

        self.exceptions = _load_gzip_json(lang, f'exceptions_{lang}.json.gz')
        self.stress_weights = _load_npz(lang, f'stress_clf_{lang}_weights.npz')

        self._init_language_settings(lang)

    def to(self, device):
        """Move model to device. Returns self for chaining.

        Note: UKR/BEL models are CPU-only (pure NumPy), but we support
        .to() for API consistency with Russian and PyTorch backends.
        """
        self._device_str = _parse_device(device)
        return self

    @property
    def device(self):
        """Return current device string."""
        return self._device_str

    def _init_language_settings(self, lang):
        """Initialize language-specific alphabet and vowels."""
        if lang == "ukr":
            alpha = "абвгґдеєжзиіїйклмнопрстуфхцчшщьюя'"
            self.vowels = "аеєиіїоуюя"
        elif lang == "bel":
            alpha = "абвгдежзйклмнопрстуфхцчшыьэюяёіў'"
            self.vowels = "аоуіэыяеёю"
        else:
            raise ValueError(f"Unsupported language: {lang}")

        alpha = ''.join(sorted(set(alpha + alpha.upper())))
        self._clean_re = re.compile(fr'[^{alpha}]')

    def _predict_stress(self, words):
        """Predict stress position for words."""
        if not words:
            return []
        embeddings = self._embed_words(words)
        logits = self._classify(embeddings, self.stress_weights)
        return np.argmax(logits, axis=1).tolist()

    def __call__(self, sentence, **kwargs):
        """Process sentence and add stress marks."""
        raw_tokens = []
        clean_tokens = []
        needs_processing = []

        for part in self._TOKEN_SPLIT_RE.split(sentence):
            subparts = part.split('-')
            if len(subparts) == 1:
                tokens = subparts
            else:
                tokens = [p + '-' for p in subparts[:-1]] + [subparts[-1]]

            for token in tokens:
                raw_tokens.append(token)
                clean = self._clean_re.sub('', token.lower())
                clean_tokens.append(clean)
                needs_processing.append(len(clean) > 0)

        words_to_predict = []
        word_indices = []
        exception_results = {}

        for i, (clean, need) in enumerate(zip(clean_tokens, needs_processing)):
            if need and self.stress_token not in raw_tokens[i].lower():
                if clean in self.exceptions:
                    exception_results[i] = self.exceptions[clean][0]
                else:
                    words_to_predict.append(clean)
                    word_indices.append(i)

        predictions = self._predict_stress(words_to_predict) if words_to_predict else []

        pred_idx = 0
        result = []
        for i, (raw, clean, need) in enumerate(zip(raw_tokens, clean_tokens, needs_processing)):
            if not need or self.stress_token in raw.lower():
                result.append(raw)
            elif i in exception_results:
                result.append(self._accentuate_word(raw, exception_results[i], is_char_pos=True))
            elif i in word_indices:
                result.append(self._accentuate_word(raw, predictions[pred_idx], is_char_pos=False))
                pred_idx += 1
            else:
                result.append(raw)

        return ''.join(result)


class AccentorRu(_AccentorBase):
    """Accentor for Russian with optional BERT homograph solver."""

    # Pre-compiled regex patterns
    _TOKEN_SPLIT_RE = re.compile(r'([\s.,!?;:<>=()/\\]+)')
    _CLEAN_RE = re.compile(r'[^А-Яа-яёЁ]')

    def __init__(self, use_homosolver=True, device='cpu'):
        super().__init__('ru')
        self._device_str = _parse_device(device)

        self.ngram_dict = _load_gzip_json('ru', 'ngram_dict_ru.json.gz')
        self.unk_idx = self.ngram_dict.get('UNK', 0)

        emb_data = _load_npz('ru', 'embedding_ru.npz')
        weights = emb_data['weights']
        if weights.dtype == np.int8:
            self.emb_weights = emb_data['scale'][0] * (weights.astype(np.float32) - emb_data['zero_point'][0])
        else:
            self.emb_weights = weights.astype(np.float32)

        self.exceptions = _load_gzip_json('ru', 'exceptions_ru.json.gz')
        self.stress_weights = _load_npz('ru', 'stress_clf_ru_weights.npz')
        self.yo_weights = _load_npz('ru', 'yo_clf_ru_weights.npz')

        self.vowels = 'аоуыэиеяёю'

        if use_homosolver and HAS_ONNX:
            try:
                self.homosolver = OnnxHomosolver(device=device)
            except Exception as e:
                print(f"Warning: Could not load homosolver: {e}")
                self.homosolver = None
        else:
            self.homosolver = None

    def to(self, device):
        """Move model to device. Returns self for chaining."""
        self._device_str = _parse_device(device)
        if self.homosolver is not None:
            self.homosolver.to(device)
        return self

    @property
    def device(self):
        """Return current device string."""
        return self._device_str

    def _predict_stress(self, words, embeddings=None):
        """Predict stress positions. Optionally accepts pre-computed embeddings."""
        if not words:
            return []
        if embeddings is None:
            embeddings = self._embed_words(words)
        logits = self._classify(embeddings, self.stress_weights)
        return np.argmax(logits, axis=1).tolist()

    def _predict_yo(self, words, embeddings=None):
        """Predict yo positions. Optionally accepts pre-computed embeddings."""
        if not words:
            return []
        if embeddings is None:
            embeddings = self._embed_words(words)
        logits = self._classify(embeddings, self.yo_weights)
        return np.argmax(logits, axis=1).tolist()

    def _yo_replace_word(self, word, yo_pos, is_char_pos=False):
        """Replace 'е' with 'ё' at predicted position."""
        if is_char_pos:
            if yo_pos < 0 or yo_pos >= len(word):
                return word
            char = word[yo_pos]
            if char == 'е':
                return word[:yo_pos] + 'ё' + word[yo_pos + 1:]
            elif char == 'Е':
                return word[:yo_pos] + 'Ё' + word[yo_pos + 1:]
            return word
        else:
            e_positions = [i for i, c in enumerate(word) if c in 'еЕ']
            if not e_positions or yo_pos >= len(e_positions):
                return word
            char_idx = e_positions[yo_pos]
            char = word[char_idx]
            if char == 'е':
                return word[:char_idx] + 'ё' + word[char_idx + 1:]
            elif char == 'Е':
                return word[:char_idx] + 'Ё' + word[char_idx + 1:]
            return word

    def __call__(self, sentence, put_stress=True, put_yo=True,
                 put_stress_homo=True, put_yo_homo=True, stress_single_vowel=True, **kwargs):
        """Process sentence.

        Args:
            sentence: Input text
            put_stress: Add stress marks to words
            put_yo: Replace е with ё where appropriate
            put_stress_homo: Add stress marks to homographs (requires homosolver)
            put_yo_homo: Replace е with ё in homographs (requires homosolver)
            stress_single_vowel: Add stress even for single-vowel words
        """
        if self.homosolver and (put_stress_homo or put_yo_homo):
            sentence = self.homosolver(sentence, put_stress_homo, put_yo_homo, stress_single_vowel)

        if not put_stress and not put_yo:
            return sentence

        raw_tokens = []
        clean_tokens = []
        needs_processing = []

        for part in self._TOKEN_SPLIT_RE.split(sentence):
            subparts = part.split('-')
            if len(subparts) == 1:
                tokens = subparts
            else:
                tokens = [p + '-' for p in subparts[:-1]] + [subparts[-1]]

            for token in tokens:
                raw_tokens.append(token)
                clean = self._CLEAN_RE.sub('', token.lower())
                clean_tokens.append(clean)
                needs_processing.append(len(clean) > 0)

        stress_words_to_predict = []
        stress_word_indices = []
        yo_words_to_predict = []
        yo_word_indices = []
        exception_stress = {}
        exception_yo = {}

        for i, (clean, need) in enumerate(zip(clean_tokens, needs_processing)):
            if not need:
                continue
            has_stress = self.stress_token in raw_tokens[i].lower()

            if clean in self.exceptions:
                exc = self.exceptions[clean]
                if put_stress and not has_stress:
                    exception_stress[i] = exc[0]
                if put_yo and len(exc) > 1 and exc[1] >= 0:
                    exception_yo[i] = exc[1]
            else:
                if put_stress and not has_stress:
                    stress_words_to_predict.append(clean)
                    stress_word_indices.append(i)
                if put_yo and 'е' in clean:
                    yo_words_to_predict.append(clean)
                    yo_word_indices.append(i)

        stress_predictions = self._predict_stress(stress_words_to_predict) if stress_words_to_predict else []

        # Compute yo embeddings once, reuse for both yo and stress predictions
        if yo_words_to_predict:
            yo_embeddings = self._embed_words(yo_words_to_predict)
            yo_predictions = self._predict_yo(yo_words_to_predict, yo_embeddings)
            yo_stress_predictions = self._predict_stress(yo_words_to_predict, yo_embeddings)
        else:
            yo_predictions = []
            yo_stress_predictions = []

        result_tokens = list(raw_tokens)

        # Build mapping from word index to stress prediction
        stress_pred_map = {}
        for idx, word_idx in enumerate(stress_word_indices):
            stress_pred_map[word_idx] = stress_predictions[idx]

        # Build mapping from word index to yo-related stress prediction
        yo_stress_pred_map = {}
        for idx, word_idx in enumerate(yo_word_indices):
            yo_stress_pred_map[word_idx] = yo_stress_predictions[idx]

        if put_yo:
            yo_pred_idx = 0
            for i, (raw, clean, need) in enumerate(zip(raw_tokens, clean_tokens, needs_processing)):
                if not need:
                    continue
                if i in exception_yo:
                    yo_char_pos = exception_yo[i]
                    # ё must be stressed - check if pre-existing stress matches yo position
                    has_stress = self.stress_token in raw
                    if has_stress:
                        stress_marker_idx = raw.lower().index(self.stress_token)
                        # Only convert if stress is on the same char
                        if stress_marker_idx == yo_char_pos:
                            result_tokens[i] = self._yo_replace_word(result_tokens[i], yo_char_pos, is_char_pos=True)
                    else:
                        # No pre-existing stress, check if exception stress matches yo
                        if i in exception_stress:
                            exc_stress_pos = exception_stress[i]
                            if exc_stress_pos == yo_char_pos:
                                result_tokens[i] = self._yo_replace_word(result_tokens[i], yo_char_pos, is_char_pos=True)
                        else:
                            # Fallback: apply yo conversion (original behavior)
                            result_tokens[i] = self._yo_replace_word(result_tokens[i], yo_char_pos, is_char_pos=True)
                elif i in yo_word_indices:
                    yo_pos = yo_predictions[yo_pred_idx]
                    yo_pred_idx += 1
                    if yo_pos == 0:
                        continue  # yo_clf says no conversion

                    e_positions = [j for j, c in enumerate(clean) if c == 'е']
                    vowel_positions = [j for j, c in enumerate(clean) if c in self.vowels]
                    yo_char_idx = e_positions[yo_pos - 1] if yo_pos - 1 < len(e_positions) else None

                    if yo_char_idx is None:
                        continue

                    has_stress = self.stress_token in raw

                    if has_stress:
                        # Pre-stressed word (e.g., from homosolver)
                        # Use ACTUAL stress position from '+' marker, not n-gram prediction
                        # ё can only appear at stressed position, so check if stressed vowel is 'е'
                        stress_marker_idx = raw.lower().index(self.stress_token)
                        # The stressed vowel is immediately after the '+'
                        if stress_marker_idx + 1 < len(raw):
                            stressed_char = raw[stress_marker_idx + 1].lower()
                            # Only convert if the stressed vowel is 'е'
                            if stressed_char == 'е':
                                # The stressed 'е' is at position stress_marker_idx in the original word (without +)
                                # since clean doesn't have '+', we need to find the char index
                                actual_stressed_char_idx = stress_marker_idx  # in clean (no +)
                                # Only convert if yo_pred points to this same 'е'
                                if actual_stressed_char_idx == yo_char_idx:
                                    result_tokens[i] = self._yo_replace_word(result_tokens[i], yo_pos - 1, is_char_pos=False)
                        # If stressed vowel is not 'е', skip yo conversion (ё must be stressed)
                    else:
                        # Non-pre-stressed word: use n-gram stress prediction
                        stress_pred = yo_stress_pred_map.get(i, 0)
                        stressed_char_idx = vowel_positions[stress_pred] if stress_pred < len(vowel_positions) else None

                        # Only convert if stress_pred vowel == yo-predicted е
                        if stressed_char_idx is not None and stressed_char_idx == yo_char_idx:
                            result_tokens[i] = self._yo_replace_word(result_tokens[i], yo_pos - 1, is_char_pos=False)

        if put_stress:
            stress_pred_idx = 0
            for i, (clean, need) in enumerate(zip(clean_tokens, needs_processing)):
                if not need or self.stress_token in result_tokens[i].lower():
                    continue

                # Count vowels in word
                num_vowels = sum(1 for c in clean if c in self.vowels)

                # Skip single-vowel words if stress_single_vowel=False
                if num_vowels == 1 and not stress_single_vowel:
                    if i in stress_word_indices:
                        stress_pred_idx += 1
                    continue

                if i in exception_stress:
                    result_tokens[i] = self._accentuate_word(result_tokens[i], exception_stress[i], is_char_pos=True)
                elif i in stress_word_indices:
                    result_tokens[i] = self._accentuate_word(result_tokens[i], stress_predictions[stress_pred_idx], is_char_pos=False)
                    stress_pred_idx += 1

        return ''.join(result_tokens)
