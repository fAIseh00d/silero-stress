"""
Pure NumPy implementation of silero-stress accentor.
NO PyTorch required! ONNX Runtime is optional (only for Russian BERT homograph solver).
"""

import os
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


def _capitalize_stress(stressed_word):
    """'з+амок' -> 'зАмок' (regex group names can't contain '+')."""
    stress_idx = stressed_word.index('+')
    return stressed_word[:stress_idx] + stressed_word[stress_idx + 1].capitalize() + stressed_word[stress_idx + 2:]


def _decapitalize_stress(stressed_word):
    """'зАмок' -> 'з+амок' (inverse of _capitalize_stress)."""
    for idx, char in enumerate(stressed_word):
        if char.isupper():
            return stressed_word[:idx] + '+' + stressed_word[idx:].lower()


def _compile_phrases(phrases_dict):
    """Compile {word: {phrase: variant}} into {word: regex}, as upstream v1.5 does.

    Phrases are matched against the [HOMO]-marked context, longest first, and the
    matching named group (variant with capitalized stressed vowel) gives the answer.
    """
    compiled = {}
    for word, phrases_to_variants in phrases_dict.items():
        variant_to_phrases = {}
        for phrase, variant in phrases_to_variants.items():
            variant_to_phrases.setdefault(variant, []).append(phrase)
        variant_patterns = []
        for variant, phrases in variant_to_phrases.items():
            group_name = _capitalize_stress(variant)
            sorted_phrases = sorted(phrases, key=len, reverse=True)
            sorted_phrases = [p.replace(word, f'[HOMO] {word} [/HOMO]').strip() for p in sorted_phrases]
            joined = '|'.join(map(re.escape, sorted_phrases))
            variant_patterns.append(rf'(?P<{group_name}>(?<![а-яА-ЯёЁ\-])(?:{joined})(?![а-яА-ЯёЁ\-]))')
        compiled[word] = re.compile('|'.join(variant_patterns), re.IGNORECASE)
    return compiled


class _LazyRegexDict(dict):
    """{word: compiled regex} that compiles each pattern on first lookup.

    Compiling all ~1.5K built-in phrase patterns eagerly takes ~2.5s at load time,
    while a typical text touches only a handful of them.
    """

    def __init__(self, sources, flags):
        super().__init__()
        self._sources = sources
        self._flags = flags

    def __contains__(self, word):
        return word in self._sources

    def __missing__(self, word):
        compiled = self[word] = re.compile(self._sources[word], self._flags)
        return compiled

    def __len__(self):
        return len(self._sources)

    def __iter__(self):
        return iter(self._sources)

    def keys(self):
        return self._sources.keys()

    def items(self):
        return ((w, self[w]) for w in self._sources)


class OnnxHomosolver:
    """ONNX-based Russian homograph solver (mirrors upstream v1.5 HomoSolver).

    Resolution order for each homograph candidate:
      1. user regexes   (load_regex_dict)   - matched on the cleaned context
      2. user phrases   (load_phrases_dict) - matched on the [HOMO]-marked context
      3. built-in phrases                    - matched on the [HOMO]-marked context
      4. BERT + classifier (only if the word is in homodict)
    """

    WINDOW_SIZE = 300  # chars of cleaned context around the homograph (half before, half after)

    # Text-cleaning regexes (identical to upstream v1.5)
    _re_remove_extra = re.compile(r'[^a-zA-Zа-яА-ЯёЁ0-9\s.!?,\-]')
    _re_spaces = re.compile(r'\s+')
    _re_double_dash = re.compile(r'-{2,}')
    _re_repeat_punct = re.compile(r'([.!?])\1+')
    _re_repeat_comma = re.compile(r',{2,}')
    _re_space_before_punct = re.compile(r'\s+([.,!?])')
    _re_punct_add_space = re.compile(r'([.,!?])(?=\S)')

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

        # Built-in phrases ship pre-compiled: {word: pattern_string} (see scripts/export_ru.py)
        self.compiled_phrases = _LazyRegexDict(_load_gzip_json('ru', 'phrases_ru.json.gz'), re.IGNORECASE)
        self.compiled_user_phrases = {}
        self.compiled_user_patterns = {}

        # Pattern matches words containing at least one Cyrillic letter, may include '+'
        # Original: r'(?=.*[а-яё])[а-яё+]+' with re.IGNORECASE
        # Upstream: r'(?=.*[а-яё])[а-яё+]+' (IGNORECASE). Its '.*' lookahead rescans the rest of
        # the line for every word (quadratic on long texts). For any run that contains a
        # Cyrillic letter, '[+]*[а-яё]' accepts exactly the same spans; it only differs on
        # runs made purely of '+', which can never be homographs.
        self.pattern = re.compile(r'(?=[+]*[а-яёА-ЯЁ])[а-яёА-ЯЁ+]+')
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

    # ------------------------------------------------------------------ #
    # User overrides (same API and JSON formats as upstream v1.5)
    # ------------------------------------------------------------------ #

    def load_phrases_dict(self, phrases_dict_path):
        """Load user-defined phrases from a JSON file (replaces previously loaded ones).

        Format: {"замок": {"старинный замок": "з+амок", "закрыто на замок": "зам+ок"}}
        """
        if not os.path.exists(phrases_dict_path):
            raise FileNotFoundError(f"There is no regex dict at: {phrases_dict_path}")
        with open(phrases_dict_path, 'r', encoding='utf-8') as f:
            self.compiled_user_phrases = _compile_phrases(json.load(f))
        print(f'Loaded new phrases dict for {len(self.compiled_user_phrases)} words')

    def load_regex_dict(self, regex_dict_path):
        """Load user-defined regexes from a JSON file (merged into previously loaded ones).

        Format: {"замок": {"з+амок": ["regex1", "regex2"], "зам+ок": ["regex3"]}}
        Regexes are matched against the cleaned context (no [HOMO] markers),
        in the order given.
        """
        if not os.path.exists(regex_dict_path):
            raise FileNotFoundError(f"There is no regex dict at: {regex_dict_path}")
        with open(regex_dict_path, 'r', encoding='utf-8') as f:
            regex_dict = json.load(f)
        for word, variant_to_regexes in regex_dict.items():
            variant_patterns = []
            for variant, regexes in variant_to_regexes.items():
                group_name = _capitalize_stress(variant)
                joined = '|'.join(f'(?:{r})' for r in regexes)
                variant_patterns.append(rf'(?P<{group_name}>(?<![а-яА-ЯёЁ])(?:{joined})(?![а-яА-ЯёЁ]))')
            try:
                self.compiled_user_patterns[word] = re.compile('|'.join(variant_patterns), re.IGNORECASE | re.UNICODE)
            except re.error as e:
                raise ValueError(f"Error during regex compilation for word '{word}': {e}")
        print(f'Loaded new regex dict for {len(self.compiled_user_patterns)} words')

    @staticmethod
    def _predict_with_pattern(pattern, text):
        match = pattern.search(text)
        if match:
            for group_name, matched_text in match.groupdict().items():
                if matched_text is not None:
                    return _decapitalize_stress(group_name)
        return None

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def __call__(self, text, put_stress_homo=True, put_yo_homo=True, stress_single_vowel=True):
        """Process text and resolve homographs.

        Args:
            text: Input text
            put_stress_homo: Add stress marks to homographs
            put_yo_homo: Replace е with ё in homographs
            stress_single_vowel: Add stress even for single-vowel words
        """
        return self.batch([text], put_stress_homo, put_yo_homo, stress_single_vowel)[0]

    def batch(self, texts, put_stress_homo=True, put_yo_homo=True, stress_single_vowel=True):
        """Resolve homographs in several independent texts with one BERT call."""
        if not (put_stress_homo or put_yo_homo):
            return list(texts)

        # word_preds[i] = list of (start, end, word, variant) for texts[i]
        word_preds = [[] for _ in texts]
        neural = []  # (text_idx, slot_idx, word_lower, raw_mark)

        for t_idx, text in enumerate(texts):
            for start, end, word, word_lower, raw_mark, raw_clean in self._find_and_tag_homos(text):
                pred = None
                if word_lower in self.compiled_user_patterns:
                    pred = self._predict_with_pattern(self.compiled_user_patterns[word_lower], raw_clean)
                if pred is None and word_lower in self.compiled_user_phrases:
                    pred = self._predict_with_pattern(self.compiled_user_phrases[word_lower], raw_mark)
                if pred is None and word_lower in self.compiled_phrases:
                    pred = self._predict_with_pattern(self.compiled_phrases[word_lower], raw_mark)

                if pred is None:
                    if word_lower not in self.homodict:
                        continue
                    neural.append((t_idx, len(word_preds[t_idx]), word_lower, raw_mark))
                word_preds[t_idx].append([start, end, word, pred])

        if neural:
            preds = self._run_model([raw_mark for _, _, _, raw_mark in neural])
            for (t_idx, slot, word_lower, _), pred in zip(neural, preds):
                word_preds[t_idx][slot][3] = sorted(self.homodict[word_lower])[pred]

        return [self._apply(text, wp, put_stress_homo, put_yo_homo, stress_single_vowel)
                for text, wp in zip(texts, word_preds)]

    def _apply(self, text, word_preds, put_stress, put_yo, stress_single_vowel):
        """Write predicted variants back into text, preserving the original case."""
        result = text
        offset = 0
        for start, end, word, word_pred in word_preds:
            start += offset
            end += offset
            if not put_yo:
                word_pred = word_pred.replace('ё', 'е')
            n_vowels = sum(c.lower() in 'аоуыэиеяёю' for c in word_pred)
            stress_idx = word_pred.index('+')
            word_pred = word_pred.replace('+', '')
            word_pred = ''.join(c2.lower() if c1.islower() else c2.upper() for c1, c2 in zip(word, word_pred))
            if (n_vowels > 1 or stress_single_vowel) and put_stress:
                word_pred = word_pred[:stress_idx] + '+' + word_pred[stress_idx:]
                offset += 1
            result = result[:start] + word_pred + result[end:]
        return result

    def _clean_text(self, text, is_start=True):
        """Normalize context around a homograph (identical to upstream v1.5)."""
        if not text:
            return ""
        text = self._re_remove_extra.sub('', text)
        text = self._re_spaces.sub(' ', text)
        text = self._re_double_dash.sub(' - ', text)
        text = self._re_repeat_punct.sub(r'\1', text)
        text = self._re_repeat_comma.sub(',', text)
        text = self._re_space_before_punct.sub(r'\1', text)
        text = self._re_repeat_punct.sub(r'\1', text)
        text = self._re_repeat_comma.sub(',', text)
        text = self._re_punct_add_space.sub(r'\1 ', text)
        text = self._re_spaces.sub(' ', text).strip()
        if is_start:
            text = text.lstrip(' .,!?-')
            if text:
                text = text[0].upper() + text[1:].lower()
        else:
            if text:
                text = text[0] + text[1:].lower()
            if text and text[-1] not in '.!?':
                text += '.'
        return text

    # Cleaning is local (every rule looks at a few neighbouring chars), so the last/first
    # `half` cleaned chars only depend on a bounded raw neighbourhood. Clean a raw chunk and
    # grow it until the result has a safety margin, instead of cleaning the whole prefix /
    # suffix for every homograph (quadratic on long texts). Output is identical to
    # _clean_text(text[:start])[-half:] / _clean_text(text[end:])[:half].
    _CLEAN_MARGIN = 32

    def _clean_left(self, text, start, half):
        chunk = 4 * half
        while True:
            lo = max(0, start - chunk)
            cleaned = self._clean_text(text[lo:start], is_start=True)
            if lo == 0 or len(cleaned) >= half + self._CLEAN_MARGIN:
                return cleaned[-half:]
            chunk *= 2

    def _clean_right(self, text, end, half):
        chunk = 4 * half
        while True:
            hi = min(len(text), end + chunk)
            cleaned = self._clean_text(text[end:hi], is_start=False)
            if hi == len(text) or len(cleaned) >= half + self._CLEAN_MARGIN:
                return cleaned[:half]
            chunk *= 2

    def _find_and_tag_homos(self, text):
        """Find homograph candidates and build their (windowed, cleaned) contexts.

        Returns list of (start, end, word, word_lower, raw_mark, raw_clean).
        """
        half = self.WINDOW_SIZE // 2
        tagged = []
        for match in self.pattern.finditer(text):
            word = match.group()
            word_lower = word.lower()
            if not (word_lower in self.homodict
                    or word_lower in self.compiled_phrases
                    or word_lower in self.compiled_user_phrases
                    or word_lower in self.compiled_user_patterns):
                continue
            start, end = match.span()
            left = self._clean_left(text, start, half)
            right = self._clean_right(text, end, half)
            raw_mark = (left + ' [HOMO] ' + word_lower + ' [/HOMO] ' + right).strip()
            raw_clean = (left + ' ' + word_lower + ' ' + right).strip()
            tagged.append((start, end, word, word_lower, raw_mark, raw_clean))
        return tagged

    def _run_model(self, marked_texts):
        """Batched BERT + classifier. Returns list of 0/1 variant indices."""
        all_ids = []
        for marked in marked_texts:
            ids = self.tokenizer.encode(marked)
            hs = ids.index(self.tokenizer.homo_start_id)
            he = ids.index(self.tokenizer.homo_end_id, hs)
            all_ids.append((ids, hs, he))

        # Pad to same length (pad_token_id)
        max_len = max(len(ids) for ids, _, _ in all_ids)
        batch_ids = np.full((len(all_ids), max_len), self.tokenizer.pad_token_id, dtype=np.int64)
        for i, (ids, _, _) in enumerate(all_ids):
            batch_ids[i, :len(ids)] = ids

        hidden_states = self.bert_session.run(None, {'input_ids': batch_ids})[0]

        # Embedding = [HOMO] marker state ++ mean of the word's token states
        dim = hidden_states.shape[-1]
        embeddings = np.empty((len(all_ids), dim * 2), dtype=np.float32)
        for i, (_, hs, he) in enumerate(all_ids):
            embeddings[i, :dim] = hidden_states[i, hs]
            embeddings[i, dim:] = hidden_states[i, hs + 1:he].mean(axis=0)

        logits = self.clf_session.run(None, {'embedding': embeddings})[0]
        # round(sigmoid(x)) == (x > 0); upstream uses torch.round (0.5 -> 0)
        return (logits[:, 0] > 0).astype(np.int64).tolist()


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
    # Sentence boundary pattern (split on .!? followed by space or end)
    _SENTENCE_SPLIT_RE = re.compile(r'((?<=[.!?])\s+)')

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

    def __call__(self, text, **kwargs):
        """Process text and add stress marks. Auto-batches if multiple sentences detected."""
        # Auto-batch if multiple sentences
        parts = self._SENTENCE_SPLIT_RE.split(text)
        if len(parts) > 1:
            sentences, seps = parts[0::2], parts[1::2]
            return ''.join(s + sep for s, sep in zip(self.batch(sentences), seps + ['']))

        # Single sentence path
        sentence = text
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

    def batch(self, sentences):
        """Process multiple sentences in a single batch for better throughput.

        Args:
            sentences: List of input sentences

        Returns:
            List of processed sentences with stress marks
        """
        if not sentences:
            return []

        # 1. Tokenize all sentences
        all_raw_tokens = []      # raw tokens per sentence
        all_clean_tokens = []    # cleaned tokens per sentence
        all_needs_processing = []  # flags per sentence

        for sentence in sentences:
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

            all_raw_tokens.append(raw_tokens)
            all_clean_tokens.append(clean_tokens)
            all_needs_processing.append(needs_processing)

        # 2. Collect all words that need prediction (across all sentences)
        words_to_predict = []
        word_locations = []  # (sentence_idx, token_idx)
        exception_results = {}  # (sentence_idx, token_idx) -> stress_pos

        for sent_idx, (raw_tokens, clean_tokens, needs_processing) in enumerate(
            zip(all_raw_tokens, all_clean_tokens, all_needs_processing)
        ):
            for tok_idx, (raw, clean, need) in enumerate(zip(raw_tokens, clean_tokens, needs_processing)):
                if need and self.stress_token not in raw.lower():
                    if clean in self.exceptions:
                        exception_results[(sent_idx, tok_idx)] = self.exceptions[clean][0]
                    else:
                        words_to_predict.append(clean)
                        word_locations.append((sent_idx, tok_idx))

        # 3. Batch predict all words at once
        if words_to_predict:
            embeddings = self._embed_words(words_to_predict)
            logits = self._classify(embeddings, self.stress_weights)
            predictions = np.argmax(logits, axis=1)
        else:
            predictions = []

        # 4. Build prediction map
        pred_map = {}
        for i, (sent_idx, tok_idx) in enumerate(word_locations):
            pred_map[(sent_idx, tok_idx)] = predictions[i]

        # 5. Reassemble all sentences
        results = []
        for sent_idx, (raw_tokens, clean_tokens, needs_processing) in enumerate(
            zip(all_raw_tokens, all_clean_tokens, all_needs_processing)
        ):
            result_tokens = []
            for tok_idx, (raw, clean, need) in enumerate(zip(raw_tokens, clean_tokens, needs_processing)):
                key = (sent_idx, tok_idx)
                if not need or self.stress_token in raw.lower():
                    result_tokens.append(raw)
                elif key in exception_results:
                    result_tokens.append(self._accentuate_word(raw, exception_results[key], is_char_pos=True))
                elif key in pred_map:
                    result_tokens.append(self._accentuate_word(raw, pred_map[key], is_char_pos=False))
                else:
                    result_tokens.append(raw)
            results.append(''.join(result_tokens))

        return results


class AccentorRu(_AccentorBase):
    """Accentor for Russian with optional BERT homograph solver."""

    # Pre-compiled regex patterns
    _TOKEN_SPLIT_RE = re.compile(r'([\s.,!?;:<>=()/\\]+)')
    _CLEAN_RE = re.compile(r'[^А-Яа-яёЁ]')
    # Sentence boundary pattern
    _SENTENCE_SPLIT_RE = re.compile(r'((?<=[.!?])\s+)')

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

    def _require_homosolver(self):
        if self.homosolver is None:
            raise RuntimeError("Homosolver is not loaded (use_homosolver=False or onnxruntime missing)")
        return self.homosolver

    def load_phrases_dict(self, path):
        """Load user phrases JSON (v1.5 format) into the homosolver.

        Same as `accentor.homosolver.load_phrases_dict(path)`.
        """
        self._require_homosolver().load_phrases_dict(str(path))

    def load_regex_dict(self, path):
        """Load user regex JSON (v1.5 format) into the homosolver.

        Same as `accentor.homosolver.load_regex_dict(path)`.
        """
        self._require_homosolver().load_regex_dict(str(path))

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

    def __call__(self, text, put_stress=True, put_yo=True,
                 put_stress_homo=True, put_yo_homo=True, stress_single_vowel=True, **kwargs):
        """Process text and add stress marks. Auto-batches if multiple sentences detected.

        Args:
            text: Input text (single sentence or multiple)
            put_stress: Add stress marks to words
            put_yo: Replace е with ё where appropriate
            put_stress_homo: Add stress marks to homographs (requires homosolver)
            put_yo_homo: Replace е with ё in homographs (requires homosolver)
            stress_single_vowel: Add stress even for single-vowel words
        """
        # Homographs first, on the FULL text: v1.5 uses a +-150 char window of context
        # around each homograph that crosses sentence boundaries, like upstream.
        if self.homosolver and (put_stress_homo or put_yo_homo):
            text = self.homosolver(text, put_stress_homo, put_yo_homo, stress_single_vowel)

        # Auto-batch the n-gram part if multiple sentences (separators are preserved)
        parts = self._SENTENCE_SPLIT_RE.split(text)
        if len(parts) > 1:
            sentences, seps = parts[0::2], parts[1::2]
            out = self.batch(sentences, put_stress=put_stress, put_yo=put_yo,
                             stress_single_vowel=stress_single_vowel,
                             put_stress_homo=False, put_yo_homo=False)
            return ''.join(s + sep for s, sep in zip(out, seps + ['']))

        # Single sentence path
        sentence = text

        if not put_stress and not put_yo:
            return sentence

        raw_tokens = []
        clean_tokens = []
        needs_processing = []

        for part in self._TOKEN_SPLIT_RE.split(sentence):
            subparts = part.split('-')
            if len(subparts) == 1:
                tokens = subparts
                # Single token - process if non-empty
                process_flags = [True]
            else:
                tokens = [p + '-' for p in subparts[:-1]] + [subparts[-1]]
                # Special case: don't process '-то' suffix (что-то, кому-то, etc.)
                # This matches original PyTorch behavior for backward compatibility
                process_flags = [True] * (len(subparts) - 1) + [subparts[-1].lower() != 'то']

            for token, should_process in zip(tokens, process_flags):
                raw_tokens.append(token)
                clean = self._CLEAN_RE.sub('', token.lower())
                clean_tokens.append(clean)
                needs_processing.append(len(clean) > 0 and should_process)

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

    def batch(self, sentences, put_stress=True, put_yo=True, stress_single_vowel=True,
              put_stress_homo=True, put_yo_homo=True):
        """Process multiple sentences in a single batch for better throughput.

        Each sentence is an independent text: homographs in all of them are resolved
        with a single BERT call, and MLP-based stress/yo prediction is fully batched.

        Args:
            sentences: List of input sentences
            put_stress: Add stress marks to words
            put_yo: Replace е with ё where appropriate
            stress_single_vowel: Add stress even for single-vowel words
            put_stress_homo: Add stress marks to homographs
            put_yo_homo: Replace е with ё in homographs

        Returns:
            List of processed sentences
        """
        if not sentences:
            return []

        # Process homographs first (one batched BERT call across all sentences)
        if self.homosolver and (put_stress_homo or put_yo_homo):
            sentences = self.homosolver.batch(sentences, put_stress_homo, put_yo_homo, stress_single_vowel)

        if not put_stress and not put_yo:
            return list(sentences)

        # 1. Tokenize all sentences
        all_raw_tokens = []
        all_clean_tokens = []
        all_needs_processing = []

        for sentence in sentences:
            raw_tokens = []
            clean_tokens = []
            needs_processing = []

            for part in self._TOKEN_SPLIT_RE.split(sentence):
                subparts = part.split('-')
                if len(subparts) == 1:
                    tokens = subparts
                    process_flags = [True]
                else:
                    tokens = [p + '-' for p in subparts[:-1]] + [subparts[-1]]
                    # Special case: don't process '-то' suffix (что-то, кому-то, etc.)
                    process_flags = [True] * (len(subparts) - 1) + [subparts[-1].lower() != 'то']

                for token, should_process in zip(tokens, process_flags):
                    raw_tokens.append(token)
                    clean = self._CLEAN_RE.sub('', token.lower())
                    clean_tokens.append(clean)
                    needs_processing.append(len(clean) > 0 and should_process)

            all_raw_tokens.append(raw_tokens)
            all_clean_tokens.append(clean_tokens)
            all_needs_processing.append(needs_processing)

        # 2. Collect words for stress prediction
        stress_words = []
        stress_locations = []  # (sent_idx, tok_idx)
        exception_stress = {}  # (sent_idx, tok_idx) -> char_pos

        # 3. Collect words for yo prediction
        yo_words = []
        yo_locations = []
        exception_yo = {}

        for sent_idx, (raw_tokens, clean_tokens, needs_processing) in enumerate(
            zip(all_raw_tokens, all_clean_tokens, all_needs_processing)
        ):
            for tok_idx, (raw, clean, need) in enumerate(zip(raw_tokens, clean_tokens, needs_processing)):
                if not need:
                    continue
                key = (sent_idx, tok_idx)
                has_stress = self.stress_token in raw.lower()

                if clean in self.exceptions:
                    exc = self.exceptions[clean]
                    if put_stress and not has_stress:
                        exception_stress[key] = exc[0]
                    if put_yo and len(exc) > 1 and exc[1] >= 0:
                        exception_yo[key] = exc[1]
                else:
                    if put_stress and not has_stress:
                        stress_words.append(clean)
                        stress_locations.append(key)
                    if put_yo and 'е' in clean:
                        yo_words.append(clean)
                        yo_locations.append(key)

        # 4. Batch predict stress
        if stress_words:
            stress_embeddings = self._embed_words(stress_words)
            stress_logits = self._classify(stress_embeddings, self.stress_weights)
            stress_preds = np.argmax(stress_logits, axis=1)
        else:
            stress_preds = []

        stress_pred_map = {loc: stress_preds[i] for i, loc in enumerate(stress_locations)}

        # 5. Batch predict yo (reuse embeddings for stress check)
        if yo_words:
            yo_embeddings = self._embed_words(yo_words)
            yo_logits = self._classify(yo_embeddings, self.yo_weights)
            yo_preds = np.argmax(yo_logits, axis=1)
            # Also get stress predictions for yo words
            yo_stress_logits = self._classify(yo_embeddings, self.stress_weights)
            yo_stress_preds = np.argmax(yo_stress_logits, axis=1)
        else:
            yo_preds = []
            yo_stress_preds = []

        yo_pred_map = {loc: yo_preds[i] for i, loc in enumerate(yo_locations)}
        yo_stress_map = {loc: yo_stress_preds[i] for i, loc in enumerate(yo_locations)}

        # 6. Reassemble all sentences
        results = []
        for sent_idx, (raw_tokens, clean_tokens, needs_processing) in enumerate(
            zip(all_raw_tokens, all_clean_tokens, all_needs_processing)
        ):
            result_tokens = list(raw_tokens)

            # Apply yo replacements first
            if put_yo:
                for tok_idx, (raw, clean, need) in enumerate(zip(raw_tokens, clean_tokens, needs_processing)):
                    if not need:
                        continue
                    key = (sent_idx, tok_idx)

                    if key in exception_yo:
                        yo_char_pos = exception_yo[key]
                        has_stress = self.stress_token in raw
                        if has_stress:
                            stress_marker_idx = raw.lower().index(self.stress_token)
                            if stress_marker_idx == yo_char_pos:
                                result_tokens[tok_idx] = self._yo_replace_word(result_tokens[tok_idx], yo_char_pos, is_char_pos=True)
                        elif key in exception_stress:
                            if exception_stress[key] == yo_char_pos:
                                result_tokens[tok_idx] = self._yo_replace_word(result_tokens[tok_idx], yo_char_pos, is_char_pos=True)
                        else:
                            result_tokens[tok_idx] = self._yo_replace_word(result_tokens[tok_idx], yo_char_pos, is_char_pos=True)

                    elif key in yo_pred_map:
                        yo_pos = yo_pred_map[key]
                        if yo_pos == 0:
                            continue

                        e_positions = [j for j, c in enumerate(clean) if c == 'е']
                        vowel_positions = [j for j, c in enumerate(clean) if c in self.vowels]
                        yo_char_idx = e_positions[yo_pos - 1] if yo_pos - 1 < len(e_positions) else None

                        if yo_char_idx is None:
                            continue

                        has_stress = self.stress_token in raw
                        if has_stress:
                            stress_marker_idx = raw.lower().index(self.stress_token)
                            if stress_marker_idx + 1 < len(raw):
                                stressed_char = raw[stress_marker_idx + 1].lower()
                                if stressed_char == 'е':
                                    actual_stressed_char_idx = stress_marker_idx
                                    if actual_stressed_char_idx == yo_char_idx:
                                        result_tokens[tok_idx] = self._yo_replace_word(result_tokens[tok_idx], yo_pos - 1, is_char_pos=False)
                        else:
                            stress_pred = yo_stress_map.get(key, 0)
                            stressed_char_idx = vowel_positions[stress_pred] if stress_pred < len(vowel_positions) else None
                            if stressed_char_idx is not None and stressed_char_idx == yo_char_idx:
                                result_tokens[tok_idx] = self._yo_replace_word(result_tokens[tok_idx], yo_pos - 1, is_char_pos=False)

            # Apply stress marks
            if put_stress:
                for tok_idx, (clean, need) in enumerate(zip(clean_tokens, needs_processing)):
                    if not need or self.stress_token in result_tokens[tok_idx].lower():
                        continue

                    key = (sent_idx, tok_idx)
                    num_vowels = sum(1 for c in clean if c in self.vowels)

                    if num_vowels == 1 and not stress_single_vowel:
                        continue

                    if key in exception_stress:
                        result_tokens[tok_idx] = self._accentuate_word(result_tokens[tok_idx], exception_stress[key], is_char_pos=True)
                    elif key in stress_pred_map:
                        result_tokens[tok_idx] = self._accentuate_word(result_tokens[tok_idx], stress_pred_map[key], is_char_pos=False)

            results.append(''.join(result_tokens))

        return results
