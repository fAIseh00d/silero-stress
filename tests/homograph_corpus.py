"""
Labeled homograph corpus helpers (shared by tests and scripts/eval_homograph_corpus.py).

tests/data/homograph_corpus_annotated.txt.gz is one long coherent Russian text in which
every occurrence of a homodict word carries its intended stress as a homodict variant,
e.g. "Старинный з+амок ... взломать зам+ок ... вс+ё". Nothing else has '+'.

From it we derive:
- plain text: annotated homographs replaced by their homodict key (stress removed,
  ё -> е, original capitalization kept); this is the model input
- gold labels: one (key, variant) per homograph occurrence, in text order
"""

import gzip
import re
from pathlib import Path

DATA = Path(__file__).parent / 'data'
ANNOTATED = DATA / 'homograph_corpus_annotated.txt.gz'
GOLDEN = DATA / 'homograph_corpus.expected.txt.gz'

_TOKEN = re.compile(r'[а-яёА-ЯЁ+]+')


def _key(token):
    return token.lower().replace('+', '').replace('ё', 'е')


def _restore_case(src, dst):
    """Apply src's per-char case to dst (same length, no '+')."""
    return ''.join(d.upper() if s.isupper() else d for s, d in zip(src, dst))


def read_gz(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return f.read()


def load_corpus(homodict, path=ANNOTATED):
    """Return (plain_text, gold).

    gold = [(plain_start, key, variant), ...] for every ANNOTATED occurrence, in text order.
    Homographs written without '+' (e.g. another chunk's list words used in passing)
    stay in the text unlabeled: they count for parity, not for accuracy.
    """
    annotated = read_gz(path)
    gold, parts = [], []
    last = plain_len = 0
    for m in _TOKEN.finditer(annotated):
        tok = m.group()
        if '+' not in tok:
            continue
        key = _key(tok)
        variant = tok.lower()
        if key not in homodict:
            raise ValueError(f"'+' on a non-homograph token {tok!r}")
        if variant not in homodict[key]:
            raise ValueError(f"Invalid annotation {tok!r}; allowed {homodict[key]}")
        chunk = annotated[last:m.start()]
        parts.append(chunk)
        plain_len += len(chunk)
        gold.append((plain_len, key, variant))
        plain_tok = _restore_case(tok.replace('+', ''), key)
        parts.append(plain_tok)
        plain_len += len(plain_tok)
        last = m.end()
    parts.append(annotated[last:])
    return ''.join(parts), gold


def predictions(output, gold):
    """Predicted variant for each gold occurrence (None if the model left it unstressed).

    Model output equals the input plus inserted '+' (and same-length е->ё changes), so an
    output token maps back to its input offset by subtracting the '+' signs before it.
    """
    wanted = {start: i for i, (start, _, _) in enumerate(gold)}
    preds = [None] * len(gold)
    plus_before = 0
    pos = 0
    for m in _TOKEN.finditer(output):
        plus_before += output.count('+', pos, m.start())
        pos = m.start()
        # a leading '+' is part of the token, so it is not counted in plus_before
        i = wanted.get(m.start() - plus_before)
        if i is not None:
            tok = m.group()
            preds[i] = tok.lower() if '+' in tok else None
    return preds


def accuracy(preds, gold):
    """Return (n_correct, n_total, errors) with errors = [(idx, key, gold_variant, pred)]."""
    errors = [(i, k, g, p) for i, ((_, k, g), p) in enumerate(zip(gold, preds)) if p != g]
    return len(gold) - len(errors), len(gold), errors
