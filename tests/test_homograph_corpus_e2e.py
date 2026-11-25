#!/usr/bin/env python3
"""
End-to-end test on one long coherent text with a stress-labeled occurrence of EVERY
homodict word (tests/data/homograph_corpus_annotated.txt.gz, see tests/homograph_corpus.py).

- parity: NumPy/ONNX on the whole text in ONE call (all homographs in one padded BERT
  batch, exactly like upstream) == upstream torch output stored in
  tests/data/homograph_corpus.expected.txt.gz (scripts/make_e2e_golden.py)
- accuracy vs the labels must not regress below ACCURACY_FLOOR
  (detailed report: scripts/eval_homograph_corpus.py)

Re-running upstream torch on the whole text takes minutes (quadratic context cleaning):
    SILERO_SLOW_TESTS=1 PYTHONPATH=src pytest tests/test_homograph_corpus_e2e.py
"""

import difflib
import os
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

from homograph_corpus import ANNOTATED, GOLDEN, accuracy, load_corpus, predictions, read_gz  # noqa: E402

# Set from the first full run against the labeled corpus; raise it when the algorithm improves.
ACCURACY_FLOOR = 0.79  # 79.6% on 2026-11-25 (v1.5 weights); upstream torch: 79.5%

try:
    import onnxruntime  # noqa: F401
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _first_diffs(expected, actual, n=10):
    """Word-level diff summary for readable failures."""
    exp_words, act_words = expected.split(), actual.split()
    out = []
    sm = difflib.SequenceMatcher(a=exp_words, b=act_words, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != 'equal':
            ctx = ' '.join(exp_words[max(0, i1 - 4):i1])
            out.append(f"  …{ctx} [{' '.join(exp_words[i1:i2])}] → [{' '.join(act_words[j1:j2])}]")
            if len(out) >= n:
                break
    return '\n'.join(out)


def _upstream_bug(torch_tok):
    """Token-level deviations where upstream is wrong and we deliberately differ.

    1. ё on an unstressed vowel (upstream n-gram ё check), e.g. "зап+оём", "вор+оньё"
    2. exception-dict words with leading punctuation: upstream indexes the raw token with
       clean-word positions and shifts the stress or corrupts letters, e.g. "«+Деньги",
       "«Те+бе", "«Бел+ёе" (we give "«Д+еньги", "«Теб+е", "«Бель+ё")
    """
    return bool(re.search(r'(?<!\+)[ёЁ]', torch_tok)) or bool(re.match(r'\W', torch_tok.replace('+', '')))


@unittest.skipUnless(ANNOTATED.exists(), "labeled corpus missing")
class TestHomographCorpusE2E(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from silero_stress._impl import _load_gzip_json
        cls.homodict = _load_gzip_json('ru', 'homodict_ru.json.gz')
        cls.text, cls.gold = load_corpus(cls.homodict)
        cls.expected = read_gz(GOLDEN) if GOLDEN.exists() else None
        cls._numpy_out = None

    @classmethod
    def numpy_output(cls):
        if cls._numpy_out is None:
            from silero_stress._impl import AccentorRu
            cls._numpy_out = AccentorRu(use_homosolver=True, device='cpu')(cls.text)
        return cls._numpy_out

    def test_corpus_matches_sources(self):
        """Committed corpus == scripts/build_homograph_corpus.py output (sources edited -> rebuild)."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'build_homograph_corpus', Path(__file__).parent.parent / 'scripts' / 'build_homograph_corpus.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        corpus, _ = mod.build(self.homodict)
        self.assertEqual(corpus, read_gz(ANNOTATED), "rerun scripts/build_homograph_corpus.py")

    def test_every_homograph_covered(self):
        """Every homodict word occurs in the text; >= 99% have a labeled occurrence
        (ambiguous occurrences are deliberately left unlabeled)."""
        present = {m.group().lower() for m in re.finditer(r'[а-яёА-ЯЁ]+', self.text)}
        self.assertEqual(sorted(set(self.homodict) - present), [])
        labeled = {k for _, k, _ in self.gold}
        self.assertGreaterEqual(len(labeled) / len(self.homodict), 0.99)

    def test_golden_is_consistent(self):
        """Golden output is the plain corpus plus stress marks / ё only
        (except tokens hit by upstream's quote-offset bug, which corrupts letters)."""
        if self.expected is None:
            self.skipTest("golden file missing (scripts/make_e2e_golden.py)")
        strip = lambda s: s.replace('+', '').replace('ё', 'е').replace('Ё', 'Е')
        exp_toks, txt_toks = self.expected.split(), self.text.split()
        self.assertEqual(len(exp_toks), len(txt_toks))
        bad = [(e, t) for e, t in zip(exp_toks, txt_toks) if strip(e) != strip(t) and not _upstream_bug(e)]
        self.assertEqual(bad, [])

    @unittest.skipUnless(HAS_ONNX, "ONNX Runtime required")
    def test_numpy_whole_text_matches_upstream(self):
        """NumPy/ONNX backend, whole text in one call == upstream golden (bit-exact)."""
        if self.expected is None:
            self.skipTest("golden file missing (scripts/make_e2e_golden.py)")
        actual = self.numpy_output()
        exp_toks, act_toks = self.expected.split(), actual.split()
        self.assertEqual(len(exp_toks), len(act_toks), "token count differs")
        diffs = [(e, a) for e, a in zip(exp_toks, act_toks) if e != a]
        unexplained = [(e, a) for e, a in diffs if not _upstream_bug(e)]
        print(f"\n{len(diffs)} token diffs vs upstream, all known upstream bugs: {not unexplained}")
        if unexplained:
            self.fail(f"{len(unexplained)} unexplained diffs vs upstream golden: {unexplained[:20]}")

    @unittest.skipUnless(HAS_ONNX, "ONNX Runtime required")
    def test_accuracy_floor(self):
        ok, n, errors = accuracy(predictions(self.numpy_output(), self.gold), self.gold)
        acc = ok / n
        print(f"\nHomograph accuracy on labeled corpus: {ok}/{n} ({100 * acc:.2f}%)")
        if ACCURACY_FLOOR is None:
            self.skipTest(f"ACCURACY_FLOOR not set yet (current: {acc:.4f})")
        self.assertGreaterEqual(acc, ACCURACY_FLOOR, f"first errors: {errors[:10]}")

    @unittest.skipUnless(HAS_TORCH and os.environ.get('SILERO_SLOW_TESTS'), "set SILERO_SLOW_TESTS=1 (takes minutes)")
    def test_torch_whole_text_matches_golden(self):
        """Upstream torch still reproduces the golden (catches a stale golden after model updates)."""
        if self.expected is None:
            self.skipTest("golden file missing")
        from silero_stress._torch_impl import TorchAccentorRu
        actual = TorchAccentorRu(use_homosolver=True, device='cpu')(self.text)
        if actual != self.expected:
            self.fail("Torch output differs from golden (regenerate with scripts/make_e2e_golden.py):\n"
                      + _first_diffs(self.expected, actual))


if __name__ == '__main__':
    unittest.main()
