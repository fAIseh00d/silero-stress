#!/usr/bin/env python3
"""
Evaluate homograph resolution on the labeled corpus (tests/data/homograph_corpus_annotated.txt.gz).

Modes (NumPy/ONNX backend):
  whole      whole text in one call: all homographs padded into one BERT batch (= upstream)
  paragraph  one call per paragraph (typical realtime usage)
  solo       every homograph's BERT row run alone: no padding at all
  upstream   upstream torch output on the whole text (from the golden file, if present)

Reports accuracy vs gold for each mode, every prediction that differs between padded and
unpadded runs (with which one gold agrees with), and writes all errors to a TSV.

Usage:
    PYTHONPATH=src python scripts/eval_homograph_corpus.py [--errors errors.tsv]
"""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'tests'))

from homograph_corpus import GOLDEN, accuracy, load_corpus, predictions, read_gz  # noqa: E402


def run_paragraphs(acc, text):
    parts = re.split(r'(\n\s*\n)', text)  # keep separators so offsets stay aligned
    return ''.join(acc(p) if i % 2 == 0 else p for i, p in enumerate(parts))


def run_solo(acc, text):
    """Like whole-text, but BERT sees each homograph row alone (no padding)."""
    hs = acc.homosolver
    orig = hs._run_model
    hs._run_model = lambda marks: [orig([m])[0] for m in marks]
    try:
        return acc(text)
    finally:
        hs._run_model = orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--errors', default=None, help='write per-occurrence errors (TSV)')
    args = ap.parse_args()

    from silero_stress._impl import AccentorRu
    acc = AccentorRu(use_homosolver=True, device='cpu')
    homodict = acc.homosolver.homodict
    text, gold = load_corpus(homodict)
    print(f"Corpus: {len(text)} chars, {len(gold)} labeled occurrences, "
          f"{len({k for _, k, _ in gold})}/{len(homodict)} homographs labeled")

    outputs = {
        'whole': acc(text),
        'paragraph': run_paragraphs(acc, text),
        'solo': run_solo(acc, text),
    }
    if GOLDEN.exists():
        outputs['upstream'] = read_gz(GOLDEN)

    preds = {name: predictions(out, gold) for name, out in outputs.items()}

    print(f"\n{'mode':10s} {'accuracy':>10s}  errors")
    for name, p in preds.items():
        ok, n, errs = accuracy(p, gold)
        print(f"{name:10s} {100 * ok / n:9.2f}%  {len(errs)}")
    if 'upstream' in preds:
        same = preds['upstream'] == preds['whole']
        print(f"\nwhole (NumPy) vs upstream torch: {'identical' if same else 'DIFFERENT'}")

    # Padding sensitivity: whole/paragraph (padded) vs solo (unpadded)
    for name in ('whole', 'paragraph'):
        diffs = [i for i, (a, b) in enumerate(zip(preds[name], preds['solo'])) if a != b]
        verdict = Counter('padded right' if preds[name][i] == gold[i][2] else
                          'solo right' if preds['solo'][i] == gold[i][2] else 'both wrong'
                          for i in diffs)
        print(f"\n{name} vs solo: {len(diffs)} differing predictions {dict(verdict)}")
        for i in diffs[:30]:
            start, key, g = gold[i]
            ctx = text[max(0, start - 60):start + 40].replace('\n', ' ')
            print(f"  {key}: gold={g} {name}={preds[name][i]} solo={preds['solo'][i]}  …{ctx}…")

    # Most frequent errors (whole text)
    _, _, errs = accuracy(preds['whole'], gold)
    by_word = Counter(k for _, k, _, _ in errs)
    print(f"\nWords with most errors (whole): {by_word.most_common(20)}")

    if args.errors:
        with open(args.errors, 'w', encoding='utf-8') as f:
            f.write("idx\tword\tgold\tpred\tcontext\n")
            for i, k, g, p in errs:
                start = gold[i][0]
                ctx = text[max(0, start - 100):start + 60].replace('\n', ' ').replace('\t', ' ')
                f.write(f"{i}\t{k}\t{g}\t{p}\t{ctx}\n")
        print(f"Errors written to {args.errors}")


if __name__ == '__main__':
    main()
