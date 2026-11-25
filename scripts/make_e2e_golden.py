#!/usr/bin/env python3
"""
Regenerate the golden output for the homograph end-to-end test.

The corpus (tests/data/homograph_corpus_annotated.txt.gz) is one long coherent Russian text
with a stress-labeled occurrence of every homodict word; the model input is that text with
the labels stripped (tests/homograph_corpus.py). The golden file is the output of the ORIGINAL upstream
model (torch package, via TorchAccentorRu) on the whole text in a single call, i.e. all
homographs go through one padded BERT batch, exactly like upstream.

Upstream's context cleaning is O(text_len * n_homographs), so this takes a few minutes.

Usage:
    PYTHONPATH=src python scripts/make_e2e_golden.py
"""

import gzip
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'tests'))
from homograph_corpus import GOLDEN, load_corpus  # noqa: E402


def main():
    from silero_stress._torch_impl import TorchAccentorRu

    homodict = json.load(gzip.open(ROOT / 'src/silero_stress/data/ru/homodict_ru.json.gz', 'rt', encoding='utf-8'))
    text, gold = load_corpus(homodict)  # plain model input (annotations stripped)
    missing = sorted(set(homodict) - {k for _, k, _ in gold})
    if missing:
        sys.exit(f"Corpus has no labeled occurrence for {len(missing)} homographs: {' '.join(missing[:50])}")

    acc = TorchAccentorRu(use_homosolver=True, device='cpu')
    print(f"Running upstream model on {len(text)} chars ...")
    t0 = time.time()
    out = acc(text)
    print(f"  done in {time.time() - t0:.1f}s")

    # mtime=0 keeps the .gz byte-identical across regenerations (clean git diffs)
    with gzip.GzipFile(GOLDEN, 'wb', mtime=0) as f:
        f.write(out.encode('utf-8'))
    print(f"Saved {GOLDEN}")


if __name__ == '__main__':
    main()
