#!/usr/bin/env python3
"""
Build the labeled homograph corpus (tests/data/homograph_corpus_annotated.txt.gz)
from its sources in tests/data/corpus_build/:

  pass1_part{1..4}.txt.gz   first annotation pass: the written text with every
                            homodict occurrence labeled ("зам+ок", "вс+ё")
  pass2_part{1..4}.txt.gz   independent blind pass over the same text (labels stripped)
  tiebreak_answers.tsv      third blind annotator, only for occurrences where pass 1
                            and pass 2 disagree: "id<TAB>variant"; ids enumerate the
                            disagreements in text order over parts 1..4
  tiebreak_unsure.txt       tie-break ids the annotator flagged as unsure -> excluded
  overrides.json            final manual verdicts, located by a unique context string:
                            {"word", "context", "label": "variant" | null, "note"}
                            (null = ambiguous -> excluded from scoring)

Label of each occurrence:
  pass1 == pass2            -> that label (consensus)
  otherwise                 -> tie-break answer (2-of-3 majority), or unlabeled if unsure
  then overrides            -> label replaced, or unlabeled if label is null

Unlabeled occurrences stay in the text (they count for upstream parity, not for
accuracy). The model is never used to decide a label.

Usage:
    PYTHONPATH=src python scripts/build_homograph_corpus.py [--check]
"""

import argparse
import collections
import gzip
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
SRC = ROOT / 'tests' / 'data' / 'corpus_build'
OUT = ROOT / 'tests' / 'data' / 'homograph_corpus_annotated.txt.gz'
PARTS = (1, 2, 3, 4)
TOKEN = re.compile(r'[а-яёА-ЯЁ+]+')


def key(token):
    return token.lower().replace('+', '').replace('ё', 'е')


def apply_label(token, label):
    """Write label (e.g. 'з+амок', 'вс+ё') onto token, keeping its capitalization.

    label=None -> plain homodict form (stress removed, ё -> е).
    """
    plain = token.replace('+', '')
    if label is None:
        return ''.join(c.upper() if p.isupper() else c for p, c in zip(plain, key(token)))
    out, j = [], 0
    for c in label:
        if c == '+':
            out.append('+')
            continue
        out.append(c.upper() if plain[j].isupper() else c)
        j += 1
    return ''.join(out)


def read_gz(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return f.read()


def labels(text, homodict):
    """[(match, label)] for every homodict occurrence; label None if unannotated."""
    return [(m, m.group().lower() if '+' in m.group() else None)
            for m in TOKEN.finditer(text) if key(m.group()) in homodict]


def build(homodict):
    answers = {}
    for line in open(SRC / 'tiebreak_answers.tsv', encoding='utf-8'):
        if line.strip() and not line.startswith('id\t'):
            i, v = line.rstrip('\n').split('\t')[:2]
            answers[int(i)] = v.strip().lower()
    unsure = {int(line.split()[0]) for line in open(SRC / 'tiebreak_unsure.txt', encoding='utf-8')
              if line.strip() and line.split()[0].isdigit()}
    overrides = json.load(open(SRC / 'overrides.json', encoding='utf-8'))

    stats = collections.Counter()
    dispute_id = 0
    parts = []
    for n in PARTS:
        p1_text = read_gz(SRC / f'pass1_part{n}.txt.gz')
        p2_text = read_gz(SRC / f'pass2_part{n}.txt.gz')
        p1, p2 = labels(p1_text, homodict), labels(p2_text, homodict)
        if [key(m.group()) for m, _ in p1] != [key(m.group()) for m, _ in p2]:
            sys.exit(f"part {n}: passes do not align")
        out, last = [], 0
        for (m, l1), (_, l2) in zip(p1, p2):
            if l1 == l2:
                label = l1
                stats['consensus'] += 1
            else:
                if dispute_id in unsure:
                    label = None
                    stats['disputed, tie-break unsure -> excluded'] += 1
                else:
                    label = answers[dispute_id]
                    side = 'pass1' if label == l1 else 'pass2' if label == l2 else None
                    if side is None:
                        sys.exit(f"tie-break id {dispute_id}: answer {label!r} matches neither pass")
                    stats[f'disputed, tie-break sided with {side}'] += 1
                dispute_id += 1
            out.append(p1_text[last:m.start()])
            out.append(apply_label(m.group(), label))
            last = m.end()
        out.append(p1_text[last:])
        parts.append(''.join(out).strip())
    if dispute_id != len(answers):
        sys.exit(f"{dispute_id} disagreements but {len(answers)} tie-break answers")
    corpus = '\n\n'.join(parts) + '\n'

    for o in overrides:
        # context matches regardless of annotations ('+', е/ё) in the corpus
        flex = lambda s: ''.join(('[её]' if c in 'её' else re.escape(c)) + (r'\+?' if c.isalpha() else '')
                                 for c in s)
        hits = list(re.finditer(r'\+?' + flex(o['context']), corpus, re.I))
        if len(hits) != 1:
            sys.exit(f"override context {o['context']!r} matched {len(hits)} times")
        seg_start = hits[0].start()
        word = re.search(r'(?<![а-яёА-ЯЁ+])\+?' + flex(o['word']) + r'(?![а-яёА-ЯЁ])', hits[0].group(), re.I)
        if not word:
            sys.exit(f"override word {o['word']!r} not found in its context")
        s, e = seg_start + word.start(), seg_start + word.end()
        corpus = corpus[:s] + apply_label(corpus[s:e], o['label']) + corpus[e:]
        stats['override: ' + ('relabeled' if o['label'] else 'excluded')] += 1
    return corpus, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='verify the committed corpus is up to date')
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT / 'src'))
    from silero_stress._impl import _load_gzip_json
    homodict = _load_gzip_json('ru', 'homodict_ru.json.gz')

    corpus, stats = build(homodict)
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    if args.check:
        same = read_gz(OUT) == corpus
        print("corpus up to date" if same else "corpus OUT OF DATE: rerun without --check")
        sys.exit(0 if same else 1)
    with gzip.GzipFile(OUT, 'wb', mtime=0) as f:  # mtime=0: byte-identical rebuilds
        f.write(corpus.encode('utf-8'))
    print(f"wrote {OUT}")


if __name__ == '__main__':
    main()
