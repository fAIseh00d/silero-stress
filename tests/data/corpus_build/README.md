# Homograph corpus sources

Sources of `tests/data/homograph_corpus_annotated.txt.gz`, a ~262K-character coherent Russian
text (four connected stories) in which every one of the 2208 homodict words occurs, with
stress-labeled occurrences used as ground truth by `tests/test_homograph_corpus_e2e.py` and
`scripts/eval_homograph_corpus.py`.

| File | Content |
|---|---|
| `pass1_part{1..4}.txt.gz` | The written text, first annotation pass: every homodict occurrence labeled with its variant (`зам+ок`, `вс+ё`) |
| `pass2_part{1..4}.txt.gz` | Independent blind pass over the same text with labels stripped |
| `tiebreak_answers.tsv` | Third blind annotator, only for the 911 occurrences where pass 1 and pass 2 disagree (`id<TAB>variant`, ids enumerate disagreements in text order) |
| `tiebreak_unsure.txt` | Tie-break ids flagged as unsure (often both stresses standard); excluded from scoring |
| `overrides.json` | Final manual verdicts by unique context: a corrected label, or `null` = ambiguous (excluded) |

Final label: consensus of passes 1 and 2, else the tie-break's 2-of-3 majority, then overrides.
Excluded occurrences stay in the text unlabeled: they count for upstream parity, not accuracy.
The model is never used to decide a label.

Text and annotations were produced by LLM agents (Claude); passes 2 and 3 never saw the other
labels. Known caveat: annotators of the same model family share some blind spots, and the text
deliberately exercises both readings of every word, so accuracy here is lower than on natural
text (where the frequent reading dominates).

To correct a label, add an entry to `overrides.json` and rebuild:

```bash
PYTHONPATH=src python scripts/build_homograph_corpus.py          # rebuild
PYTHONPATH=src python scripts/build_homograph_corpus.py --check  # verify up to date
PYTHONPATH=src python scripts/eval_homograph_corpus.py           # accuracy report
```

Label changes do not affect the upstream golden output (`homograph_corpus.expected.txt.gz`);
only changes to the text itself require `scripts/make_e2e_golden.py` (about 9 minutes).
