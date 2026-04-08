# htr-evaluation

Small Python package to evaluate HTR/OCR output stored as **ALTO XML** or **PAGE XML**.

It supports:
- single prediction/ground-truth pair evaluation
- folder evaluation where files are matched by stem (e.g. `page_001.xml` with `page_001.xml`)
- in folder mode, prediction and GT must also share the same XML format (`page` or `alto`)
- per-page and aggregate metrics

## Metrics
- **CER** (character error rate): Levenshtein edit distance divided by number of GT characters
- **BoW Precision / Recall / F1**: case-insensitive bag-of-words overlap using token multiplicity

## Install

```bash
pip install -e .
```

## CLI usage

Single pair:

```bash
htr-eval --pred-file path/to/pred.xml --gt-file path/to/gt.xml
```

Folder mode:

```bash
htr-eval --pred-dir path/to/preds --gt-dir path/to/gts
```

## Python usage

```python
from htr_evaluation import evaluate_pair, evaluate_folders

pair = evaluate_pair("pred.xml", "gt.xml")
print(pair.cer, pair.bow_f1)

corpus = evaluate_folders("pred_dir", "gt_dir")
print(corpus["aggregate"])
print(corpus["format_mismatches"])
```
