# htr-evaluation

Small Python package to evaluate HTR/OCR output stored as **ALTO XML** or **PAGE XML**.

It supports:
- single prediction/ground-truth pair evaluation
- full-page HTRFlow output JSON prediction vs. PAGE XML ground truth evaluation
- recursive folder evaluation where files are matched by relative path and stem (e.g. `volume_1/page_001.xml` with `volume_1/page_001.xml`)
- in folder mode, prediction and GT must also share the same XML format (`page` or `alto`)
- per-page, per-subfolder, and whole-folder aggregate metrics

## Metrics
- **CER** (character error rate): Levenshtein edit distance divided by number of GT characters
- **BoW Precision / Recall / F1**: case-insensitive bag-of-words overlap using token multiplicity

## Install

```bash
pip install -e .
```

## CLI usage

Single XML pair:

```bash
htr-eval --pred-file path/to/pred.xml --gt-file path/to/gt.xml
```

Full-page HTRFlow JSON prediction against PAGE XML ground truth:

```bash
htr-eval --pred-htrflow-json path/to/htrflow-output.json --gt-page-file path/to/gt.xml
# equivalent console entry point
htrflow-page-eval --pred-htrflow-json path/to/htrflow-output.json --gt-page-file path/to/gt.xml
```

Folder mode recursively evaluates every XML file under the prediction and ground-truth directories. Files are paired by their path relative to each root without the `.xml` suffix, so `preds/chapter_1/page_001.xml` is compared with `gts/chapter_1/page_001.xml`. The JSON output contains:

- `aggregate`: whole-folder metrics across all matched pages
- `per_subfolder`: aggregate metrics for each relative subfolder
- `per_page`: metrics for each matched page
- `format_mismatches`, `missing_predictions`, and `missing_ground_truth`: unmatched or incompatible inputs

```bash
htr-eval --pred-dir path/to/preds --gt-dir path/to/gts
```

Write any evaluation result to a JSON file instead of stdout:

```bash
htr-eval --pred-dir path/to/preds --gt-dir path/to/gts --output-file results.json
```

## Python usage

```python
from htr_evaluation import evaluate_folders, evaluate_htrflow_page, evaluate_pair

pair = evaluate_pair("pred.xml", "gt.xml")
print(pair.cer, pair.bow_f1)

htrflow_page = evaluate_htrflow_page("htrflow-output.json", "gt-page.xml")
print(htrflow_page.cer, htrflow_page.bow_f1)

corpus = evaluate_folders("pred_dir", "gt_dir")
print(corpus["aggregate"])       # whole-folder metrics
print(corpus["per_subfolder"])   # metrics grouped by relative subfolder
print(corpus["per_page"])        # metrics for each page
print(corpus["format_mismatches"])
```
