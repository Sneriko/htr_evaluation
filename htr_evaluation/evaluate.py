from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .htrflow import extract_text_from_htrflow_json
from .metrics import bow_details, cer_details
from .parsers import detect_xml_format, extract_text


@dataclass
class PageMetrics:
    stem: str
    xml_format: str
    prediction_file: str
    gt_file: str
    cer: float
    edit_distance: int
    gt_char_count: int
    bow_precision: float
    bow_recall: float
    bow_f1: float
    bow_true_positive: int
    bow_false_positive: int
    bow_false_negative: int
    relative_stem: str | None = None
    subfolder: str = "."


@dataclass
class AggregateMetrics:
    pair_count: int
    cer: float
    edit_distance: int
    gt_char_count: int
    bow_precision: float
    bow_recall: float
    bow_f1: float
    bow_true_positive: int
    bow_false_positive: int
    bow_false_negative: int


def _evaluate_texts(
    *,
    pred_text: str,
    gt_text: str,
    stem: str,
    input_format: str,
    prediction_file: str | Path,
    gt_file: str | Path,
) -> PageMetrics:
    cer, distance, gt_chars = cer_details(pred_text, gt_text)
    precision, recall, f1, tp, fp, fn = bow_details(pred_text, gt_text)

    return PageMetrics(
        stem=stem,
        xml_format=input_format,
        prediction_file=str(prediction_file),
        gt_file=str(gt_file),
        cer=cer,
        edit_distance=distance,
        gt_char_count=gt_chars,
        bow_precision=precision,
        bow_recall=recall,
        bow_f1=f1,
        bow_true_positive=tp,
        bow_false_positive=fp,
        bow_false_negative=fn,
    )


def evaluate_pair(prediction_file: str | Path, gt_file: str | Path) -> PageMetrics:
    pred_format = detect_xml_format(prediction_file)
    gt_format = detect_xml_format(gt_file)
    xml_format = pred_format if pred_format == gt_format else f"{pred_format}_vs_{gt_format}"

    return _evaluate_texts(
        pred_text=extract_text(prediction_file),
        gt_text=extract_text(gt_file),
        stem=Path(prediction_file).stem,
        input_format=xml_format,
        prediction_file=prediction_file,
        gt_file=gt_file,
    )


def evaluate_htrflow_page(prediction_json: str | Path, gt_page_file: str | Path) -> PageMetrics:
    """Evaluate a full-page HTRFlow JSON prediction against PAGE XML ground truth."""
    gt_format = detect_xml_format(gt_page_file)
    if gt_format != "page":
        raise ValueError(f"Expected PAGE XML ground truth, got {gt_format!r}: {gt_page_file}")

    return _evaluate_texts(
        pred_text=extract_text_from_htrflow_json(prediction_json),
        gt_text=extract_text(gt_page_file),
        stem=Path(prediction_json).stem,
        input_format="htrflow_json_vs_page",
        prediction_file=prediction_json,
        gt_file=gt_page_file,
    )


def _index_xml_files(directory: str | Path) -> dict[str, dict[str, Path]]:
    root = Path(directory)
    files = sorted(root.rglob("*.xml"))
    indexed: dict[str, dict[str, Path]] = {}

    for path in files:
        relative_stem = path.relative_to(root).with_suffix("").as_posix()
        xml_format = detect_xml_format(path)
        stem_bucket = indexed.setdefault(relative_stem, {})

        if xml_format in stem_bucket:
            raise ValueError(
                f"Duplicate relative path/format detected for '{relative_stem}' "
                f"({xml_format}): {stem_bucket[xml_format]} and {path}"
            )
        stem_bucket[xml_format] = path

    return indexed


def _resolve_pair(stem: str, pred_variants: dict[str, Path], gt_variants: dict[str, Path]) -> tuple[str, Path, Path] | None:
    shared_formats = sorted(set(pred_variants) & set(gt_variants))
    if shared_formats:
        fmt = shared_formats[0]
        return fmt, pred_variants[fmt], gt_variants[fmt]

    return None


def _aggregate_pages(pages: list[PageMetrics]) -> AggregateMetrics:
    total_distance = sum(page.edit_distance for page in pages)
    total_gt_chars = sum(page.gt_char_count for page in pages)
    total_tp = sum(page.bow_true_positive for page in pages)
    total_fp = sum(page.bow_false_positive for page in pages)
    total_fn = sum(page.bow_false_negative for page in pages)

    cer = total_distance / total_gt_chars if total_gt_chars > 0 else 0.0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return AggregateMetrics(
        pair_count=len(pages),
        cer=cer,
        edit_distance=total_distance,
        gt_char_count=total_gt_chars,
        bow_precision=precision,
        bow_recall=recall,
        bow_f1=f1,
        bow_true_positive=total_tp,
        bow_false_positive=total_fp,
        bow_false_negative=total_fn,
    )


def _subfolder_for_relative_stem(relative_stem: str) -> str:
    parent = Path(relative_stem).parent.as_posix()
    return parent if parent != "." else "."


def evaluate_folders(prediction_dir: str | Path, gt_dir: str | Path) -> dict[str, object]:
    pred_files = _index_xml_files(prediction_dir)
    gt_files = _index_xml_files(gt_dir)

    common_stems = sorted(set(pred_files) & set(gt_files))
    if not common_stems:
        raise ValueError(
            "No matching XML file relative paths were found between prediction_dir and gt_dir."
        )

    per_page: list[PageMetrics] = []
    format_mismatches: list[str] = []

    for relative_stem in common_stems:
        resolved = _resolve_pair(
            relative_stem, pred_files[relative_stem], gt_files[relative_stem]
        )
        if resolved is None:
            format_mismatches.append(relative_stem)
            continue

        xml_format, pred_path, gt_path = resolved
        page = evaluate_pair(pred_path, gt_path)
        page.xml_format = xml_format
        page.relative_stem = relative_stem
        page.subfolder = _subfolder_for_relative_stem(relative_stem)
        per_page.append(page)

    if not per_page:
        raise ValueError(
            "No matching prediction/ground-truth pairs with compatible XML format were found."
        )

    subfolder_names = sorted({page.subfolder for page in per_page})
    per_subfolder = [
        {
            "subfolder": subfolder,
            "aggregate": asdict(
                _aggregate_pages([page for page in per_page if page.subfolder == subfolder])
            ),
        }
        for subfolder in subfolder_names
    ]

    return {
        "aggregate": asdict(_aggregate_pages(per_page)),
        "per_subfolder": per_subfolder,
        "per_page": [asdict(p) for p in per_page],
        "format_mismatches": format_mismatches,
        "missing_predictions": sorted(set(gt_files) - set(pred_files)),
        "missing_ground_truth": sorted(set(pred_files) - set(gt_files)),
    }
