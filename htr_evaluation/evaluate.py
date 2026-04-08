from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

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


def evaluate_pair(prediction_file: str | Path, gt_file: str | Path) -> PageMetrics:
    pred_format = detect_xml_format(prediction_file)
    gt_format = detect_xml_format(gt_file)
    xml_format = pred_format if pred_format == gt_format else f"{pred_format}_vs_{gt_format}"

    pred_text = extract_text(prediction_file)
    gt_text = extract_text(gt_file)

    cer, distance, gt_chars = cer_details(pred_text, gt_text)
    precision, recall, f1, tp, fp, fn = bow_details(pred_text, gt_text)

    return PageMetrics(
        stem=Path(prediction_file).stem,
        xml_format=xml_format,
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


def _index_xml_files(directory: str | Path) -> dict[str, dict[str, Path]]:
    files = sorted(Path(directory).rglob("*.xml"))
    indexed: dict[str, dict[str, Path]] = {}

    for path in files:
        stem = path.stem
        xml_format = detect_xml_format(path)
        stem_bucket = indexed.setdefault(stem, {})

        if xml_format in stem_bucket:
            raise ValueError(
                f"Duplicate stem/format detected for '{stem}' ({xml_format}): "
                f"{stem_bucket[xml_format]} and {path}"
            )
        stem_bucket[xml_format] = path

    return indexed


def _resolve_pair(stem: str, pred_variants: dict[str, Path], gt_variants: dict[str, Path]) -> tuple[str, Path, Path] | None:
    shared_formats = sorted(set(pred_variants) & set(gt_variants))
    if shared_formats:
        fmt = shared_formats[0]
        return fmt, pred_variants[fmt], gt_variants[fmt]

    return None


def evaluate_folders(prediction_dir: str | Path, gt_dir: str | Path) -> dict[str, object]:
    pred_files = _index_xml_files(prediction_dir)
    gt_files = _index_xml_files(gt_dir)

    common_stems = sorted(set(pred_files) & set(gt_files))
    if not common_stems:
        raise ValueError("No matching XML file stems were found between prediction_dir and gt_dir.")

    per_page: list[PageMetrics] = []
    format_mismatches: list[str] = []

    total_distance = 0
    total_gt_chars = 0
    total_tp = total_fp = total_fn = 0

    for stem in common_stems:
        resolved = _resolve_pair(stem, pred_files[stem], gt_files[stem])
        if resolved is None:
            format_mismatches.append(stem)
            continue

        xml_format, pred_path, gt_path = resolved
        page = evaluate_pair(pred_path, gt_path)
        page.xml_format = xml_format
        per_page.append(page)

        total_distance += page.edit_distance
        total_gt_chars += page.gt_char_count
        total_tp += page.bow_true_positive
        total_fp += page.bow_false_positive
        total_fn += page.bow_false_negative

    if not per_page:
        raise ValueError(
            "No matching prediction/ground-truth pairs with compatible XML format were found."
        )

    cer = total_distance / total_gt_chars if total_gt_chars > 0 else 0.0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    aggregate = AggregateMetrics(
        pair_count=len(per_page),
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

    return {
        "aggregate": asdict(aggregate),
        "per_page": [asdict(p) for p in per_page],
        "format_mismatches": format_mismatches,
        "missing_predictions": sorted(set(gt_files) - set(pred_files)),
        "missing_ground_truth": sorted(set(pred_files) - set(gt_files)),
    }
