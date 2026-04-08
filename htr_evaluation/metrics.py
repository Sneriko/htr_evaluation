from __future__ import annotations

from collections import Counter
import re


_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)


def levenshtein_distance(a: str, b: str) -> int:
    """Compute Levenshtein edit distance using a two-row DP table."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    if len(a) > len(b):
        a, b = b, a

    previous = list(range(len(a) + 1))
    for i, b_char in enumerate(b, start=1):
        current = [i]
        for j, a_char in enumerate(a, start=1):
            insertions = previous[j] + 1
            deletions = current[j - 1] + 1
            substitutions = previous[j - 1] + (a_char != b_char)
            current.append(min(insertions, deletions, substitutions))
        previous = current
    return previous[-1]


def cer_details(pred_text: str, gt_text: str) -> tuple[float, int, int]:
    """
    Return (cer, edit_distance, gt_char_count).
    """
    distance = levenshtein_distance(pred_text, gt_text)
    gt_len = len(gt_text)
    cer = distance / gt_len if gt_len > 0 else 0.0
    return cer, distance, gt_len


def tokenize_for_bow(text: str) -> list[str]:
    return [token.lower() for token in _WORD_RE.findall(text)]


def bow_details(pred_text: str, gt_text: str) -> tuple[float, float, float, int, int, int]:
    """
    Return (precision, recall, f1, true_positive, false_positive, false_negative).

    Tokens are case-insensitive and counted as a multiset (bag of words).
    """
    pred_counts = Counter(tokenize_for_bow(pred_text))
    gt_counts = Counter(tokenize_for_bow(gt_text))

    tp = sum((pred_counts & gt_counts).values())
    fp = sum((pred_counts - gt_counts).values())
    fn = sum((gt_counts - pred_counts).values())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return precision, recall, f1, tp, fp, fn
