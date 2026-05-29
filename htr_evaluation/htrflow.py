from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _best_text(text_result: dict[str, Any]) -> str | None:
    texts = text_result.get("texts")
    if not isinstance(texts, list) or not texts:
        return None

    string_candidates = [text for text in texts if isinstance(text, str)]
    if not string_candidates:
        return None

    scores = text_result.get("scores")
    if isinstance(scores, list) and len(scores) == len(texts):
        best_index: int | None = None
        best_score: float | None = None
        for index, score in enumerate(scores):
            if not isinstance(texts[index], str):
                continue
            try:
                numeric_score = float(score)
            except (TypeError, ValueError):
                continue
            if best_score is None or numeric_score > best_score:
                best_index = index
                best_score = numeric_score
        if best_index is not None:
            return texts[best_index].strip()

    return string_candidates[0].strip()


def _collect_text_lines(node: Any, lines: list[str]) -> None:
    if isinstance(node, dict):
        text_result = node.get("text_result")
        if isinstance(text_result, dict):
            text = _best_text(text_result)
            if text:
                lines.append(text)

        contains = node.get("contains")
        if isinstance(contains, list):
            for child in contains:
                _collect_text_lines(child, lines)
    elif isinstance(node, list):
        for child in node:
            _collect_text_lines(child, lines)


def extract_text_from_htrflow_json(json_path: str | Path) -> str:
    """
    Extract page text from an HTRFlow output JSON document.

    The expected HTRFlow page shape contains nested ``contains`` lists whose
    text-line objects include a ``text_result`` mapping with ``texts`` and
    optional ``scores``. The traversal preserves the order in the JSON, which
    corresponds to HTRFlow's page/region/line reading order in the bundled
    examples.
    """
    path = Path(json_path)
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)

    lines: list[str] = []
    _collect_text_lines(payload, lines)
    return "\n".join(lines)
