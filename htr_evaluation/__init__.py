"""Utilities for evaluating PAGE XML / ALTO OCR output."""

from .evaluate import evaluate_folders, evaluate_htrflow_page, evaluate_pair

__all__ = ["evaluate_pair", "evaluate_folders", "evaluate_htrflow_page"]
