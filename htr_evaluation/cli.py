from __future__ import annotations

import argparse
import json

from .evaluate import evaluate_folders, evaluate_pair


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate PAGE XML or ALTO predictions using CER and bag-of-words precision/recall/F1."
        )
    )

    parser.add_argument("--pred-file", help="Path to a single prediction XML file")
    parser.add_argument("--gt-file", help="Path to a single ground-truth XML file")
    parser.add_argument("--pred-dir", help="Directory with prediction XML files")
    parser.add_argument("--gt-dir", help="Directory with ground-truth XML files")
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation for output (default: 2)",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    pair_mode = args.pred_file and args.gt_file
    folder_mode = args.pred_dir and args.gt_dir

    if pair_mode and folder_mode:
        parser.error("Use either --pred-file/--gt-file or --pred-dir/--gt-dir, not both.")

    if pair_mode:
        result = evaluate_pair(args.pred_file, args.gt_file).__dict__
    elif folder_mode:
        result = evaluate_folders(args.pred_dir, args.gt_dir)
    else:
        parser.error("Provide either file pair args or folder pair args.")

    print(json.dumps(result, indent=args.indent, ensure_ascii=False))


if __name__ == "__main__":
    main()
