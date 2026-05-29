from __future__ import annotations

import argparse
import json

from .evaluate import evaluate_folders, evaluate_htrflow_page, evaluate_pair


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate PAGE XML, ALTO XML, or HTRFlow JSON predictions using "
            "CER and bag-of-words precision/recall/F1."
        )
    )

    parser.add_argument("--pred-file", help="Path to a single prediction XML file")
    parser.add_argument("--gt-file", help="Path to a single ground-truth XML file")
    parser.add_argument("--pred-dir", help="Directory with prediction XML files")
    parser.add_argument("--gt-dir", help="Directory with ground-truth XML files")
    parser.add_argument(
        "--pred-htrflow-json",
        help="Path to a single full-page HTRFlow prediction JSON file",
    )
    parser.add_argument(
        "--gt-page-file",
        help="Path to a single ground-truth PAGE XML file for --pred-htrflow-json",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation for output (default: 2)",
    )
    parser.add_argument(
        "--output-file",
        "-o",
        help="Write JSON output to this file instead of printing it to stdout",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    pair_mode = args.pred_file and args.gt_file
    folder_mode = args.pred_dir and args.gt_dir
    htrflow_mode = args.pred_htrflow_json and args.gt_page_file

    modes_selected = sum(bool(mode) for mode in (pair_mode, folder_mode, htrflow_mode))
    if modes_selected > 1:
        parser.error(
            "Use only one mode: --pred-file/--gt-file, --pred-dir/--gt-dir, "
            "or --pred-htrflow-json/--gt-page-file."
        )

    if pair_mode:
        result = evaluate_pair(args.pred_file, args.gt_file).__dict__
    elif folder_mode:
        result = evaluate_folders(args.pred_dir, args.gt_dir)
    elif htrflow_mode:
        result = evaluate_htrflow_page(args.pred_htrflow_json, args.gt_page_file).__dict__
    else:
        parser.error(
            "Provide either file pair args, folder pair args, or "
            "--pred-htrflow-json with --gt-page-file."
        )

    output_json = json.dumps(result, indent=args.indent, ensure_ascii=False)
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as output_file:
            output_file.write(output_json)
            output_file.write("\n")
    else:
        print(output_json)


if __name__ == "__main__":
    main()
