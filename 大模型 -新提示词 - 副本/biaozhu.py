# -*- coding: utf-8 -*-
"""
Re-label scored_input2_augmented.json into 3 classes using:
- 1/2 -> class 1
- 3 and (4 with docstring==0) -> class 2
- (4 with docstring!=0) and 5 -> class 3

Usage:
  python relabel_3class.py --input scored_input2_augmented.json --output scored_input2_3class.json
"""

import argparse
import json
from typing import Any, Dict, List


def get_has_docstring(metrics: Dict[str, Any]) -> int:
    """
    Prefer metrics["has_docstring"] if present; otherwise fall back to docstring_strength>0.
    Returns 0 or 1.
    """
    if metrics is None:
        return 0
    if "has_docstring" in metrics and metrics["has_docstring"] is not None:
        try:
            return 1 if int(metrics["has_docstring"]) != 0 else 0
        except Exception:
            return 1 if bool(metrics["has_docstring"]) else 0

    strength = metrics.get("docstring_strength", 0)
    try:
        return 1 if float(strength) > 0 else 0
    except Exception:
        return 0


def map_to_3class(item: Dict[str, Any]) -> int:
    score_raw = item.get("score")
    try:
        score = int(score_raw)
    except Exception:
        raise ValueError(f"Invalid score: {score_raw!r} (identifier={item.get('identifier')!r})")

    metrics = item.get("metrics", {})
    has_docstring = get_has_docstring(metrics)

    if score in (1, 2):
        return 1
    if score == 3:
        return 2
    if score == 4:
        return 3 if has_docstring != 0 else 2
    if score == 5:
        return 3

    raise ValueError(f"Unexpected score={score} (identifier={item.get('identifier')!r})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSON file (list of records)")
    parser.add_argument("--output", required=True, help="Output JSON file (list of records)")
    parser.add_argument("--label_key", default="label_3class", help="Field name to store new label")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    for item in data:
        item[args.label_key] = map_to_3class(item)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
