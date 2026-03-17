# -*- coding: utf-8 -*-
r"""
Split scored_input2_3class.json into stratified train/val/test
and export JSONL for LLM finetuning (instruction, input, output).

Outputs (in same dir by default):
- train_code.jsonl, val_code.jsonl, test_code.jsonl
- train_code_metrics.jsonl, val_code_metrics.jsonl, test_code_metrics.jsonl

Usage:
cd "C:\Users\lenovo\Desktop\大模型"
python .\split_and_export_jsonl.py --input .\scored_input2_3class.json --outdir .\.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split


DEFAULT_METRIC_KEYS = [
    "line_count",
    "assertion_count",
    "has_docstring",
    "comment_count",
    "cyclomatic_complexity",
    "has_mocks",
    "duplicate_code",
    "dc_coherence_codet5",
    "docstring_strength",
]


# English instruction (strict, classification-only) — Version 1 (cluster-derived rules)
INSTRUCTION = (
    "You are a strict unit-test code quality evaluator. "
    "Given the unit-test code and the provided metrics, output EXACTLY ONE digit label: 1, 2, or 3. "
    "Output only the digit, with no explanation or extra text. "
    "Evaluate the overall quality of the unit test using both the code and the metrics. "
    "Important signals: "
    "Tests with has_mocks == 1 are often lower quality. "
    "Tests with has_mocks == 0 and docstring_strength > 0 are often higher quality. "
    "Tests with has_mocks == 0 but docstring_strength == 0 are often medium quality. "
    "Also consider whether the test is clear, well-documented, and meaningfully checks behavior rather than only setup. "
    "Label meanings: "
    "1 = low quality. 2 = medium quality. 3 = high quality. "
    "Return only: 1 or 2 or 3."
)


def build_metrics_text(metrics: Dict[str, Any], keys: List[str]) -> str:
    if not isinstance(metrics, dict):
        metrics = {}
    parts = []
    for k in keys:
        v = metrics.get(k, None)
        parts.append(f"{k}: {v}")
    return "\n".join(parts)


def to_jsonl_records(
    items: List[Dict[str, Any]],
    mode: str,
    metric_keys: List[str],
    label_key: str,
) -> List[Dict[str, str]]:
    """
    mode:
      - "code": input only code
      - "code_metrics": input code + metrics block
    """
    records = []
    for it in items:
        code = it.get("text", "")
        label = it.get(label_key, None)
        if label is None:
            raise ValueError(f"Missing {label_key} in item: {it.get('identifier')}")

        if mode == "code":
            model_input = f"Code:\n{code}\n"
        elif mode == "code_metrics":
            metrics_text = build_metrics_text(it.get("metrics", {}), metric_keys)
            model_input = f"Code:\n{code}\n\nMetrics:\n{metrics_text}\n"
        else:
            raise ValueError(f"Unknown mode: {mode}")

        records.append(
            {
                "instruction": INSTRUCTION,
                "input": model_input,
                "output": str(int(label)),  # ensure "1"/"2"/"3"
            }
        )
    return records


def save_jsonl(path: Path, records: List[Dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="scored_input2_3class.json")
    parser.add_argument("--outdir", default=".")
    parser.add_argument("--label_key", default="label_3class")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.20)   # 20% fixed test
    parser.add_argument("--val_size", type=float, default=0.10)    # 10% of remaining 80% => 8% overall
    args = parser.parse_args()

    in_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data: List[Dict[str, Any]] = json.loads(in_path.read_text(encoding="utf-8"))

    y = np.array([int(x[args.label_key]) for x in data], dtype=int)

    # 1) Split out test (stratified)
    trainval, test = train_test_split(
        data,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    y_trainval = np.array([int(x[args.label_key]) for x in trainval], dtype=int)

    # 2) Split train/val from remaining (val_size is fraction of trainval)
    train, val = train_test_split(
        trainval,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=y_trainval,
    )

    # Print label distribution (sanity check)
    def dist(items: List[Dict[str, Any]]) -> Dict[int, int]:
        d: Dict[int, int] = {}
        for it in items:
            k = int(it[args.label_key])
            d[k] = d.get(k, 0) + 1
        return dict(sorted(d.items()))

    print("Total:", len(data), "Dist:", dist(data))
    print("Train:", len(train), "Dist:", dist(train))
    print("Val:", len(val), "Dist:", dist(val))
    print("Test:", len(test), "Dist:", dist(test))

    # Export JSONL (code-only)
    train_code = to_jsonl_records(train, "code", DEFAULT_METRIC_KEYS, args.label_key)
    val_code = to_jsonl_records(val, "code", DEFAULT_METRIC_KEYS, args.label_key)
    test_code = to_jsonl_records(test, "code", DEFAULT_METRIC_KEYS, args.label_key)

    save_jsonl(outdir / "train_code.jsonl", train_code)
    save_jsonl(outdir / "val_code.jsonl", val_code)
    save_jsonl(outdir / "test_code.jsonl", test_code)

    # Export JSONL (code + metrics)
    train_cm = to_jsonl_records(train, "code_metrics", DEFAULT_METRIC_KEYS, args.label_key)
    val_cm = to_jsonl_records(val, "code_metrics", DEFAULT_METRIC_KEYS, args.label_key)
    test_cm = to_jsonl_records(test, "code_metrics", DEFAULT_METRIC_KEYS, args.label_key)

    save_jsonl(outdir / "train_code_metrics.jsonl", train_cm)
    save_jsonl(outdir / "val_code_metrics.jsonl", val_cm)
    save_jsonl(outdir / "test_code_metrics.jsonl", test_cm)

    print("\nSaved JSONL files to:", str(outdir.resolve()))


if __name__ == "__main__":
    main()

