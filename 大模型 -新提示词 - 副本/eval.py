# -*- coding: utf-8 -*-

import json
import re
from typing import List

import torch
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)
from transformers import AutoModelForCausalLM, AutoTokenizer


# =========================
# Config
# =========================
BASE_MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"
LORA_MODEL_PATH = "./phi35_lora_code_metrics_cls"   # 训练输出目录
TEST_FILE = "test_code.jsonl"               # 如果测 code+metrics，改成 test_code_metrics.jsonl

MAX_NEW_TOKENS = 8
OUTPUT_PRED_FILE = "test_code(no_metics)predictions.json"


# =========================
# Utils
# =========================
def extract_label(text: str):
    """
    从模型生成文本中提取第一个合法标签 1/2/3
    """
    if text is None:
        return None
    match = re.search(r"\b([123])\b", text)
    if match:
        return int(match.group(1))
    return None


def load_jsonl(path: str) -> List[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_prompt(instruction: str, user_input: str) -> str:
    """
    与训练时格式保持一致
    测试时不要把 output 喂给模型
    """
    return (
        f"Instruction:\n{instruction.strip()}\n\n"
        f"Input:\n{user_input.strip()}\n\n"
        f"Response:\n"
    )


def get_dtype():
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


# =========================
# Main
# =========================
def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        dtype=get_dtype(),
        attn_implementation="eager",
    )

    if torch.cuda.is_available():
        base_model = base_model.to("cuda")

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH)
    model.eval()
    model.config.use_cache = True

    print(f"Loading test set from: {TEST_FILE}")
    test_data = load_jsonl(TEST_FILE)

    y_true = []
    y_pred = []
    pred_records = []

    print(f"Start inference on {len(test_data)} samples...\n")

    for idx, item in enumerate(test_data, start=1):
        instruction = str(item["instruction"])
        user_input = str(item["input"])
        true_label = int(item["output"])

        prompt = build_prompt(instruction, user_input)

        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        input_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_length:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        pred_label = extract_label(generated_text)

        y_true.append(true_label)
        y_pred.append(pred_label if pred_label is not None else -1)

        pred_records.append(
            {
                "index": idx,
                "true_label": true_label,
                "pred_label": pred_label,
                "raw_generation": generated_text,
                "input_preview": user_input[:300],
            }
        )

        if idx <= 5:
            print(f"[Sample {idx}]")
            print("True :", true_label)
            print("Pred :", pred_label)
            print("Raw  :", repr(generated_text))
            print("-" * 60)

    invalid_count = sum(1 for x in y_pred if x == -1)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

    print("\n================ Test Results ================")
    print(f"Total samples      : {len(y_true)}")
    print(f"Invalid predictions: {invalid_count}")
    print(f"Accuracy           : {acc:.6f}")
    print(f"Macro-F1           : {macro_f1:.6f}")
    print(f"Macro-Precision    : {macro_precision:.6f}")
    print(f"Macro-Recall       : {macro_recall:.6f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[1, 2, 3],
            zero_division=0,
            digits=6,
        )
    )

    print("Confusion Matrix (labels=1,2,3):")
    print(confusion_matrix(y_true, y_pred, labels=[1, 2, 3]))

    with open(OUTPUT_PRED_FILE, "w", encoding="utf-8") as f:
        json.dump(pred_records, f, ensure_ascii=False, indent=2)

    print(f"\nSaved detailed predictions to: {OUTPUT_PRED_FILE}")


if __name__ == "__main__":
    main()