import re
import random
import numpy as np
import torch

from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)

from trl import SFTTrainer, SFTConfig


# =========================
# Config
# =========================
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"

TRAIN_FILE = "train_code_metrics.jsonl"
VAL_FILE = "val_code_metrics.jsonl"

OUTPUT_DIR = "./qwen25_lora_code_metrics_cls"

SEED = 42

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Training
PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 2e-4
NUM_TRAIN_EPOCHS = 3

MAX_LENGTH = 1024  # TRL新版本用 max_length（不是 max_seq_length）


# =========================
# Utils
# =========================
def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_text_with_chat_template(example, tokenizer):
    """
    Convert one JSONL sample into a single training text using chat template.
    Input JSONL format:
    {
      "instruction": "...",
      "input": "...",
      "output": "1"
    }
    """
    instruction = str(example["instruction"]).strip()
    user_input = str(example["input"]).strip()
    output = str(example["output"]).strip()

    user_content = f"{instruction}\n\n{user_input}"

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    else:
        # fallback
        text = (
            f"Instruction:\n{instruction}\n\n"
            f"Input:\n{user_input}\n\n"
            f"Response:\n{output}"
        )

    return {"text": text}


# =========================
# Main
# =========================
def main():
    set_all_seeds(SEED)

    # 1) Load dataset
    dataset = load_dataset(
        "json",
        data_files={"train": TRAIN_FILE, "validation": VAL_FILE},
    )

    # 2) Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 3) QLoRA 4-bit config (recommended on RTX 4060)
    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # 4) Model (4-bit)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=bnb_config,
        attn_implementation="eager",
        dtype=compute_dtype,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # 5) Build "text" field (after tokenizer is ready)
    dataset["train"] = dataset["train"].map(lambda ex: build_text_with_chat_template(ex, tokenizer))
    dataset["validation"] = dataset["validation"].map(lambda ex: build_text_with_chat_template(ex, tokenizer))

    # 6) LoRA config (Qwen/LLaMA-like module names)
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    # 7) TRL SFTConfig (NOTE: use max_length, not max_seq_length)
    sft_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        remove_unused_columns=False,
        max_length=MAX_LENGTH,      # ✅ 关键：新版本用 max_length
        packing=False,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and (not torch.cuda.is_bf16_supported()),
    )

    # 8) Trainer (no max_seq_length here)
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=peft_config,
        formatting_func=lambda example: example["text"],
    )

    # 9) Train
    trainer.train()

    # 10) Eval
    eval_result = trainer.evaluate()
    print("\nValidation results:")
    for key, value in eval_result.items():
        print(f"{key}: {value}")

    # 11) Save
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nTraining finished. Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()