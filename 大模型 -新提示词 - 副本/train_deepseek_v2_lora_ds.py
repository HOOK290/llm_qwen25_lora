import argparse
import random
import numpy as np
import torch
import inspect

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_text_with_chat_template(example, tokenizer):
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
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
    else:
        text = (
            f"Instruction:\n{instruction}\n\n"
            f"Input:\n{user_input}\n\n"
            f"Response:\n{output}"
        )
    return {"text": text}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct")
    parser.add_argument("--train", default="train_code_metrics.jsonl")
    parser.add_argument("--val", default="val_code_metrics.jsonl")
    parser.add_argument("--outdir", default="./deepseek_v2_lora_metrics_cls")

    parser.add_argument("--seed", type=int, default=42)

    # training
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--packing", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--bsz", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)

    # deepspeed
    parser.add_argument("--deepspeed", default="ds_zero2_bf16.json")

    # qlora (optional)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args()

    set_all_seeds(args.seed)

    dataset = load_dataset(
        "json",
        data_files={"train": args.train, "validation": args.val}
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    compute_dtype = (
        torch.bfloat16
        if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        else torch.float16
    )

    quant_config = None
    if args.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        device_map=None,
        torch_dtype=compute_dtype,
        quantization_config=quant_config,
        attn_implementation="eager",
    )

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    dataset["train"] = dataset["train"].map(
        lambda ex: build_text_with_chat_template(ex, tokenizer)
    )
    dataset["validation"] = dataset["validation"].map(
        lambda ex: build_text_with_chat_template(ex, tokenizer)
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    sft_config_sig = inspect.signature(SFTConfig.__init__)
    sft_config_params = sft_config_sig.parameters

    sft_config_kwargs = {
        "output_dir": args.outdir,
        "overwrite_output_dir": True,
        "per_device_train_batch_size": args.bsz,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "num_train_epochs": args.epochs,
        "logging_steps": 10,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": "none",
        "remove_unused_columns": False,
        "packing": bool(args.packing),
        "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "fp16": torch.cuda.is_available() and (not torch.cuda.is_bf16_supported()),
        "deepspeed": args.deepspeed,
    }

    if "save_strategy" in sft_config_params:
        sft_config_kwargs["save_strategy"] = "epoch"
    if "evaluation_strategy" in sft_config_params:
        sft_config_kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in sft_config_params:
        sft_config_kwargs["eval_strategy"] = "epoch"

    length_set_in_config = False
    if "max_seq_length" in sft_config_params:
        sft_config_kwargs["max_seq_length"] = args.max_length
        length_set_in_config = True
    elif "max_length" in sft_config_params:
        sft_config_kwargs["max_length"] = args.max_length
        length_set_in_config = True

    sft_args = SFTConfig(**sft_config_kwargs)

    trainer_sig = inspect.signature(SFTTrainer.__init__)
    trainer_params = trainer_sig.parameters

    trainer_kwargs = {
        "model": model,
        "args": sft_args,
        "train_dataset": dataset["train"],
        "eval_dataset": dataset["validation"],
        "peft_config": peft_config,
    }

    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    if "dataset_text_field" in trainer_params:
        trainer_kwargs["dataset_text_field"] = "text"
    else:
        trainer_kwargs["formatting_func"] = lambda example: example["text"]

    if not length_set_in_config:
        if "max_seq_length" in trainer_params:
            trainer_kwargs["max_seq_length"] = args.max_length
        elif "max_length" in trainer_params:
            trainer_kwargs["max_length"] = args.max_length

    trainer = SFTTrainer(**trainer_kwargs)

    trainer.train()

    eval_result = trainer.evaluate()
    print("\nValidation results:")
    for k, v in eval_result.items():
        print(f"{k}: {v}")

    trainer.model.save_pretrained(args.outdir)
    tokenizer.save_pretrained(args.outdir)
    print(f"\nSaved to: {args.outdir}")


if __name__ == "__main__":
    main()