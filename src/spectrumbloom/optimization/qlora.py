"""QLoRA fine-tuning pipeline and hardware feasibility guard for small Qwen models."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from spectrumbloom.optimization.vram_budget import QWEN_SPECS, estimate_qlora_training_vram


@dataclass
class QLoRAConfig:
    base_model_id: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    dataset_jsonl: Path = Path("benchmarks/distilled_eda_sft.jsonl")
    output_dir: Path = Path("artifacts/qlora-qwen-eda")
    max_seq_length: int = 512
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8


def check_qlora_feasibility(model_id: str, max_seq_length: int = 512, lora_r: int = 8) -> dict:
    """Assess whether QLoRA fits inside a 4 GB RTX 3050 budget before loading weights."""
    matched_spec = QWEN_SPECS["Qwen2.5-Coder-0.5B"]
    lower = model_id.lower()
    if "1.5b" in lower:
        matched_spec = QWEN_SPECS["Qwen2.5-Coder-1.5B"]
    elif "3b" in lower:
        matched_spec = QWEN_SPECS["Qwen2.5-Coder-3B"]
    elif "4b" in lower:
        matched_spec = QWEN_SPECS["Qwen3-4B"]
    return estimate_qlora_training_vram(
        matched_spec,
        seq_len=max_seq_length,
        lora_rank=lora_r,
    )


def run_qlora_training(cfg: QLoRAConfig, *, allow_high_vram_models: bool = False) -> dict:
    """Execute 4-bit NF4 QLoRA SFT if `[optimization]` extras and sufficient VRAM are present."""
    feasibility = check_qlora_feasibility(
        cfg.base_model_id, max_seq_length=cfg.max_seq_length, lora_r=cfg.lora_r
    )
    if not feasibility["feasible_on_rtx3050_4gb"] and not allow_high_vram_models:
        raise RuntimeError(
            f"Refusing QLoRA for '{cfg.base_model_id}' on 4 GB VRAM budget: "
            f"est {feasibility['estimated_total_mb']} MB > "
            f"{feasibility['usable_4gb_vram_mb']} MB. "
            "Pass --allow-high-vram on larger GPUs or use <=1.5B models."
        )

    if not cfg.dataset_jsonl.exists():
        raise FileNotFoundError(
            f"Training dataset not found at '{cfg.dataset_jsonl}'. "
            "Run `python -m spectrumbloom.optimization.distillation` first."
        )

    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError(
            "QLoRA training requires optional `[optimization]` dependencies: "
            "`uv pip install -e '.[optimization]'`."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not detected; 4-bit bitsandbytes QLoRA requires CUDA.")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_id, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=False,
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
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

    dataset = load_dataset("json", data_files=str(cfg.dataset_jsonl), split="train")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    sft_args = SFTConfig(
        output_dir=str(cfg.output_dir),
        max_seq_length=cfg.max_seq_length,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        optim="paged_adamw_8bit",
        fp16=True,
        logging_steps=5,
        save_strategy="epoch",
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )
    train_result = trainer.train()
    trainer.save_model(str(cfg.output_dir))

    peak_vram_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
    return {
        "model": cfg.base_model_id,
        "output_dir": str(cfg.output_dir),
        "train_loss": float(train_result.training_loss),
        "peak_cuda_allocated_mb": peak_vram_mb,
        "feasibility_estimate": feasibility,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="SpectrumBloom QLoRA feasibility & trainer")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    parser.add_argument("--dataset", type=Path, default=Path("benchmarks/distilled_eda_sft.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/qlora-qwen-eda"))
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--allow-high-vram", action="store_true")
    args = parser.parse_args()

    if args.check_only:
        print(json.dumps(check_qlora_feasibility(args.model_id), indent=2))
        return 0

    cfg = QLoRAConfig(
        base_model_id=args.model_id,
        dataset_jsonl=args.dataset,
        output_dir=args.output_dir,
    )
    result = run_qlora_training(cfg, allow_high_vram_models=args.allow_high_vram)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
