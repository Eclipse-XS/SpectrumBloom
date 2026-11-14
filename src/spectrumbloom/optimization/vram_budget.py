"""Exact VRAM and quantization feasibility estimator targeting RTX 3050 4 GB."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    params_billions: float
    num_layers: int
    num_kv_heads: int
    head_dim: int


QWEN_SPECS: dict[str, ModelSpec] = {
    "Qwen2.5-Coder-0.5B": ModelSpec("Qwen2.5-Coder-0.5B", 0.49, 24, 2, 64),
    "Qwen2.5-Coder-1.5B": ModelSpec("Qwen2.5-Coder-1.5B", 1.54, 28, 2, 128),
    "Qwen2.5-Coder-3B": ModelSpec("Qwen2.5-Coder-3B", 3.09, 36, 2, 128),
    "Qwen3-4B": ModelSpec("Qwen3-4B", 4.02, 36, 8, 128),
}

QUANT_BITS_PER_WEIGHT: dict[str, float] = {
    "FP16": 16.0,
    "Q8_0": 8.5,
    "Q5_K_M": 5.7,
    "Q4_K_M": 4.85,
    "Q3_K_M": 3.9,
    "Q2_K": 3.0,
    "NF4": 4.25,
}


@dataclass(frozen=True)
class InferenceMemoryBudget:
    model_name: str
    quantization: str
    num_ctx: int
    weights_mb: float
    kv_cache_mb: float
    runtime_overhead_mb: float
    total_vram_mb: float
    target_vram_mb: float
    fits_strict_vram: bool
    recommended_note: str


def estimate_inference_vram(
    spec: ModelSpec,
    quantization: str,
    *,
    num_ctx: int = 4096,
    target_vram_mb: float = 4096.0,
    usable_vram_mb: float = 3500.0,
) -> InferenceMemoryBudget:
    """Estimate weights + FP16 KV cache + CUDA/llama.cpp overhead in MiB."""
    bpw = QUANT_BITS_PER_WEIGHT[quantization]
    weights_mb = (spec.params_billions * 1e9 * (bpw / 8.0)) / (1024 * 1024)
    # KV cache (K + V, 2 bytes FP16): 2 * layers * kv_heads * head_dim * ctx * 2 bytes
    kv_bytes = 2 * spec.num_layers * spec.num_kv_heads * spec.head_dim * num_ctx * 2
    kv_cache_mb = kv_bytes / (1024 * 1024)
    runtime_overhead_mb = 350.0
    total_mb = weights_mb + kv_cache_mb + runtime_overhead_mb
    fits = total_mb <= usable_vram_mb

    if fits:
        note = "Fits comfortably on 4 GB GPU (RTX 3050) alongside display overhead."
    elif total_mb <= target_vram_mb:
        note = "Tight fit: requires dedicated GPU without concurrent display/VRAM consumers."
    else:
        note = "Exceeds 4 GB VRAM; requires partial CPU layer offload or smaller quantization."

    return InferenceMemoryBudget(
        model_name=spec.name,
        quantization=quantization,
        num_ctx=num_ctx,
        weights_mb=round(weights_mb, 1),
        kv_cache_mb=round(kv_cache_mb, 1),
        runtime_overhead_mb=runtime_overhead_mb,
        total_vram_mb=round(total_mb, 1),
        target_vram_mb=target_vram_mb,
        fits_strict_vram=fits,
        recommended_note=note,
    )


def estimate_qlora_training_vram(
    spec: ModelSpec,
    *,
    seq_len: int = 512,
    lora_rank: int = 8,
    usable_vram_mb: float = 3400.0,
) -> dict[str, object]:
    """Estimate QLoRA (4-bit NF4 + LoRA + paged 8-bit AdamW + grad checkpointing) VRAM."""
    base_nf4_mb = (spec.params_billions * 1e9 * (4.25 / 8.0)) / (1024 * 1024)
    # Approx ~0.8-1.5% trainable params in FP16 + 8-bit paged optimizer states + gradients
    trainable_params_m = spec.params_billions * 1000.0 * (0.008 * (lora_rank / 8.0))
    adapter_and_opt_mb = (trainable_params_m * 1e6 * 6) / (1024 * 1024)
    # Activation memory under gradient checkpointing, batch_size=1
    activation_mb = 280.0 + (seq_len / 512.0) * spec.num_layers * 14.0
    cuda_context_mb = 420.0
    total_mb = base_nf4_mb + adapter_and_opt_mb + activation_mb + cuda_context_mb
    feasible = total_mb <= usable_vram_mb

    return {
        "model": spec.name,
        "seq_len": seq_len,
        "lora_rank": lora_rank,
        "base_nf4_mb": round(base_nf4_mb, 1),
        "adapter_and_optimizer_mb": round(adapter_and_opt_mb, 1),
        "activations_mb": round(activation_mb, 1),
        "cuda_overhead_mb": cuda_context_mb,
        "estimated_total_mb": round(total_mb, 1),
        "usable_4gb_vram_mb": usable_vram_mb,
        "feasible_on_rtx3050_4gb": feasible,
        "recommendation": (
            "Feasible on RTX 3050 4 GB (batch_size=1, grad_checkpointing, paged_adamw_8bit)."
            if feasible
            else "Infeasible in 4 GB VRAM without CPU offload; use >=8-12 GB GPU."
        ),
    }


def build_feasibility_matrix() -> dict[str, object]:
    inference_rows = [
        asdict(estimate_inference_vram(spec, quant, num_ctx=4096))
        for spec in QWEN_SPECS.values()
        for quant in ("FP16", "Q8_0", "Q4_K_M", "Q3_K_M")
    ]
    qlora_rows = [
        estimate_qlora_training_vram(spec, seq_len=512, lora_rank=8) for spec in QWEN_SPECS.values()
    ]
    return {"inference_matrix": inference_rows, "qlora_matrix": qlora_rows}


if __name__ == "__main__":
    import json

    print(json.dumps(build_feasibility_matrix(), indent=2))
