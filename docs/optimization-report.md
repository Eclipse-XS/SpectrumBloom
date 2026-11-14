# SpectrumBloom Quantization, Memory & Optimization Report

This report documents empirical benchmarking and architectural optimization for running **SpectrumBloom** locally on a **4 GB VRAM consumer GPU** (`NVIDIA GeForce RTX 3050 Laptop GPU`, `4096 MiB` total VRAM).

## 1. Target Hardware Constraints

Observed baseline GPU state (`nvidia-smi` prior to model load):

- **GPU:** `NVIDIA GeForce RTX 3050 Laptop GPU` (`4096 MiB` total VRAM)
- **Active display / system baseline VRAM used:** `1131 MiB` (`2832 MiB` free at idle baseline)
- **Inference engine:** Ollama local daemon (`http://localhost:11434`), `num_ctx=4096`, `temperature=0.0`, `think=False`
- **Benchmark suite:** [`benchmarks/eda_codegen_tasks.json`](../benchmarks/eda_codegen_tasks.json) (`8` EDA tasks on [`examples/sample_ecommerce.csv`](../examples/sample_ecommerce.csv), `180` rows x `12` columns), evaluated inside the fail-closed subprocess sandbox (`SubprocessSandbox`, 15 s timeout, 512 MB cap).

Because Windows WDDM display overhead typically consumes ~0.6–1.1 GB of VRAM, only ~2.8–3.4 GB of headroom remains for weights + KV cache + CUDA scratchpad unless partial CPU offloading is used.

---

## 2. Empirical Quantization Sweep (Code-Generation Mode)

Generated via:

```bash
python -m spectrumbloom.optimization.benchmark --mode codegen --out-prefix latest_benchmark \
  --models qwen2.5-coder:1.5b-instruct-q2_K qwen2.5-coder:1.5b-instruct-q4_K_M qwen2.5-coder:1.5b-instruct-q8_0 qwen3:4b-instruct
```

Full machine-readable telemetry: [`benchmarks/results/latest_benchmark.json`](../benchmarks/results/latest_benchmark.json).

| Model | Quant | Disk (MB) | Ollama Loaded (MB) | Ollama VRAM (MB) | AST Pass | Sandbox Pass | Functional Pass | Mean Latency (ms) | Throughput (tok/s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen2.5-coder:1.5b-instruct-q2_K` | `Q2_K` | 645.0 | 819.8 | 819.8 | 75.0% | 62.5% | 25.0% | 4068.7 | 100.55 |
| `qwen2.5-coder:1.5b-instruct-q4_K_M` | `Q4_K_M` | 940.4 | 1115.2 | 1115.2 | 100.0% | 87.5% | 87.5% | 3324.2 | 95.93 |
| `qwen2.5-coder:1.5b-instruct-q8_0` | `Q8_0` | 1570.3 | 1745.1 | 1745.1 | 100.0% | 87.5% | 87.5% | 3409.5 | 79.63 |
| `qwen3:4b-instruct` | `Q4_K_M` | 2381.6 | 3371.2 | 2236.0 | 87.5% | 87.5% | 75.0% | 8583.1 | 20.88 |

### Key Empirical Findings (Code-Generation Sweep)

1. **`Q2_K` severe degradation:** Pushing 1.5B down to `Q2_K` (`819.8 MB` VRAM) reaches `100.55 tok/s`, but collapses functional correctness to **25.0% (2/8)** due to syntax corruption (`plotly_revenue_histogram`, `mean_revenue_by_category`) and hallucinated modules (`plotly.figure`).
2. **`Q4_K_M` sweet spot for 1.5B:** `qwen2.5-coder:1.5b-instruct-q4_K_M` fits completely inside GPU VRAM (`1115.2 / 1115.2 MB`), achieves **100.0% AST safety pass** and **87.5% (7/8) functional pass** at **95.93 tok/s** and **3.32 s mean latency**.
3. **`Q8_0` diminishing returns:** `qwen2.5-coder:1.5b-instruct-q8_0` consumes **+630 MB VRAM** (`1745.1 MB`) and reduces throughput by **17.0%** (`79.63 tok/s`) with **zero functional accuracy gain** (`87.5%`) over `Q4_K_M`.
4. **`Qwen3-4B Q4_K_M` CPU offload threshold:** Under concurrent desktop GPU usage, Ollama allocates `3371.2 MB` total (`2236.0 MB` on GPU VRAM + partial host RAM spill), dropping decode throughput from ~96 tok/s (1.5B) to **20.88 tok/s** (`8.58 s` mean latency).

---

## 3. Native Agent Tool-Calling Benchmark (`1.5B` vs `4B`)

Evaluating pure fenced code-generation alone misses LangGraph multi-tool orchestration. Running the same 8 tasks with native `tools=[python_repl]` enabled (`--mode tool_calling`, full results in [`benchmarks/results/latest_tool_benchmark.json`](../benchmarks/results/latest_tool_benchmark.json)):

| Model | Quant | AST Pass | Sandbox Pass | Functional Pass | Mean Latency (ms) | Throughput (tok/s) | Failure Mode |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `qwen2.5-coder:1.5b-instruct-q4_K_M` | `Q4_K_M` | 25.0% | 25.0% | 0.0% | 3053.6 | 92.82 | Hallucinates non-existent JSON tool names (`missing_value_count`, `groupby_mean`, `plotly_histogram`) instead of invoking `python_repl` |
| `qwen3:4b-instruct` | `Q4_K_M` | 87.5% | 87.5% | 87.5% | 8271.9 | 19.02 | Passes 7/8 tasks on first attempt (`import stats` rejected safely by AST gate on task 8; resolved by bounded self-repair in live graph) |

**Architectural decision:** SpectrumBloom selects **`qwen3:4b-instruct` (`Q4_K_M`)** as the default local agent orchestrator (`87.5%` first-try native tool-calling accuracy) while retaining `<tool_call>`, ````json`, and ````python` fallback parsing plus **context-window compaction** (`EDAAgentGraph._compact_history` and bounded `ToolMessage` truncation) to stay safely inside `num_ctx=4096`.

---

## 4. Analytical 4 GB VRAM Budget & QLoRA / Distillation Status

> **Note:** Tables in this section are **analytical estimates** computed from weight bit-widths, KV cache formulas, and overhead constants in [`src/spectrumbloom/optimization/vram_budget.py`](../src/spectrumbloom/optimization/vram_budget.py) and [`src/spectrumbloom/optimization/qlora.py`](../src/spectrumbloom/optimization/qlora.py), **not** measured PyTorch CUDA profiler runs.

### Analytical Inference Envelope (`num_ctx=4096`)

| Model | Quant | Weights (MB) | KV Cache (MB) | Runtime (MB) | Estimated Total (MB) | Fits Strict `<= 3500 MB` Headroom |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `Qwen2.5-Coder-0.5B` | `Q4_K_M` | 283.3 | 48.0 | 350.0 | 681.3 | Yes |
| `Qwen2.5-Coder-1.5B` | `Q4_K_M` | 890.4 | 112.0 | 350.0 | 1352.4 | Yes |
| `Qwen2.5-Coder-1.5B` | `Q8_0` | 1560.4 | 112.0 | 350.0 | 2022.4 | Yes |
| `Qwen2.5-Coder-3B` | `Q4_K_M` | 1786.5 | 144.0 | 350.0 | 2280.5 | Yes |
| `Qwen3-4B` | `Q4_K_M` | 2324.2 | 576.0 | 350.0 | 3250.2 | Yes (borderline under Windows display use) |
| `Qwen3-4B` | `Q8_0` | 4073.4 | 576.0 | 350.0 | 4999.4 | No (exceeds 4096 MB) |

### QLoRA Fine-Tuning Feasibility (`seq_len=512`, `lora_rank=8`)

| Model | Base NF4 (MB) | Adapter + Paged 8-bit Opt (MB) | Activations (MB) | CUDA Overhead (MB) | Estimated Total (MB) | Feasible on RTX 3050 4 GB |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `Qwen2.5-Coder-0.5B` | 248.3 | 22.4 | 616.0 | 420.0 | 1306.7 | Yes (`batch_size=1`, grad checkpointing) |
| `Qwen2.5-Coder-1.5B` | 780.2 | 70.5 | 672.0 | 420.0 | 1942.7 | Yes (`batch_size=1`, grad checkpointing) |
| `Qwen2.5-Coder-3B` | 1565.5 | 141.4 | 784.0 | 420.0 | 2911.0 | Tight (`batch_size=1`, no concurrent VRAM consumers) |
| `Qwen3-4B` | 2036.7 | 184.0 | 784.0 | 420.0 | 3424.7 | **No** (blocked by fail-safe guard unless `--allow-high-vram` / `>= 8-12 GB` GPU) |

- **What was executed vs. left optional:**
  - `qlora.py --check-only` feasibility guards ran against both `Qwen/Qwen3-4B` (correctly flagged **infeasible** at `3424.7 MB > 3400 MB` usable cap) and `Qwen/Qwen2.5-Coder-0.5B-Instruct` (`1306.7 MB`, feasible).
  - Full CUDA PyTorch QLoRA gradient updates (`bitsandbytes` + `peft` + `trl`) were **not executed locally** on this workstation to avoid pulling multi-GB CUDA wheels into the lightweight inference environment and because 4B QLoRA exceeds 4 GB VRAM headroom without CPU offload. The complete script is provided under the optional `.[optimization]` dependency extra for GPU environments with `>= 8 GB` VRAM or for `0.5B–1.5B` distillation fine-tuning.
- **Execution-Verified Knowledge Distillation Run:**
  - Executed [`src/spectrumbloom/optimization/distillation.py`](../src/spectrumbloom/optimization/distillation.py) with teacher `ollama:qwen3:4b-instruct` across the 8 benchmark tasks.
  - Only AST-valid, sandbox-passing, figure-verified, and assertion-passing trajectories are emitted: **6 / 8 trajectories accepted (`75.0%`)**, **2 rejected**, saved to [`benchmarks/distilled_eda_sft.jsonl`](../benchmarks/distilled_eda_sft.jsonl).
