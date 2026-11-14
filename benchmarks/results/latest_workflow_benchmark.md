# SpectrumBloom End-to-End Agent Workflow Benchmark

Generated: `2026-11-14T22:33:28.240812+00:00` | Sandbox: `docker` | `num_ctx=4096`

| Model | Run | Correctness | Tool Exec | Retry Rate | Timeouts | P50 (s) | P95 (s) | GPU Offload | Peak VRAM (MB) | Peak RAM (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen2.5-coder:1.5b-instruct-q4_K_M` | 1 | 48/67 (71.6%) | 51.7% | 47.8% | 0 | 4.76 | 22.66 | 100.0% | 1390.0 | 314.15 |
| `qwen2.5-coder:1.5b-instruct-q4_K_M` | 2 | 47/67 (70.1%) | 51.4% | 47.8% | 0 | 4.96 | 14.28 | 100.0% | 1489.0 | 348.43 |
| `qwen2.5-coder:3b-instruct-q4_K_M` | 1 | 47/67 (70.1%) | 77.1% | 20.9% | 0 | 4.59 | 14.43 | 100.0% | 2407.0 | 345.85 |
| `qwen2.5-coder:3b-instruct-q4_K_M` | 2 | 47/67 (70.1%) | 78.3% | 19.4% | 0 | 4.4 | 12.18 | 100.0% | 2407.0 | 365.92 |
| `qwen3:4b-instruct` | 1 | 61/67 (91.0%) | 90.8% | 9.0% | 0 | 11.72 | 33.91 | 66.3% | 2623.0 | 321.77 |
| `qwen3:4b-instruct` | 2 | 61/67 (91.0%) | 90.6% | 9.0% | 0 | 17.49 | 40.75 | 66.3% | 2617.0 | 265.51 |
