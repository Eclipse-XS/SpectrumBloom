# SpectrumBloom Quantization & Code-Generation Benchmark

- Timestamp (UTC): `2026-11-14T16:07:21.096576+00:00`
- Dataset: `examples\sample_ecommerce.csv` (180 rows x 12 cols)
- Tasks evaluated: `8`

| Model | Quant | Disk (MB) | Ollama VRAM (MB) | AST Pass | Sandbox Pass | Functional Pass | Mean Latency (ms) | Throughput (tok/s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen2.5-coder:1.5b-instruct-q4_K_M` | Q4_K_M | 940.4 | 1115.2 | 25.0% | 25.0% | 0.0% | 3053.6 | 92.82 |
| `qwen3:4b-instruct` | Q4_K_M | 2381.6 | 2236.0 | 87.5% | 87.5% | 87.5% | 8271.9 | 19.02 |
