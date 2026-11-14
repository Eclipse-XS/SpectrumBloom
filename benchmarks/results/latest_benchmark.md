# SpectrumBloom Quantization & Code-Generation Benchmark

- Timestamp (UTC): `2026-11-14T16:04:45.448993+00:00`
- Dataset: `examples\sample_ecommerce.csv` (180 rows x 12 cols)
- Tasks evaluated: `8`

| Model | Quant | Disk (MB) | Ollama VRAM (MB) | AST Pass | Sandbox Pass | Functional Pass | Mean Latency (ms) | Throughput (tok/s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen2.5-coder:1.5b-instruct-q2_K` | Q2_K | 645.0 | 819.8 | 75.0% | 62.5% | 25.0% | 4068.7 | 100.55 |
| `qwen2.5-coder:1.5b-instruct-q4_K_M` | Q4_K_M | 940.4 | 1115.2 | 100.0% | 87.5% | 87.5% | 3324.2 | 95.93 |
| `qwen2.5-coder:1.5b-instruct-q8_0` | Q8_0 | 1570.3 | 1745.1 | 100.0% | 87.5% | 87.5% | 3409.5 | 79.63 |
| `qwen3:4b-instruct` | Q4_K_M | 2381.6 | 2236.0 | 87.5% | 87.5% | 75.0% | 8583.1 | 20.88 |
