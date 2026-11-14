---
title: SpectrumBloom Deterministic EDA & ML Studio
emoji: 🌿
colorFrom: teal
colorTo: slate
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# SpectrumBloom — Public Hugging Face Spaces Demo (Safe Deterministic Mode)

This Hugging Face Spaces bundle runs both the **FastAPI** backend (`127.0.0.1:8000`) and the **Streamlit** UI (`0.0.0.0:7860`) inside a single container.

## Security Policy for Public Spaces (`SANDBOX_BACKEND=disabled`)

Shared multi-tenant Hugging Face Spaces containers cannot safely launch sibling Docker containers or privileged user namespaces. Therefore, this Space sets:

- `SANDBOX_BACKEND=disabled`: Dynamic agent-generated Python code execution (`python_repl`) is **explicitly disabled** rather than falling back to an unsafe subprocess backend.
- All **deterministic EDA** (CSV/Excel profiling, missingness, normality, correlations, IQR/Z/MAD outliers, IsolationForest, Kruskal-Wallis group tests, interactive Plotly charts), **leak-safe ML Lab** (`Dummy`, `LogisticRegression`/`Ridge`, `RandomForest`, `HistGradientBoosting` + permutation importance), and **deterministic copilot tools** (`get_statistical_summary`, `aggregate_data`, `compute_statistical_test`, `create_standard_chart`, `run_ml_baseline`) run immediately and safely.

## Optional LLM Inference Configuration (for Copilot Deterministic Tool Calling)

1. **OpenAI-compatible / Hugging Face Inference Endpoint**:
   Configure Space Secrets:
   - `LLM_PROVIDER=openai`
   - `OPENAI_BASE_URL=https://router.huggingface.co/v1`
   - `OPENAI_API_KEY=<your HF token>`
   - `LLM_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct`
2. **Offline / Keyless Mode**:
   Even with no external API key configured, full automated dataset profiling, statistical diagnostics, interactive Plotly exploration, ML baseline training, and CSV/HTML/JSON exports work out of the box.
