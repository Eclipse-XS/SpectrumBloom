---
title: SpectrumBloom Agentic EDA
emoji: 🌿
colorFrom: teal
colorTo: slate
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# SpectrumBloom — Hugging Face Spaces Demo

This optional Hugging Face Spaces Docker bundle runs both the **FastAPI** backend (`127.0.0.1:8000`) and the **Streamlit** UI (`0.0.0.0:7860`) inside a single container.

## Compatible Inference Backends

1. **OpenAI-compatible / Hugging Face Inference Endpoint** (recommended on CPU Basic Spaces):
   Configure Space Secrets:
   - `LLM_PROVIDER=openai`
   - `OPENAI_BASE_URL=https://router.huggingface.co/v1`
   - `OPENAI_API_KEY=<your HF token>`
   - `LLM_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct` (or `Qwen/Qwen3-4B-Instruct-2507`)
2. **Embedded `llama.cpp` GGUF server** (on GPU Spaces or CPU with small Q4_K_M GGUF):
   Point `LLM_PROVIDER=llamacpp` and `LLAMACPP_BASE_URL=http://127.0.0.1:8080/v1`.
3. **Deterministic EDA + ML Lab** works immediately with zero external LLM keys!
