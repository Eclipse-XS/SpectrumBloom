# SpectrumBloom

**Local-first, fail-closed, production-oriented Agentic Exploratory Data Analysis (EDA) platform.**

SpectrumBloom is an independently engineered modular monolith built with **FastAPI**, **Pydantic v2**, **LangGraph**, deterministic statistical & Plotly EDA engines, a **fail-closed sandboxed Python runtime**, **OpenTelemetry + Langfuse + DeepEval** evaluation/observability, and **local quantized Qwen inference** tuned for **4 GB VRAM consumer GPUs** (e.g., NVIDIA RTX 3050 4 GB via **Ollama** or **llama.cpp**), with external cloud LLMs strictly optional.

---

## Architecture Overview

```text
Streamlit UI (src/spectrumbloom/ui/app.py)
   │ HTTP REST (httpx)
   ▼
FastAPI Gateway (src/spectrumbloom/api/routes.py)  ── Bearer Auth + CORS + OTel
   │
   ▼
SpectrumBloomService (src/spectrumbloom/services/session_service.py)
   ├── Repositories (repositories/)       : Thread-safe Memory / Redis sessions + Local / S3 datasets
   ├── Deterministic EDA (eda/)           : Zip-bomb guarded loader, profiler, stats engine, Plotly builder
   ├── Leak-Safe ML Lab (ml/)             : Pipeline + ColumnTransformer CV baseline benchmark + permutation imp
   ├── LangGraph Agent (agent/, llm/)     : UnifiedLLMProvider + bounded retry loop + context compaction
   ├── Fail-Closed Sandbox (sandbox/)     : AST gate + Parquet IPC + python -I + JobObject/rlimits/nsjail
   └── Eval & Telemetry (services/, core/): Deterministic turn verifier, DeepEval Ollama judge, OTel + Langfuse
```

See full architectural details in [docs/architecture.md](docs/architecture.md), security & clean-room audit findings in [docs/reference-audit.md](docs/reference-audit.md), and local 4 GB VRAM quantization results in [docs/optimization-report.md](docs/optimization-report.md).

---

## Core Capabilities

1. **Deterministic & Visual EDA (`eda/`)**
   - Hardened tabular ingestion (`csv`, `tsv`, `txt`, `xlsx`, `xlsm`, `xls`, `ods`) with byte-size limit, encoding fallback, CSV delimiter sniffing, Excel formula-injection neutralization, and XML zip-bomb bounds (`loader.py`).
   - Semantic column typing (`numeric_continuous`, `numeric_discrete`, `categorical`, `boolean`, `datetime`, `identifier`, `text`, `empty`), skewness/kurtosis, and data quality alerts (`profiler.py`).
   - Deep statistical diagnostics (`statistics.py`): missingness co-occurrence pairs, Pearson & Spearman correlations, Cramér's V categorical associations, Shapiro-Wilk / D'Agostino normality tests, IQR / Z-score / robust MAD outliers, `IsolationForest` multivariate anomalies, and Kruskal-Wallis group tests.
   - Interactive Plotly visualizations (`charts.py`): missingness heatmap, numeric distributions, correlation matrix, box/outlier diagnostics, and top categorical breakdowns.

2. **Agentic EDA Workspace (`agent/`, `llm/`)**
   - Local-first inference via `UnifiedLLMProvider` supporting **Ollama** (`qwen3:4b-instruct` default), **llama.cpp** OpenAI-compatible server, and optional external **OpenAI-compatible** or **Gemini** endpoints.
   - Stateful LangGraph orchestration (`EDAAgentGraph`) binding `python_repl`, `get_statistical_summary`, and `run_ml_baseline`, with native tool-calling plus `<tool_call>` XML, bare JSON, ````json`, and ````python` fallback parsing.
   - Context-window compaction for 4 GB local inference (`num_ctx=4096`) and bounded self-repair retries (`MAX_REPAIR_RETRIES = 2`).

3. **Fail-Closed Python Execution Sandbox (`sandbox/`)**
   - **Never** falls back to unrestricted execution when an isolation backend is unavailable (`SandboxUnavailableError`).
   - Static AST gate rejects disallowed imports, dunder traversal, `eval`/`exec`/`getattr`/`open`, and pandas `df.eval()` / `df.query()`.
   - Pickle-free IPC via PyArrow Parquet (with typed JSON fallback).
   - **Isolation Boundary Distinction:**
     - `strict_subprocess` (portable local default): **process-level application restriction**, *not* equivalent to OS container/kernel isolation — combines `python -I`, stripped env, Python socket/DNS kill-switch, prefix-restricted read-only `open()`, and fail-closed OS limits (Windows Kernel Job Objects with explicit assignment verification, or POSIX `rlimit` + process-tree physical RSS watchdog).
     - `docker` (`Dockerfile.sandbox`, verified E2E): non-privileged per-run Linux container (`--network=none --read-only --cap-drop=ALL --security-opt=no-new-privileges:true --pids-limit=64 --memory=...`).
     - `nsjail` (`sandbox/nsjail.cfg`): implemented fail-closed adapter for Linux hosts with `nsjail` installed (not verified in CI without an `nsjail` binary).

4. **Separate Baseline ML Lab (`ml/experiments.py`)**
   - Leak-safe `ColumnTransformer` + `Pipeline` with cross-validation (`Dummy`, `LogisticRegression`/`Ridge`, `RandomForest`, `HistGradientBoosting`), held-out test metrics, permutation feature importance, and interactive Plotly diagnostics.

5. **Evaluation, Observability & Optimization (`services/eval_service.py`, `optimization/`)**
   - Automated per-turn verification (tool execution success, Plotly JSON validity, tolerance-aware numeric grounding, `<think>` leakage check) + optional `DeepEval` `GEval` groundedness scoring backed by a local Ollama judge (`OllamaDeepEvalJudge`).
   - OpenTelemetry trace/metric instrumentation (`core/telemetry.py`, compatible with Jaeger / OTLP) and optional Langfuse callbacks.

---

## Empirical Local LLM & Quantization Summary (RTX 3050 4 GB)

Measured on an **NVIDIA GeForce RTX 3050 Laptop GPU (4096 MiB)** running Ollama (`num_ctx=4096`, `temperature=0.0`) over [`benchmarks/eda_codegen_tasks.json`](benchmarks/eda_codegen_tasks.json) (`8` EDA tasks on [`examples/sample_ecommerce.csv`](examples/sample_ecommerce.csv)):

| Model | Quant | Mode | Ollama VRAM (MB) | AST Pass | Sandbox Pass | Functional Pass | Mean Latency (ms) | Throughput (tok/s) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen2.5-coder:1.5b-instruct-q2_K` | `Q2_K` | `codegen` | 819.8 | 75.0% | 62.5% | 25.0% | 4068.7 | 100.55 |
| `qwen2.5-coder:1.5b-instruct-q4_K_M` | `Q4_K_M` | `codegen` | 1115.2 | 100.0% | 87.5% | **87.5%** | 3324.2 | **95.93** |
| `qwen2.5-coder:1.5b-instruct-q8_0` | `Q8_0` | `codegen` | 1745.1 | 100.0% | 87.5% | 87.5% | 3409.5 | 79.63 |
| `qwen3:4b-instruct` | `Q4_K_M` | `codegen` | 2236.0 | 87.5% | 87.5% | 75.0% | 8583.1 | 20.88 |
| `qwen2.5-coder:1.5b-instruct-q4_K_M` | `Q4_K_M` | `tool_calling` | 1115.2 | 25.0% | 25.0% | 0.0% | 3053.6 | 92.82 |
| `qwen3:4b-instruct` | `Q4_K_M` | `tool_calling` | 2236.0 | 87.5% | 87.5% | **87.5%** | 8271.9 | 19.02 |

See [docs/optimization-report.md](docs/optimization-report.md) for the breakdown between measured Ollama benchmarks, analytical 4 GB VRAM estimates, dry-run QLoRA feasibility guards (`qlora.py` CUDA training not executed), and execution-verified SFT distillation trajectories (**6/8 verified trajectories** saved to [`benchmarks/distilled_eda_sft.jsonl`](benchmarks/distilled_eda_sft.jsonl)).

---

## Verification, Estimates & Implementation Status

| Component | Status | Evidence / Scope |
| --- | --- | --- |
| FastAPI REST + Deterministic EDA + Leak-Safe ML + Auth Guards | Verified | Unit & integration tests (`tests/`) + hardened production config validator |
| `SubprocessSandbox` (Windows Job Object & Linux RSS Watchdog) | Verified | Unit tests on Windows + Linux (CI), plus failed Job Object assignment tests |
| `DockerSandbox` (`Dockerfile.sandbox`) | Verified | Real end-to-end container tests (`tests/integration/test_docker_sandbox.py`) |
| Full E2E Agent Workflow (LangGraph + Ollama `qwen3:4b-instruct` + Sandbox + Plotly) | Verified | Executed live against Ollama + Uvicorn using [`examples/sample_api_workflow.py`](examples/sample_api_workflow.py) |
| Quantization Sweep (`Q2_K` / `Q4_K_M` / `Q8_0` / `Qwen3-4B`) | Measured | Empirical Ollama benchmark runs in `benchmarks/results/` |
| Inference VRAM Budget & QLoRA Memory Table (`vram_budget.py`, `qlora.py`) | Analytical estimate | Calculated from parameter count/quant formulas + `qlora.py --check-only`; CUDA weight updates **not run** |
| SFT Trajectory Distillation (`distillation.py`) | Executed (dataset only) | Generated `6/8` execution-verified JSONL examples; downstream PyTorch QLoRA fine-tuning **not run** |
| `NsjailSandbox`, `RedisSessionRepository`, `S3DatasetStore`, `llama.cpp`, External OpenAI/Gemini, Live OTLP/Langfuse/DeepEval, Cloud Run, HF Spaces | Implemented, not E2E tested live | Code/adapters/manifests implemented and tested offline or fail-closed where applicable; live external deployments not run |

---

## Quickstart (Local Setup)

### 1. Prerequisites
- Python **3.12** (supported: `>=3.11,<3.14`) and [`uv`](https://docs.astral.sh/uv/)
- Local **Ollama** (`ollama pull qwen3:4b-instruct`) or **llama.cpp** OpenAI-compatible server

### 2. Install & Configure
```bash
uv venv --python 3.12 .venv
# Linux / macOS: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1

uv pip install -e ".[eval,dev]"
cp .env.example .env
```

### 3. Start Backend API & Streamlit UI
Terminal 1 — FastAPI backend:
```bash
uvicorn spectrumbloom.main:app --host 127.0.0.1 --port 8000
```
Terminal 2 — SpectrumBloom UI:
```bash
streamlit run src/spectrumbloom/ui/app.py --server.port 8501
```
Then open `http://localhost:8501` (API interactive docs at `http://127.0.0.1:8000/docs`), or run the standalone REST workflow script:
```bash
python examples/sample_api_workflow.py
```

---

## REST API Surface

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Readiness & dependency status (storage, sandbox, LLM, OTel/Langfuse) |
| `POST` | `/api/v1/sessions` | Create an isolated EDA session (provider/model config) |
| `GET` | `/api/v1/sessions` | List active sessions |
| `GET` | `/api/v1/sessions/{session_id}` | Session status & active dataset metadata |
| `PATCH` | `/api/v1/sessions/{session_id}/provider` | Switch LLM provider/model at runtime |
| `POST` | `/api/v1/sessions/{session_id}/clear` | Clear session conversation history |
| `DELETE` | `/api/v1/sessions/{session_id}` | Delete session and optionally clean stored dataset |
| `GET` | `/api/v1/sessions/{session_id}/history` | Retrieve chat history & last tool traces |
| `POST` | `/api/v1/sessions/{session_id}/upload` | Upload CSV/TSV/Excel/ODS dataset and run full deterministic EDA |
| `GET` | `/api/v1/sessions/{session_id}/data/preview` | Formula-safe paginated dataset preview |
| `GET` | `/api/v1/sessions/{session_id}/eda` | Retrieve cached dataset profile, stats & Plotly charts |
| `POST` | `/api/v1/sessions/{session_id}/ml/experiment` | Run leak-safe CV baseline ML comparison & permutation importance |
| `POST` | `/api/v1/sessions/{session_id}/query` | Execute a multi-step agentic EDA query with evaluation |

`API_HOST` defaults to `127.0.0.1` to prevent accidental unauthenticated network exposure. When `APP_ENV=production` or `staging`, startup requires a strong `API_AUTH_TOKEN` (`>= 16` chars, non-placeholder) and disallows wildcard CORS (`*`). In `development`, binding a non-loopback `API_HOST` without `API_AUTH_TOKEN` is rejected at startup. When configured, `/api/v1/*` routes enforce `Authorization: Bearer <token>`.

---

## Docker Compose & Optional Deployments

- **Non-privileged Docker Compose (`cap_drop: ALL`, `no-new-privileges: true`):**
  ```bash
  docker compose up --build
  ```
  Optional profiles add Redis persistence (`--profile persistence`), containerized Ollama (`--profile local-llm`), or Jaeger OpenTelemetry tracing (`--profile observability`).
- **Optional Hugging Face Spaces Demo:** See [`deploy/hf-spaces/README.md`](deploy/hf-spaces/README.md).
- **Optional Google Cloud Run:** See [`deploy/gcp/README.md`](deploy/gcp/README.md) (no mandatory paid services required for local use).

---

## Tests, Benchmarks & Quality Checks

```bash
ruff check src tests
ruff format --check src tests
pytest -v

# Re-run quantization & codegen benchmarks
python -m spectrumbloom.optimization.benchmark --mode codegen --models qwen2.5-coder:1.5b-instruct-q4_K_M qwen3:4b-instruct
python -m spectrumbloom.optimization.vram_budget
python -m spectrumbloom.optimization.qlora --check-only --model-id Qwen/Qwen3-4B
```

---

## Repository Layout

```text
SpectrumBloom/
├── .github/workflows/ci.yml        # Ruff + Pytest + Docker build CI
├── benchmarks/                     # Standardized EDA tasks, JSONL SFT outputs, benchmark JSON/MD results
├── deploy/
│   ├── gcp/                        # Optional Google Cloud Run manifest & guide
│   └── hf-spaces/                  # Optional single-container HF Spaces demo bundle
├── docs/                           # Architecture, reference audit, optimization report
├── examples/                       # Sample e-commerce CSV & REST client workflow script
├── src/spectrumbloom/
│   ├── agent/                      # LangGraph state machine, tool schemas & repair loop
│   ├── api/                        # FastAPI routers & dependency injection
│   ├── core/                       # Settings, security utilities, OTel + Langfuse setup
│   ├── domain/                     # Pydantic contracts, interfaces & typed exceptions
│   ├── eda/                        # Loader, profiler, statistics engine & Plotly charts
│   ├── llm/                        # Ollama, llama.cpp & optional provider factory + prompts
│   ├── ml/                         # Separate leak-safe ML baseline pipeline & permutation imp
│   ├── optimization/               # VRAM estimator, quant benchmarker, distillation & QLoRA guard
│   ├── repositories/               # Memory/Redis session stores & Local/S3 dataset stores
│   ├── sandbox/                    # AST validator, isolated runner, JobObject/rlimits, nsjail cfg
│   ├── services/                   # SessionService & deterministic + DeepEval evaluators
│   └── ui/                         # 4-tab Streamlit REST UI client
├── tests/                          # Unit (sandbox, EDA, ML), integration (FastAPI + OTel) & eval tests
├── Dockerfile                      # Non-root API/UI runtime image
├── Dockerfile.sandbox              # Air-gapped non-root sandbox image
├── docker-compose.yml              # Hardened local stack + optional profiles
├── LICENSE                         # MIT License
└── pyproject.toml                  # Hatchling build & optional extras (eval, redis, s3, optimization)
```

---

## License

Released under the [MIT License](LICENSE). Clean-room implementation inspired conceptually by [Anas-Abd-ElAziz/Data-science-agent](https://github.com/Anas-Abd-ElAziz/Data-science-agent) with independent architecture, local-first LLM optimization, fail-closed security isolation, and comprehensive testing.
