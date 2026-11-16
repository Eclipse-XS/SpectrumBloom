# SpectrumBloom

**Local-first, fail-closed, production-oriented Agentic Exploratory Data Analysis (EDA) platform.**

SpectrumBloom is an independently engineered modular monolith built with **FastAPI**, **Pydantic v2**, **LangGraph**, deterministic statistical & Plotly EDA engines, a **fail-closed sandboxed Python runtime**, **OpenTelemetry + Langfuse + DeepEval** evaluation/observability, and **local quantized Qwen inference** tuned for **4 GB VRAM consumer GPUs** (e.g., NVIDIA RTX 3050 4 GB via **Ollama** or **llama.cpp**), with external cloud LLMs strictly optional.

---

## Architecture Overview

```text
Angular 22 + TypeScript Workspace (frontend/, Nginx :8501 Compose / :4200 dev)
   │ HTTP REST (HttpClient + memory-only Bearer Interceptor)
   ▼
FastAPI Gateway (src/spectrumbloom/api/routes.py)  ── Bearer Auth + CORS + Prod/Host Guards + OTel
   │
   ▼
SpectrumBloomService (src/spectrumbloom/services/session_service.py)
   ├── Repositories (repositories/)       : Thread-safe Memory / Redis sessions + Local / S3 datasets
   ├── Deterministic EDA (eda/)           : Zip-bomb guarded loader, profiler, stats engine, Plotly builder
   ├── Leak-Safe ML Lab (ml/)             : Pipeline + ColumnTransformer CV baseline benchmark + permutation imp
   ├── LangGraph Agent (agent/, llm/)     : Deterministic-first tools + conditional python_repl + timeout/retry bounds
   ├── Isolated Sandbox Boundary (sandbox/):
   │      ├── Compose: RemoteDockerSandbox ─(internal HTTP + token)─▶ sandbox-executor ─▶ DockerSandbox (no sock on API!)
   │      ├── Host:    DockerSandbox (--network=none --read-only --cap-drop=ALL --pids-limit=64 --cpus=1 --memory)
   │      ├── Public:  DisabledSandbox (HF Spaces / Cloud Run — deterministic EDA + ML only, zero arbitrary Python)
   │      └── Dev:     SubprocessSandbox (strict_subprocess restricted to APP_ENV=development; JobObj/POSIX rlimits)
   └── Eval & Telemetry (services/, core/): Deterministic turn verifier, DeepEval Ollama judge, OTel + Langfuse
```
*(Legacy Streamlit UI remains available under Docker Compose `--profile legacy-ui` on port `8502`.)*

See full architectural details in [docs/architecture.md](docs/architecture.md), security & clean-room audit findings in [docs/reference-audit.md](docs/reference-audit.md), and local 4 GB VRAM benchmarks in [docs/optimization-report.md](docs/optimization-report.md).

---

## Interactive UI Workspace

**Angular 22 + TypeScript frontend (`frontend/`):** **Overview → Data Explorer → Automated EDA → AI Analyst → ML Lab → Reports & Export → Sessions & Settings**

Built with standalone Angular components and signals, memory-only Bearer token handling (`AuthTokenService` — never persisted to `localStorage` or bundled assets), `marked` + `DOMPurify` XSS-sanitized markdown (`SafeMarkdownPipe`), lazy-loaded `plotly.js-dist-min` visualizations, light/dark themes, and unprivileged Nginx serving (`nginxinc/nginx-unprivileged:1.27-alpine`). Captured live against FastAPI, `DockerSandbox`, and Ollama (`qwen3:4b-instruct`):

| Overview Workspace (Light) | Automated EDA & Plotly Charts |
| :---: | :---: |
| ![Angular Overview Light](docs/screenshots/angular_overview_light.png) | ![Angular Automated EDA](docs/screenshots/angular_eda_light.png) |

| AI Analyst, Traces & Groundedness Verification | Leak-Safe Baseline ML Lab |
| :---: | :---: |
| ![Angular AI Analyst](docs/screenshots/angular_analyst_light.png) | ![Angular ML Lab](docs/screenshots/angular_ml_light.png) |

| Overview Workspace (Dark Mode) | Mobile Responsive Navigation & Settings (390px) |
| :---: | :---: |
| ![Angular Overview Dark](docs/screenshots/angular_overview_dark.png) | ![Angular Settings Mobile](docs/screenshots/angular_settings_mobile.png) |

---

## Core Capabilities

1. **Deterministic & Visual EDA (`eda/`)**
   - Hardened tabular ingestion (`csv`, `tsv`, `txt`, `xlsx`, `xlsm`, `xls`, `ods`) with byte-size limit, encoding fallback, CSV delimiter sniffing, Excel formula-injection neutralization, and XML zip-bomb bounds (`loader.py`).
   - Semantic column typing (`numeric_continuous`, `numeric_discrete`, `categorical`, `boolean`, `datetime`, `identifier`, `text`, `empty`), skewness/kurtosis, and data quality alerts (`profiler.py`).
   - Deep statistical diagnostics (`statistics.py`): missingness co-occurrence pairs, Pearson & Spearman correlations, Cramér's V categorical associations, Shapiro-Wilk / D'Agostino normality tests, IQR / Z-score / robust MAD outliers, `IsolationForest` multivariate anomalies, and Kruskal-Wallis group tests.
   - Interactive Plotly visualizations (`charts.py`): missingness heatmap, numeric distributions, correlation matrix, box/outlier diagnostics, top categorical breakdowns, and standalone HTML chart exports.

2. **Deterministic-First Agentic EDA (`agent/`, `llm/`)**
   - Local-first inference via `UnifiedLLMProvider` supporting **Ollama** (`qwen3:4b-instruct` default, `num_ctx=4096`), **llama.cpp** OpenAI-compatible server, and optional external **OpenAI-compatible** or **Gemini** endpoints, guarded by configurable `LLM_TIMEOUT_SECONDS` (`LLMTimeoutError`).
   - Stateful LangGraph orchestration (`EDAAgentGraph`) preferring deterministic tools (`get_statistical_summary`, `aggregate_data`, `compute_statistical_test`, `create_standard_chart`, `run_ml_baseline`) and invoking sandboxed `python_repl` only for custom calculations/plots.
   - Supports native tool-calling plus `<tool_call>` XML, bare JSON, ````json`, and ````python` fallback parsing, context-window compaction (`num_ctx=4096`), bounded self-repair retries (`MAX_REPAIR_RETRIES = 2`), explicit `FIGURES_GENERATED` accounting, and fail-closed error states.

3. **Fail-Closed Python Execution Sandbox (`sandbox/`)**
   - **Never** falls back or downgrades to unrestricted execution when a configured isolation backend is unavailable (`SandboxUnavailableError`).
   - Static AST gate rejects disallowed imports, dunder traversal, `eval`/`exec`/`getattr`/`open`, and pandas `df.eval()` / `df.query()`, transferring DataFrames strictly via PyArrow Parquet / typed JSON (never `pickle`).
   - **Isolation Boundaries:**
     - `remote` (`RemoteDockerSandbox` → `sandbox/executor_app.py`, Compose default): isolates the Docker daemon behind an internal authenticated service so `spectrumbloom-api` never mounts `/var/run/docker.sock`.
     - `docker` (`DockerSandbox` + `Dockerfile.sandbox`): non-privileged per-run Linux container (`--network=none --read-only --cap-drop=ALL --security-opt=no-new-privileges:true --user=10001:10001 --pids-limit=64 --cpus=1 --memory=... --tmpfs=/tmp:rw,noexec,nosuid,size=16m`), bytecode-compiled wheels + lazy imports for fast cold startup.
     - `disabled` (`DisabledSandbox`): disables dynamic `python_repl` in multi-tenant/serverless public deployments (HF Spaces / Cloud Run) while keeping all deterministic EDA, ML Lab, and deterministic copilot tools working.
     - `strict_subprocess` (local `APP_ENV=development` only — rejected everywhere else): **process-level defense-in-depth**, *not* OS container isolation (`python -I`, stripped env, socket/DNS kill-switch, prefix-locked `open()`, Windows Kernel Job Objects or POSIX `rlimit` + RSS watchdog).
     - `nsjail` (`sandbox/nsjail.cfg`): optional Linux namespace adapter (implemented fail-closed; not E2E tested without a native `nsjail` binary).

4. **Separate Baseline ML Lab (`ml/experiments.py`)**
   - Leak-safe `ColumnTransformer` + `Pipeline` with cross-validation (`Dummy`, `LogisticRegression`/`Ridge`, `RandomForest`, `HistGradientBoosting`), identifier-column leakage dropping, held-out test metrics, permutation feature importance, and interactive Plotly diagnostics.

5. **Evaluation, Observability & Optimization (`services/eval_service.py`, `optimization/`)**
   - Automated per-turn verification (tool execution success, Plotly JSON validity, tolerance-aware numeric grounding, `<think>` leakage & ungrounded chart-reference guards) + optional `DeepEval` `GEval` groundedness scoring backed by a local Ollama judge (`OllamaDeepEvalJudge`).
   - OpenTelemetry trace/metric instrumentation (`core/telemetry.py`, compatible with Jaeger / OTLP) and optional Langfuse callbacks.

---

## Empirical End-to-End Agent Benchmark (67 Tasks × 6 Datasets, RTX 3050 4 GB)

Evaluated across **67 analytical tasks spanning 6 diverse datasets** (`sample_ecommerce.csv`, `palmer_penguins.csv`, `breast_cancer_uci.csv`, `wine_cultivars_uci.csv`, `us_macro_fred.csv`, `mauna_loa_co2.csv`; provenance & SHA-256 checksums in [`benchmarks/datasets/PROVENANCE.md`](benchmarks/datasets/PROVENANCE.md)) over **2 back-to-back repeats** inside `DockerSandbox` (`Ollama`, `num_ctx=4096`):

| Model | Run | Correctness | Tool Exec (Std / All) | Retry Rate | Timeouts | P50 (s) | P95 (s) | GPU Offload | Peak VRAM (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen2.5-coder:1.5b-instruct-q4_K_M` | 1 | 48/67 (71.6%) | 51.7% / 51.9% | 47.8% | 0 | 4.76 | 22.66 | 100.0% | 1390.0 |
| `qwen2.5-coder:1.5b-instruct-q4_K_M` | 2 | 47/67 (70.1%) | 51.4% / 51.6% | 47.8% | 0 | 4.96 | 14.28 | 100.0% | 1489.0 |
| `qwen2.5-coder:3b-instruct-q4_K_M` | 1 | 47/67 (70.1%) | 77.1% / 71.0% | 20.9% | 0 | 4.59 | 14.43 | 100.0% | 2407.0 |
| `qwen2.5-coder:3b-instruct-q4_K_M` | 2 | 47/67 (70.1%) | 78.3% / 72.0% | 19.4% | 0 | 4.40 | 12.18 | 100.0% | 2407.0 |
| **`qwen3:4b-instruct` (Default)** | **1** | **61/67 (91.0%)** | **90.8% / 90.8%** | **9.0%** | **0** | **11.72** | **33.91** | 66.3% | 2623.0 |
| **`qwen3:4b-instruct` (Default)** | **2** | **61/67 (91.0%)** | **90.6% / 90.6%** | **9.0%** | **0** | **17.49** | **40.75** | 66.3% | 2617.0 |

- **`qwen3:4b-instruct` selected as empirical default:** In the initial 3-model baseline (`2026-11-14`), `qwen3:4b-instruct` reached **61/67 (91.0%)** with `90.6–90.8%` tool execution, outperforming 1.5B (`70.1–71.6%`) and 3B (`70.1%`, `0/3` multi-step).

### Post-Hardening Evaluation (`qwen3:4b-instruct`, `DockerSandbox`, `num_ctx=4096`, `2026-11-15`)

After fixing analytical aggregation/correlation/chart-range tools, normalizing evaluation entity/exponent parsing, adding atomic session/dataset concurrency controls, and building a new 28-task out-of-dataset generalization suite (`benchmarks/eval_suite_independent.json`), **`qwen3:4b-instruct`** achieved the following measured results inside `DockerSandbox` (`num_ctx=4096`):

| Suite | Run | Complete | Answer | Tool Exec | Tool Sel | Grounded | Chart | Retry Rate | Timeouts | P50 (s) | P95 (s) | Peak VRAM (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dev 67 Tasks (`latest_workflow_benchmark`) | 1 | **67/67 (100.0%)** | **67/67 (100.0%)** | **97.0%** | **100.0%** | **100.0%** | **100.0%** | 3.0% | 0 | 11.25 | 23.01 | 2341.0 |
| Dev 67 Tasks (`latest_workflow_benchmark`) | 2 | **67/67 (100.0%)** | **67/67 (100.0%)** | **97.0%** | **100.0%** | **100.0%** | **100.0%** | 3.0% | 0 | 11.06 | 23.45 | 2341.0 |
| Holdout 30 Tasks (`holdout_workflow_benchmark`) | 1 | **29/30 (96.7%)** | **29/30 (96.7%)** | **97.3%** | **100.0%** | **100.0%** | **100.0%** | 3.3% | 0 | 13.55 | 28.57 | 2337.0 |
| Independent 28 Tasks (`independent_workflow_benchmark`) | 1 | **28/28 (100.0%)** | **28/28 (100.0%)** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **0.0%** | 0 | **9.75** | **15.66** | 2337.0 |
| Independent 28 Tasks (`independent_workflow_benchmark`) | 2 | **28/28 (100.0%)** | **28/28 (100.0%)** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **0.0%** | 0 | **9.59** | **16.36** | 2337.0 |

See [docs/optimization-report.md](docs/optimization-report.md) for per-category scores, honest remaining Holdout variance (`hold_co2_04_diff`), historical 8-task quantization sweeps, analytical 4 GB VRAM/QLoRA budgets, and execution-verified SFT trajectories ([`benchmarks/distilled_eda_sft.jsonl`](benchmarks/distilled_eda_sft.jsonl)).

---

## Verification, Estimates & Implementation Status

| Component | Status | Evidence / Scope |
| --- | --- | --- |
| FastAPI REST + Deterministic EDA + Leak-Safe ML + Auth Guards | Verified | Unit & integration tests (`tests/`), strict prod/non-loopback auth guards |
| Angular 22 + TypeScript Workspace (`frontend/`) | Verified | Typecheck, Jasmine/Karma specs, production `ng build`, deterministic CI Playwright suite + live FastAPI/Ollama/Docker Playwright E2E |
| Security Isolation & Integrity Suite (atomic upload, optimistic concurrency, cache eviction, sandbox & auth bounds) | Verified | `tests/unit/test_concurrency_and_integrity.py`, `tests/unit/test_sandbox_security.py`, `tests/unit/test_security_config.py`, `tests/integration/test_docker_sandbox.py` |
| Isolated Docker Compose Stack (`frontend` Nginx `:8501` → `api` → `sandbox-executor` → `spectrumbloom-sandbox`) | Verified E2E | Live `docker compose up -d --build --wait`, `/health`, `401` auth guard, sample CSV upload, Playwright UI smoke (zero token in `localStorage`/bundle), live `python_repl` query via `remote` sandbox |
| 67-Task Dev (`×2`) + 30-Task Holdout + 28-Task Independent (`×2`) Agent Evaluations | Measured | `qwen3:4b-instruct` inside `DockerSandbox` (`67/67 = 100.0%` Dev, `29/30 = 96.7%` Holdout w/ `97.3%` tool exec, `28/28 = 100.0%` Independent) in [`latest_workflow_benchmark.json`](benchmarks/results/latest_workflow_benchmark.json), [`holdout_workflow_benchmark.json`](benchmarks/results/holdout_workflow_benchmark.json) & [`independent_workflow_benchmark.json`](benchmarks/results/independent_workflow_benchmark.json) |
| Historical 8-Task Quantization Sweep (`Q2_K` / `Q4_K_M` / `Q8_0` / `Qwen3-4B`) | Measured | Baseline microbenchmark in [`benchmarks/results/latest_benchmark.json`](benchmarks/results/latest_benchmark.json) & `latest_tool_benchmark.json` |
| Inference VRAM Budget & QLoRA Memory Table (`vram_budget.py`, `qlora.py`) | Analytical estimate | Formula-based VRAM bounds + `qlora.py --check-only`; CUDA weight updates **not run** |
| SFT Trajectory Distillation (`distillation.py`) | Executed (dataset only) | Generated `6/8` execution-verified JSONL examples; downstream PyTorch QLoRA fine-tuning **not run** |
| `NsjailSandbox`, `RedisSessionRepository`, `S3DatasetStore`, `llama.cpp`, External OpenAI/Gemini, Live OTLP/Langfuse/DeepEval, Cloud Run, HF Spaces | Implemented, not E2E tested live | Code/adapters/manifests implemented & tested offline or fail-closed (`SANDBOX_BACKEND=disabled` for public demos); live remote deployments not run |

---

## Quickstart (Local Setup)

### 1. Prerequisites
- Python **3.12** (supported: `>=3.11,<3.14`) and [`uv`](https://docs.astral.sh/uv/)
- **Node.js 22** & `npm` (for the Angular 22 frontend workspace in `frontend/`)
- **Docker** (required for default `SANDBOX_BACKEND=docker` and isolated Compose execution; local `APP_ENV=development` workstations without Docker can explicitly opt into `SANDBOX_BACKEND=strict_subprocess`)
- Local **Ollama** (`ollama pull qwen3:4b-instruct`) or **llama.cpp** OpenAI-compatible server

### 2. Install & Configure
```bash
uv venv --python 3.12 .venv
# Linux / macOS: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1

uv pip install -e ".[eval,dev]"
cp .env.example .env
docker build -f Dockerfile.sandbox -t spectrumbloom-sandbox:latest .

# Install Angular frontend dependencies
cd frontend && npm ci && cd ..
```

### 3. Start Backend API & Angular Workspace
Terminal 1 — FastAPI backend:
```bash
uvicorn spectrumbloom.main:app --host 127.0.0.1 --port 8000
```
Terminal 2 — Angular 22 frontend (dev proxy `/api` + `/health` -> `127.0.0.1:8000`):
```bash
cd frontend && npm start
```
Then open `http://localhost:4200` (API interactive docs at `http://127.0.0.1:8000/docs`), or run the standalone REST workflow script:
```bash
python examples/sample_api_workflow.py
```
*(Legacy Streamlit UI remains runnable locally with `streamlit run src/spectrumbloom/ui/app.py --server.port 8502`.)*

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
  Set `API_AUTH_TOKEN` and `SANDBOX_EXECUTOR_TOKEN` (>= 16 random chars) in a gitignored `.env`, then start the Angular Nginx UI (`http://127.0.0.1:8501`), FastAPI (`127.0.0.1:8000`), and isolated `sandbox-executor`:
  ```bash
  docker compose up -d --build --wait
  ```
  Optional profiles add the legacy Streamlit UI (`--profile legacy-ui` on `127.0.0.1:8502`), Redis persistence (`--profile persistence`), containerized Ollama (`--profile local-llm`), or Jaeger OpenTelemetry tracing (`--profile observability`).
- **Optional Hugging Face Spaces Demo:** See [`deploy/hf-spaces/README.md`](deploy/hf-spaces/README.md).
- **Optional Google Cloud Run:** See [`deploy/gcp/README.md`](deploy/gcp/README.md) (no mandatory paid services required for local use).

---

## Tests, Benchmarks & Quality Checks

```bash
ruff check src tests benchmarks/prepare_datasets.py
ruff format --check src tests benchmarks/prepare_datasets.py
pytest -v

# Angular 22 frontend checks & E2E (deterministic + live FastAPI/Ollama/Docker)
cd frontend
npm run typecheck
npx tsc --noEmit -p tsconfig.spec.json
npm run test:ci
npm run build
npm run e2e
cd ..

# Re-prepare all evaluation datasets and run the 67-task Dev, 30-task Holdout, and 28-task Independent benchmarks
python benchmarks/prepare_datasets.py
python -m spectrumbloom.optimization.workflow_benchmark --sandbox-backend docker --num-ctx 4096 --repeats 2
python -m spectrumbloom.optimization.workflow_benchmark --suite benchmarks/eval_suite_holdout_30.json --models qwen3:4b-instruct --sandbox-backend docker --num-ctx 4096 --repeats 1 --output-prefix holdout_workflow_benchmark
python -m spectrumbloom.optimization.workflow_benchmark --suite benchmarks/eval_suite_independent.json --models qwen3:4b-instruct --sandbox-backend docker --num-ctx 4096 --repeats 2 --output-prefix independent_workflow_benchmark
python -m spectrumbloom.optimization.vram_budget
python -m spectrumbloom.optimization.qlora --check-only --model-id Qwen/Qwen3-4B
```

---

## Repository Layout

```text
SpectrumBloom/
├── .github/workflows/ci.yml        # Backend Ruff/Pytest + Frontend CI (tsc/Karma/ng build) + Docker builds
├── benchmarks/                     # 67-task Dev, 30-task Holdout & 28-task Independent suites, provenance, results
├── deploy/
│   ├── gcp/                        # Optional Google Cloud Run manifest & guide (SANDBOX_BACKEND=disabled)
│   └── hf-spaces/                  # Optional single-container HF Spaces demo (SANDBOX_BACKEND=disabled)
├── docs/                           # Architecture, reference audit, optimization report & UI screenshots
├── examples/                       # Sample e-commerce CSV & REST client workflow script
├── frontend/                       # Angular 22 + TypeScript workspace, Karma/Playwright tests, unprivileged Nginx Dockerfile
├── src/spectrumbloom/
│   ├── agent/                      # LangGraph state machine, deterministic/sandbox tools & repair loop
│   ├── api/                        # FastAPI routers & dependency injection
│   ├── core/                       # Settings, security utilities, OTel + Langfuse setup
│   ├── domain/                     # Pydantic contracts, interfaces & typed exceptions
│   ├── eda/                        # Loader, profiler, statistics engine & Plotly charts
│   ├── llm/                        # Ollama, llama.cpp & optional provider factory + prompts
│   ├── ml/                         # Separate leak-safe ML baseline pipeline & permutation imp
│   ├── optimization/               # VRAM estimator, workflow/quant benchmarkers, distillation & QLoRA guard
│   ├── repositories/               # Memory/Redis session stores & Local/S3 dataset stores
│   ├── sandbox/                    # AST gate, isolated runner, Docker/Remote executor service, JobObj/rlimits
│   ├── services/                   # SessionService & deterministic + DeepEval evaluators
│   └── ui/                         # Legacy Streamlit UI (optional Compose legacy-ui profile)
├── tests/                          # Security, sandbox, EDA, ML, API workflow & Docker isolation tests
├── Dockerfile                      # Non-root API runtime image
├── Dockerfile.executor             # Isolated internal Docker executor service image
├── Dockerfile.sandbox              # Air-gapped non-root sandbox image (--compile-bytecode)
├── docker-compose.yml              # Multi-network Compose stack (frontend -> api -> sandbox-executor -> sandbox)
├── LICENSE                         # MIT License
└── pyproject.toml                  # Hatchling build & optional extras (eval, redis, s3, optimization)
```

---

## License

Released under the [MIT License](LICENSE). Clean-room implementation inspired conceptually by [Anas-Abd-ElAziz/Data-science-agent](https://github.com/Anas-Abd-ElAziz/Data-science-agent) with independent architecture, local-first LLM optimization, fail-closed security isolation, and comprehensive testing.
