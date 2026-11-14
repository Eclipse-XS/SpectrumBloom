# SpectrumBloom Architecture

SpectrumBloom is a modular monolith engineered for **local-first, secure, observable Exploratory Data Analysis (EDA)** on consumer hardware (specifically tuned for 4 GB VRAM GPUs such as the NVIDIA RTX 3050 Laptop GPU).

## Component Layers

```text
+-------------------------------------------------------------------+
|                     Streamlit UI (src/spectrumbloom/ui)            |
+---------------------------------+---------------------------------+
                                  | HTTP / JSON (httpx)
                                  v
+-------------------------------------------------------------------+
|                     FastAPI Layer (src/spectrumbloom/api)         |
|  - Bearer token guard, CORS policy, OTel FastAPI instrumentation  |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|               Application Service (services/session_service.py)   |
|  - Per-session RLock, EDA caching, Agent orchestration, Eval      |
+--------+-----------------+-----------------+----------------------+
         |                 |                 |
         v                 v                 v
+----------------+ +---------------+ +------------------------------+
| Repositories   | | Deterministic | | Agentic Engine & Sandbox     |
| - SessionRepo  | | EDA & ML      | | - LangGraph StateGraph       |
|   (Mem / Redis)| | - loader.py   | | - UnifiedLLMProvider         |
| - DatasetStore | | - profiler.py | |   (Ollama / llama.cpp / Opt) |
|   (Local / S3) | | - statistics  | | - Fail-Closed CodeSandbox    |
|                | | - charts.py   | |   (AST + Parquet + JobObj /  |
|                | | - ml/         | |    POSIX rlimits / Docker)   |
+----------------+ +---------------+ +------------------------------+
```

## Key Architectural Flows

### 1. Upload & Deterministic EDA (`POST /api/v1/sessions/{id}/upload`)
1. `eda.loader.parse_tabular_bytes` enforces byte-size caps (`MAX_UPLOAD_MB`), allowed extensions, UTF-8/CP1252/Latin-1 decoding, CSV delimiter sniffing, and OOXML/ODF zip-bomb protection before invoking `pandas` / `openpyxl`.
2. `eda.profiler.profile_dataframe` infers semantic roles (`numeric_continuous`, `numeric_discrete`, `categorical`, `boolean`, `datetime`, `identifier`, `text`) and generates dataset quality warnings.
3. `eda.statistics.compute_statistics` computes missingness co-occurrence, Pearson & Spearman correlations, Cramér's V categorical associations, Shapiro-Wilk / D'Agostino normality tests, IQR / Z-score / robust MAD outliers, `IsolationForest` multivariate anomalies, and Kruskal-Wallis group difference tests.
4. `eda.charts.build_eda_figures` creates ready-to-render interactive Plotly visualizations.

### 2. Agentic EDA & Self-Repair Loop (`POST /api/v1/sessions/{id}/query`)
1. `EDAAgentGraph` injects the compact schema & profile summary into the system prompt so small local models (e.g., `qwen3:4b-instruct` at `num_ctx=4096`) do not waste turns guessing column names.
2. Bound tools:
   - `python_repl` -> routed to the fail-closed `CodeSandbox`
   - `get_statistical_summary` -> returns cached deterministic statistical insights
   - `run_ml_baseline` -> triggers the separate leak-safe `ml/experiments.py` module
3. If native `tool_calls` are omitted by a smaller GGUF backend, `extract_tool_calls_with_fallback` parses `<tool_call>` XML payloads, bare/`json` tool-call blocks, or ````python```` fenced blocks.
4. Sandbox errors trigger bounded self-repair retries (`MAX_REPAIR_RETRIES = 2`).
5. Every turn is scored by `eval_service.evaluate_turn_deterministically` and traced via OpenTelemetry spans/metrics (plus optional Langfuse callbacks).

### 3. Fail-Closed Code Sandbox (`sandbox/`)
- **Isolation Tiers (Explicit Boundary Distinction):**
  - `strict_subprocess` provides **application-level process defense-in-depth** (not OS kernel/container isolation). It reduces risk when Docker/nsjail are unavailable on a local developer workstation, but does **not** replace container namespaces, seccomp, or filesystem mounts.
  - `docker` (`DockerSandbox` + `Dockerfile.sandbox`) provides **non-privileged container isolation per execution**: `--network=none`, `--read-only`, `--cap-drop=ALL`, `--security-opt=no-new-privileges:true`, `--pids-limit=64`, `--cpus=1`, hard cgroup `--memory`/`--memory-swap`, and ephemeral `/tmp` tmpfs.
  - `nsjail` (`NsjailSandbox` + `sandbox/nsjail.cfg`) provides optional Linux namespace/rlimit confinement where installed (implemented fail-closed; untested without a native Linux `nsjail` binary).
  - Neither backend falls back to unrestricted host execution (`SandboxUnavailableError`).
- **Static AST Gate:** Blocks disallowed imports, dunder attributes, reflection builtins (`getattr`, `eval`, `exec`, `open`), file I/O (`read_*`, `to_csv`, `to_pickle`), and pandas expression evaluation (`df.eval`, `df.query`).
- **Pickle-Free IPC:** DataFrames cross the execution boundary via PyArrow Parquet (with safe typed JSON fallback), never `pickle`.
- **Runtime Hardening Inside Runner:**
  - `python -I` isolated mode with stripped environment variables (`OPENBLAS_NUM_THREADS=1`).
  - Socket class & DNS resolution replaced with `PermissionError` kill-switches.
  - Filesystem `open()` restricted strictly to read-only access under `sys.prefix` / `sys.base_prefix`.
  - **Windows (`strict_subprocess`):** Kernel Job Objects (`JOB_OBJECT_LIMIT_JOB_MEMORY | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_ACTIVE_PROCESS`) enforce memory and process-tree termination; process creation and `AssignProcessToJobObject` failures fail closed (`SandboxUnavailableError`).
  - **POSIX / Linux (`strict_subprocess`):** `RLIMIT_CPU`, `RLIMIT_FSIZE`, high virtual-arena safety cap `RLIMIT_AS` (allowing OpenBLAS/PyArrow/Plotly shared mappings without virtual-address starvation), process-group `SIGKILL`, and `/proc/<pid>/statm` + child-tree physical RSS watchdog.
