# SpectrumBloom Architecture

SpectrumBloom is a modular monolith engineered for **local-first, secure, observable Exploratory Data Analysis (EDA)** on consumer hardware (specifically tuned for 4 GB VRAM GPUs such as the NVIDIA RTX 3050 Laptop GPU).

## Component Layers

```text
+-------------------------------------------------------------------+
|                     Streamlit UI (src/spectrumbloom/ui)            |
|  5-tab studio: Overview -> Auto EDA -> Ask AI -> ML Lab -> Export |
+---------------------------------+---------------------------------+
                                  | HTTP / JSON (httpx + Bearer)
                                  v
+-------------------------------------------------------------------+
|                     FastAPI Layer (src/spectrumbloom/api)         |
|  - Bearer token guard, CORS policy, prod/host auth validation     |
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
| Repositories   | | Deterministic | | Agentic Engine & Isolation   |
| - SessionRepo  | | EDA & ML      | | - LangGraph EDAAgentGraph    |
|   (Mem / Redis)| | - loader.py   | | - UnifiedLLMProvider         |
| - DatasetStore | | - profiler.py | |   (Ollama/llama.cpp/Opt)     |
|   (Local / S3) | | - statistics  | | - RemoteDockerSandbox client |
|                | | - charts.py   | |   -> sandbox-executor :9100  |
|                | | - ml/         | |   -> DockerSandbox (no sock  |
|                | |               | |      mounted on API host!)   |
+----------------+ +---------------+ +------------------------------+
```

## Key Architectural Flows

### 1. Upload & Deterministic EDA (`POST /api/v1/sessions/{id}/upload`)
1. `eda.loader.parse_tabular_bytes` enforces byte-size caps (`MAX_UPLOAD_MB`), allowed extensions, UTF-8/CP1252/Latin-1 decoding, CSV delimiter sniffing, and OOXML/ODF zip-bomb protection before invoking `pandas` / `openpyxl`.
2. `eda.profiler.profile_dataframe` infers semantic roles (`numeric_continuous`, `numeric_discrete`, `categorical`, `boolean`, `datetime`, `identifier`, `text`) and generates dataset quality warnings.
3. `eda.statistics.compute_statistics` computes missingness co-occurrence, Pearson & Spearman correlations, Cramér's V categorical associations, Shapiro-Wilk / D'Agostino normality tests, IQR / Z-score / robust MAD outliers, `IsolationForest` multivariate anomalies, and Kruskal-Wallis group difference tests.
4. `eda.charts.build_eda_figures` creates ready-to-render interactive Plotly visualizations.

### 2. Agentic EDA & Bounded Repair Loop (`POST /api/v1/sessions/{id}/query`)
1. `EDAAgentGraph` injects the compact schema & profile summary into the system prompt so small local models (default `qwen3:4b-instruct` at `num_ctx=4096`) ground exact column names without extra discovery turns, and enforces explicit rules against reusing unfiltered profile metrics on conditional/filtered follow-up questions.
2. Deterministic-first tool suite (`agent/tools.py`, `agent/deterministic_tools.py`):
   - `get_statistical_summary` -> cached dataset profile, missingness, correlations, outliers, normality, group tests
   - `aggregate_data` -> deterministic multi-condition filtering (`filters`), `group_by`, `agg_func`, top/bottom sorting
   - `compute_statistical_test` -> deterministic Pearson/Spearman correlation, Shapiro normality, t-test/Mann-Whitney, Kruskal-Wallis, IQR outlier inspection
   - `create_standard_chart` -> validated histogram, box, bar, scatter, correlation heatmap Plotly specs
   - `run_ml_baseline` -> separate leak-safe `ml/experiments.py` cross-validated modeling
   - `python_repl` -> routed to the fail-closed `CodeSandbox` only when deterministic tools cannot satisfy custom logic/charts
3. If native `tool_calls` are omitted by a smaller GGUF backend, `extract_tool_calls_with_fallback` parses `<tool_call>` XML payloads, bare/`json` tool-call blocks, or ````python```` fenced blocks.
4. Configurable `LLM_TIMEOUT_SECONDS` enforces bounded inference wait times (`LLMTimeoutError`), while invalid code/tool errors trigger bounded self-repair retries (`MAX_REPAIR_RETRIES = 2`, `MAX_TOOL_STEPS = 6`), explicit `FIGURES_GENERATED` accounting, and preserved failure states rather than fabricated conclusions.
5. Every turn is scored by `eval_service.evaluate_turn_deterministically` and traced via OpenTelemetry spans/metrics (plus optional Langfuse callbacks).

### 3. Fail-Closed Code Sandbox (`sandbox/`)
- **Isolation Tiers (Explicit Boundary Distinction):**
  - `remote` (`RemoteDockerSandbox` -> `sandbox/executor_app.py`, Docker Compose default): Separates the trust boundary over an internal network (`sandbox_internal`) authenticated by `SANDBOX_EXECUTOR_TOKEN` so **`spectrumbloom-api` never mounts `/var/run/docker.sock`**. The executor spawns ephemeral `DockerSandbox` sibling containers.
  - `docker` (`DockerSandbox` + `Dockerfile.sandbox`): Provides **non-privileged container isolation per execution**: `--network=none`, `--read-only`, `--cap-drop=ALL`, `--security-opt=no-new-privileges:true`, `--user=10001:10001`, `--pids-limit=64`, `--cpus=1`, hard cgroup `--memory`/`--memory-swap`, and ephemeral `/tmp` tmpfs. Python wheels and `/app/src` are bytecode-compiled (`--compile-bytecode` + `compileall`) and `runner.py` defers `plotly`/`scipy.stats` imports unless referenced, cutting container execution time from `>30s` cold timeouts to `~7.7s` under `--cpus=1 --read-only`.
  - `disabled` (`DisabledSandbox`): Explicitly disables agent-generated `python_repl` (`SandboxUnavailableError`) in multi-tenant / serverless public deployments (`deploy/hf-spaces`, `deploy/gcp`) while keeping deterministic EDA, ML Lab, and deterministic copilot tools available.
  - `strict_subprocess`: Explicitly restricted to trusted local development (`APP_ENV=development` only); provides **application-level process defense-in-depth** (*not* OS container isolation). Any non-development environment attempting `strict_subprocess` fails closed (`SandboxUnavailableError`), and SpectrumBloom never silently downgrades isolation backends.
  - `nsjail` (`NsjailSandbox` + `sandbox/nsjail.cfg`): Optional Linux namespace/rlimit confinement where installed (implemented fail-closed; untested without a native Linux `nsjail` binary).
- **Static AST Gate & Runner Hardening:**
  - Rejects disallowed imports, dunder attributes, reflection builtins (`getattr`, `eval`, `exec`, `open`), file I/O (`read_*`, `to_csv`, `to_pickle`), and pandas expression evaluation (`df.eval`, `df.query`).
  - DataFrames cross via PyArrow Parquet (with safe typed JSON fallback), never `pickle`.
  - Runs under `python -I` isolated mode with socket/DNS kill-switches, prefix-locked read-only `open()`, figure collection across `plotly_figures`, `_sb_last_expr`, and top-level figure bindings (`fig`), plus OS Kernel Job Objects (Windows) or POSIX `rlimit` + RSS watchdog (`strict_subprocess`).
