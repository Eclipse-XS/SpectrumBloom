# SpectrumBloom Architecture

SpectrumBloom is a modular monolith engineered for **local-first, secure, observable Exploratory Data Analysis (EDA)** on consumer hardware (specifically tuned for 4 GB VRAM GPUs such as the NVIDIA RTX 3050 Laptop GPU).

## Component Layers

```text
+-------------------------------------------------------------------+
|        Angular 22 Workspace (frontend/, Nginx :8501 / dev :4200)  |
|  Overview -> Explorer -> Auto EDA -> AI Analyst -> ML Lab -> Export|
|  (Legacy Streamlit UI available via Compose legacy-ui on :8502)   |
+---------------------------------+---------------------------------+
                                  | HTTP / JSON (HttpClient + Bearer)
                                  v
+-------------------------------------------------------------------+
|                     FastAPI Layer (src/spectrumbloom/api)         |
|  - Bearer token guard, CORS, 64 KB streaming upload limit check   |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|               Application Service (services/session_service.py)   |
|  - Atomic staged upload/rollback, lock-first optimistic version   |
|  - Bounded TTL/LRU runtime cache, turn idempotency, LangGraph sync|
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

### 1. Atomic Upload & Deterministic EDA (`POST /api/v1/sessions/{id}/upload`)
1. `api.routes` enforces `MAX_UPLOAD_MB` in `64 KB` streaming chunks before buffering the full upload in memory (aligned with Nginx `client_max_body_size 256M`), and `eda.loader.parse_tabular_bytes` enforces allowed extensions, UTF-8/CP1252/Latin-1 decoding, CSV delimiter sniffing, and OOXML/ODF zip-bomb protection before invoking `pandas` / `openpyxl`.
2. `SpectrumBloomService.upload_dataset` stages storage writes (`{digest}_{nonce}_{safe_name}` via atomic `os.replace`), computes profiling (`eda.profiler.profile_dataframe`), statistics (`eda.statistics.compute_statistics`), and Plotly figures (`eda.charts.build_eda_figures`), and persists the updated `SessionRecord` with optimistic version checking (`expected_version`, Redis `WATCH`/`MULTI`/`EXEC`) inside per-session locking. Failures during storage, profiling, statistics, figure generation, or persistence roll back staged files and leave previous dataset metadata, EDA results, and conversation history intact; expired/excess entries in `_runtime_cache` are evicted by TTL and bounded LRU (`MAX_CACHED_SESSIONS`).
3. Restored sessions (including Redis-backed restarts) re-seed LangGraph thread checkpoints via `EDAAgentGraph.seed_history_if_empty` and enforce `client_turn_id` idempotency to prevent duplicate turns on network retries.

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

### 4. Production Angular 22 Frontend (`frontend/`)
- **Workspace Modules (`frontend/src/app/`):** Standalone Angular 22 components and operation-scoped signal/service state management (`WorkspaceStateService`, `ThemeService`) organized into lazy-routed workspace views: `/onboarding`, `/overview`, `/explorer`, `/eda`, `/analyst`, `/ml`, `/reports`, `/sessions`, and `/settings`. Cancellation subjects (`cancelUpload$`, `cancelQuery$`, `cancelMl$`, `cancelSessionLoads$`), operation IDs, `finalize()` cleanups, and session/SHA-256 ownership guards prevent stale cross-session results and permanent loading states, while `client_turn_id` + history reconciliation handle uncertain timeouts without duplicate turns.
- **Auth & Security Boundary:** `AuthTokenService` keeps `API_AUTH_TOKEN` strictly in an Angular signal in session memory (never written to `localStorage`, URL params, or built bundle assets). `AuthInterceptor` injects `Authorization: Bearer` headers only for relative `/api/*` requests or configured trusted API origins, triggering `AuthBannerComponent` ("API authentication required") on HTTP `401` challenges. Note that the current shared `API_AUTH_TOKEN` protects local/single-tenant or trusted-team deployments and does not provide multi-user per-identity RBAC or tenant isolation.
- **Markdown XSS, Offline Export & Visualization Isolation:** `SafeMarkdownPipe` renders assistant responses with `marked` and sanitizes the HTML output through a `DOMPurify` tag/attribute allowlist. `PlotlyChartComponent` (`sb-plotly-chart`) lazily imports `plotly.js-dist-min` and adapts chart layouts across light/dark themes. Standalone HTML figure exports attempt to bundle inline `plotly.min.js` for offline rendering and surface a clear external CDN fallback notice when bundled offline assets are unavailable.
- **Containerized Serving (`frontend/Dockerfile`, `frontend/nginx.conf`):** Built in a multi-stage Node 22 build and served by non-root `nginxinc/nginx-unprivileged:1.27-alpine` on port `8501` (`client_max_body_size 256M`), proxying `/api/` (with `proxy_pass_header Authorization` and `240s` read timeout) and `/health` over `app_net` to `api:8000`.
