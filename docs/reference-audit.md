# Reference Repository Audit (`Anas-Abd-ElAziz/Data-science-agent`)

This document summarizes the architectural, dependency, security, and licensing audit of [`Anas-Abd-ElAziz/Data-science-agent`](https://github.com/Anas-Abd-ElAziz/Data-science-agent) conducted prior to designing **SpectrumBloom**.

## 1. Licensing & Clean-Room Decision

- **Observation:** GitHub repository metadata reported `"licenseInfo": null` because there is no standalone `LICENSE` file in the repository root, although `README.md` includes a trailing `MIT` heading.
- **Decision:** Because standalone license files are the authoritative legal signal for code reuse, **SpectrumBloom** reuses **zero source files** from the reference repository and was implemented independently from scratch under a clean root [LICENSE](../LICENSE) (MIT).

## 2. Architectural Findings

| Area | Reference (`Data-science-agent`) | SpectrumBloom Redesign |
| --- | --- | --- |
| Layering | Monolithic `AgentSession` god-object (`agent/service.py`) mixing DataFrame I/O, LLM creation, LangGraph, and state; module-level mutable `_sessions` dict in `api.py` | Clean modular monolith: `core/` -> `domain/` -> `repositories/` -> `eda/` & `ml/` & `sandbox/` & `llm/` & `agent/` -> `services/` -> `api/` |
| Frontend Architecture | `streamlit_app.py` bypasses `api.py` entirely and runs separate in-memory sessions | `ui/app.py` communicates strictly over REST with the FastAPI service |
| LLM Provider | Hardcoded to `ChatGoogleGenerativeAI` (`gemini-3.1-flash-lite-preview`), requiring an external API key per session | Local-first `UnifiedLLMProvider` targeting quantized Qwen models via **Ollama** or **llama.cpp**, with OpenAI/Gemini optional |
| EDA Capabilities | No deterministic profiler; every summary/plot forced through LLM code generation | Dedicated deterministic `eda/` engine (`loader`, `profiler`, `statistics`, `charts`) + optional separate `ml/` module |
| Cloud Coupling | Hardcoded AWS ECS `.aws/task-definition.json`, ECR account ID, CloudWatch logs, and `boto3` S3 store | Cloud-agnostic defaults (`LocalFilesystemDatasetStore` + `InMemorySessionRepository`); optional Redis, S3-compatible (MinIO/R2/S3), GCP Cloud Run, HF Spaces |
| CI/CD & Tests | Only `.github/workflows/deploy.yml` (Docker build & ECS deploy); zero automated tests or lint checks | `.github/workflows/ci.yml` running `ruff`, `pytest` (unit + integration + eval), and container build verification |

## 3. Security Weaknesses Identified

1. **Silent Sandbox Fallback (`agent/helpers.py`):** When `nsjail` returned a non-zero exit code, the runner logged a warning and silently fell back to plain subprocess execution.
2. **Privileged Container Execution (`docker-compose.yml` & `.aws/task-definition.json`):** The entire FastAPI API server ran with `privileged: true` and `CAP_SYS_ADMIN` to host `nsjail`, widening the blast radius of any web-layer compromise.
3. **AST Guard Bypass Vectors (`agent/sandbox_runner.py`):** `SafetyVisitor` blocked `eval` as a `Name` node, but did not block `df.eval(...)` or `df.query(...)` as attribute calls, nor did it block `pickle`-related IPC exposure inside a shared sandbox directory.
4. **Pickle IPC & Cross-Platform Resource Gaps:** `helpers.py` serialized `df` using `pickle.dump` / `pickle.load`. Additionally, `RLIMIT_*` checks skipped silently on Windows (`except ImportError: return`), leaving Windows dev environments without memory enforcement.
5. **Network Kill-Switch Gaps:** `_disable_network` only monkeypatched four `socket` functions rather than replacing the `socket.socket` class and `getaddrinfo`/`gethostbyname`.
6. **API Surface Hardening:** Wildcard CORS (`allow_origins=["*"]`), no upload byte-size or Excel zip-bomb protection, unauthenticated `GET /sessions` enumeration, and no formula-injection sanitization on CSV/Excel previews.
