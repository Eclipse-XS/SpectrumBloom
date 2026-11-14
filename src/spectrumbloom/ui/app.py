"""SpectrumBloom — Original Agentic Exploratory Data Analysis & ML Studio."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import plotly.io as pio
import streamlit as st

DEFAULT_API_URL = os.getenv("SPECTRUMBLOOM_API_URL", "http://localhost:8000").rstrip("/")
DEFAULT_API_TOKEN = os.getenv("SPECTRUMBLOOM_DEFAULT_API_TOKEN", "")
SAMPLE_CSV_PATH = Path(__file__).resolve().parents[3] / "examples" / "sample_ecommerce.csv"

st.set_page_config(
    page_title="SpectrumBloom — Agentic EDA Studio",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .sb-hero {
        padding: 1.05rem 1.35rem;
        border-radius: 12px;
        background: linear-gradient(120deg, #0f766e 0%, #115e59 55%, #1e293b 100%);
        color: #f8fafc;
        margin-bottom: 0.9rem;
    }
    .sb-hero h1 { margin: 0; font-size: 1.7rem; font-weight: 700; }
    .sb-hero p { margin: 0.25rem 0 0; opacity: 0.92; font-size: 0.94rem; }
    .sb-badge {
        display: inline-block;
        padding: 0.16rem 0.6rem;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 600;
        background: rgba(255,255,255,0.16);
        margin-right: 0.35rem;
        margin-top: 0.35rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _headers() -> dict[str, str]:
    token = st.session_state.get("api_token", "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def _api_request(
    method: str,
    path: str,
    *,
    json_body: dict[str, Any] | None = None,
    files: dict[str, Any] | None = None,
    timeout: float = 130.0,
) -> Any:
    base = st.session_state.get("api_url", DEFAULT_API_URL).rstrip("/")
    with httpx.Client(timeout=timeout) as client:
        resp = client.request(
            method,
            f"{base}{path}",
            json=json_body,
            files=files,
            headers=_headers(),
        )
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        raise RuntimeError(f"API {resp.status_code}: {detail}")
    return resp.json()


def _render_plotly_list(
    figures: list[dict[str, Any]],
    key_prefix: str,
    *,
    allow_html_download: bool = False,
) -> None:
    for idx, fig_payload in enumerate(figures):
        fig_json = fig_payload.get("figure_json")
        if not fig_json:
            continue
        title = str(fig_payload.get("title") or f"Chart {idx + 1}")
        fig_key = f"{key_prefix}_{fig_payload.get('id', idx)}_{idx}"
        try:
            fig = pio.from_json(fig_json)
            st.plotly_chart(fig, use_container_width=True, key=fig_key)
            if allow_html_download:
                html_str = fig.to_html(include_plotlyjs="cdn", full_html=True)
                st.download_button(
                    label=f"Download HTML: {title[:40]}",
                    data=html_str.encode("utf-8"),
                    file_name=f"{fig_key}.html",
                    mime="text/html",
                    key=f"dl_{fig_key}",
                )
        except Exception as exc:
            st.warning(f"Unable to render figure '{title}': {exc}")


def _ensure_session() -> str:
    if not st.session_state.get("session_id"):
        created = _api_request(
            "POST",
            "/api/v1/sessions",
            json_body={
                "provider": st.session_state.get("provider", "ollama"),
                "model": st.session_state.get("model", "qwen3:4b-instruct"),
            },
        )
        st.session_state.session_id = created["session_id"]
    return str(st.session_state.session_id)


def _submit_copilot_query(prompt_text: str) -> None:
    st.session_state.last_failed_query = ""
    st.session_state.last_error_message = ""
    if (
        not st.session_state.chat_messages
        or st.session_state.chat_messages[-1].get("role") != "user"
        or st.session_state.chat_messages[-1].get("content") != prompt_text
    ):
        st.session_state.chat_messages.append({"role": "user", "content": prompt_text})

    try:
        sid = _ensure_session()
        with st.spinner("Running LangGraph tools, validation, and grounded synthesis..."):
            res = _api_request(
                "POST",
                f"/api/v1/sessions/{sid}/query",
                json_body={"query": prompt_text},
            )
        st.session_state.chat_messages.append(
            {
                "role": "assistant",
                "content": res["answer"],
                "status": res.get("status", "success"),
                "error": res.get("error"),
                "timed_out": bool(res.get("timed_out", False)),
                "latency_ms": res.get("latency_ms", 0.0),
                "retries_used": res.get("retries_used", 0),
                "figures": res.get("figures", []),
                "tool_traces": res.get("tool_traces", []),
                "evaluation": res.get("evaluation"),
            }
        )
        if res.get("status") == "failed":
            st.session_state.last_failed_query = prompt_text
            st.session_state.last_error_message = res.get("error") or res["answer"]
    except Exception as exc:
        st.session_state.last_failed_query = prompt_text
        st.session_state.last_error_message = str(exc)


# -- Initialize UI session defaults --------------------------------------------
for key, default_val in (
    ("api_url", DEFAULT_API_URL),
    ("api_token", DEFAULT_API_TOKEN),
    ("provider", os.getenv("LLM_PROVIDER", "ollama")),
    ("model", os.getenv("LLM_MODEL", "qwen3:4b-instruct")),
    ("session_id", ""),
    ("upload_signature", ""),
    ("eda_report", None),
    ("preview_data", None),
    ("chat_messages", []),
    ("ml_result", None),
    ("last_failed_query", ""),
    ("last_error_message", ""),
):
    if key not in st.session_state:
        st.session_state[key] = default_val

health_data: dict[str, Any] | None = None
try:
    health_data = _api_request("GET", "/health", timeout=3.0)
except Exception as exc:
    health_data = {"status": "unreachable", "error": str(exc)}

sandbox_mode = str(health_data.get("sandbox_backend", "unknown"))
dynamic_python_enabled = bool(health_data.get("sandbox_dynamic_python_enabled", False))

st.markdown(
    f"""
    <div class="sb-hero">
      <h1>🌿 SpectrumBloom</h1>
      <p>Upload → Overview → Automated EDA → Ask AI → Inspect Traces &amp; Charts → Export</p>
      <div>
        <span class="sb-badge">LangGraph Agent</span>
        <span class="sb-badge">Quantized Qwen / Ollama</span>
        <span class="sb-badge">Sandbox: {sandbox_mode}</span>
        <span class="sb-badge">OpenTelemetry + Deterministic Eval</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -- Sidebar -------------------------------------------------------------------
with st.sidebar:
    st.subheader("1. Backend & Model")
    st.session_state.api_url = st.text_input("FastAPI Endpoint", value=st.session_state.api_url)
    st.session_state.api_token = st.text_input(
        "Bearer Token (required in prod)",
        value=st.session_state.api_token,
        type="password",
    )

    if health_data.get("status") == "ok":
        llm_health = health_data.get("llm") or {}
        st.caption(
            f"API: `ok` | Sandbox: `{sandbox_mode}` | "
            f"LLM ({llm_health.get('provider', 'ollama')}): `{llm_health.get('status', 'n/a')}`"
        )
    else:
        st.error(f"API unreachable: {health_data.get('error', 'unknown error')}")

    provider_opts = ["ollama", "llamacpp", "openai", "gemini"]
    provider = st.selectbox(
        "LLM Provider",
        options=provider_opts,
        index=provider_opts.index(
            st.session_state.provider if st.session_state.provider in provider_opts else "ollama"
        ),
    )
    default_model_map = {
        "ollama": os.getenv("LLM_MODEL", "qwen3:4b-instruct"),
        "llamacpp": "qwen3-4b-instruct-q4_k_m",
        "openai": "gpt-4o-mini",
        "gemini": "gemini-2.5-flash",
    }
    model_name = st.text_input(
        "Model Tag",
        value=st.session_state.model
        if provider == st.session_state.provider
        else default_model_map[provider],
    )
    ext_api_key = ""
    if provider in {"openai", "gemini"}:
        ext_api_key = st.text_input(
            "External Provider API Key (session memory only)", type="password"
        )

    if st.button("Apply Model Settings", use_container_width=True):
        try:
            sid = _ensure_session()
            _api_request(
                "PATCH",
                f"/api/v1/sessions/{sid}/provider",
                json_body={
                    "provider": provider,
                    "model": model_name,
                    "api_key": ext_api_key or None,
                },
            )
            st.session_state.provider = provider
            st.session_state.model = model_name
            st.success(f"Model configured: {provider}:{model_name}")
        except Exception as exc:
            st.error(str(exc))

    st.divider()
    st.subheader("2. Upload Dataset")
    uploaded = st.file_uploader(
        "Upload CSV, TSV, or Excel",
        type=["csv", "tsv", "xlsx", "xls", "xlsm", "ods"],
    )
    load_sample = st.button("Load Sample E-Commerce Dataset", use_container_width=True)

    upload_payload: tuple[str, bytes, str] | None = None
    if load_sample and SAMPLE_CSV_PATH.exists():
        raw = SAMPLE_CSV_PATH.read_bytes()
        upload_payload = (SAMPLE_CSV_PATH.name, raw, f"sample:{len(raw)}")
    elif uploaded is not None:
        raw = uploaded.getvalue()
        sig = f"upload:{uploaded.name}:{len(raw)}"
        if sig != st.session_state.upload_signature:
            upload_payload = (uploaded.name, raw, sig)

    if upload_payload is not None:
        fname, fbytes, usig = upload_payload
        try:
            sid = _ensure_session()
            with st.spinner(f"Profiling `{fname}` and running statistical tests..."):
                eda = _api_request(
                    "POST",
                    f"/api/v1/sessions/{sid}/upload",
                    files={"file": (fname, fbytes, "application/octet-stream")},
                )
                preview = _api_request("GET", f"/api/v1/sessions/{sid}/data/preview?rows=25")
            st.session_state.upload_signature = usig
            st.session_state.eda_report = eda
            st.session_state.preview_data = preview
            st.session_state.chat_messages = []
            st.session_state.ml_result = None
            st.session_state.last_failed_query = ""
            st.session_state.last_error_message = ""
            st.success(
                f"Loaded `{fname}` ({eda['profile']['rows']:,} rows x "
                f"{eda['profile']['columns']} cols)"
            )
        except Exception as exc:
            st.error(f"Dataset upload failed: {exc}")

    if st.session_state.session_id:
        st.divider()
        st.caption(f"Active Session: `{st.session_state.session_id[:12]}...`")
        if st.button("Clear Conversation History", use_container_width=True):
            try:
                _api_request("POST", f"/api/v1/sessions/{st.session_state.session_id}/clear")
                st.session_state.chat_messages = []
                st.session_state.last_failed_query = ""
                st.session_state.last_error_message = ""
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

# -- Main Workspace ------------------------------------------------------------
eda_report = st.session_state.eda_report
preview_data = st.session_state.preview_data

if eda_report is None:
    st.info(
        "Start by uploading a CSV/TSV/Excel dataset in the sidebar or clicking "
        "**Load Sample E-Commerce Dataset**."
    )
    st.stop()

profile = eda_report["profile"]
stats_rep = eda_report["statistics"]
eda_figures = eda_report["figures"]

tab_overview, tab_eda, tab_copilot, tab_ml, tab_export = st.tabs(
    [
        "📋 1. Data Overview",
        "📊 2. Automated EDA",
        "🤖 3. Ask AI & Traces",
        "🧪 4. ML Lab",
        "📥 5. Export Results",
    ]
)

with tab_overview:
    st.subheader(f"Dataset: `{profile['filename']}`")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", f"{profile['rows']:,}")
    c2.metric("Columns", profile["columns"])
    c3.metric("Completeness", f"{profile['completeness_pct']}%")
    c4.metric("Duplicates", profile["duplicate_rows"])
    c5.metric("Memory", f"{profile['memory_mb']} MB")

    if profile["warnings"]:
        for warn in profile["warnings"]:
            st.warning(warn)

    st.subheader("Inferred Column Types & Summary Statistics")
    col_rows = []
    for cp in profile["column_profiles"]:
        ns = cp.get("numeric_stats") or {}
        col_rows.append(
            {
                "Column": cp["name"],
                "Dtype": cp["dtype"],
                "Semantic Role": cp["semantic_type"],
                "Missing (%)": f"{cp['null_count']} ({cp['null_pct']}%)",
                "Unique": cp["unique_count"],
                "Mean": ns.get("mean"),
                "Std": ns.get("std"),
                "Median": ns.get("median"),
                "Min": ns.get("min"),
                "Max": ns.get("max"),
            }
        )
    st.dataframe(pd.DataFrame(col_rows), use_container_width=True, hide_index=True)

    if preview_data:
        st.subheader("Sample Rows (Formula-Injection Safe)")
        st.dataframe(
            pd.DataFrame(preview_data["preview"]), use_container_width=True, hide_index=True
        )

with tab_eda:
    miss = stats_rep["missingness"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Missing Cells", f"{miss['total_missing_cells']:,}")
    m2.metric(
        "Rows with Missing Values",
        f"{miss['rows_with_any_missing']} ({miss['rows_with_any_missing_pct']}%)",
    )
    m3.metric("Multivariate Anomalies (IsoForest)", stats_rep["multivariate_anomaly_count"])

    if miss.get("columns_with_missing"):
        st.subheader("Missing Value Breakdown")
        st.dataframe(
            pd.DataFrame(miss["columns_with_missing"]), use_container_width=True, hide_index=True
        )

    if stats_rep["top_correlations"]:
        st.subheader("Top Numeric Correlations (Pearson & Spearman)")
        st.dataframe(
            pd.DataFrame(stats_rep["top_correlations"]), use_container_width=True, hide_index=True
        )

    if stats_rep["outliers"]:
        st.subheader("Univariate Outliers (IQR, Z-Score, Robust MAD)")
        st.dataframe(pd.DataFrame(stats_rep["outliers"]), use_container_width=True, hide_index=True)

    if stats_rep["normality_checks"]:
        st.subheader("Normality Diagnostics")
        st.dataframe(
            pd.DataFrame(stats_rep["normality_checks"]), use_container_width=True, hide_index=True
        )

    if stats_rep["group_tests"]:
        st.subheader("Significant Group Differences (Kruskal-Wallis)")
        st.dataframe(
            pd.DataFrame(stats_rep["group_tests"]), use_container_width=True, hide_index=True
        )

    st.subheader("Automated Interactive Visualizations")
    _render_plotly_list(eda_figures, key_prefix="auto_eda", allow_html_download=True)

with tab_copilot:
    if not dynamic_python_enabled:
        st.info(
            f"Sandbox mode `{sandbox_mode}`: dynamic arbitrary Python (`python_repl`) is disabled. "
            "Deterministic statistics, filtered aggregations, hypothesis tests, Plotly charts, "
            "and ML baselines remain active."
        )
    else:
        st.caption(
            f"Deterministic analytical tools are prioritized; custom Python runs inside "
            f"`{sandbox_mode}` isolation with AST validation, no network, and resource limits."
        )

    if st.session_state.last_error_message:
        err_col, retry_col = st.columns([4, 1])
        err_col.error(f"Last query failed: {st.session_state.last_error_message}")
        if st.session_state.last_failed_query and retry_col.button(
            "Retry Last Query", use_container_width=True
        ):
            _submit_copilot_query(st.session_state.last_failed_query)
            st.rerun()

    for idx, msg in enumerate(st.session_state.chat_messages):
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                status_label = msg.get("status", "success")
                lat_ms = msg.get("latency_ms", 0.0)
                if status_label == "failed":
                    st.warning(f"Status: failed | Latency: {lat_ms:.0f} ms")
                else:
                    st.caption(
                        f"Status: verified ({status_label}) | Latency: {lat_ms:.0f} ms | "
                        f"Retries: {msg.get('retries_used', 0)}"
                    )

            st.markdown(msg["content"])

            if msg.get("figures"):
                _render_plotly_list(
                    msg["figures"], key_prefix=f"chat_{idx}", allow_html_download=True
                )

            if msg.get("tool_traces") or msg.get("evaluation"):
                with st.expander(
                    "Inspect Calculations, Generated Python & Tool Traces", expanded=False
                ):
                    for t_idx, trace in enumerate(msg.get("tool_traces", []), start=1):
                        badge = "ERROR" if trace.get("error") else "OK"
                        st.markdown(
                            f"**Step {t_idx}: `{trace['tool']}`** — `{badge}` "
                            f"({trace.get('duration_ms', 0)} ms)"
                        )
                        if trace.get("code"):
                            st.code(trace["code"], language="python")
                        if trace.get("stdout"):
                            st.code(trace["stdout"], language="text")
                        if trace.get("result") is not None:
                            result_val = trace["result"]
                            if isinstance(result_val, (dict, list)):
                                st.json(result_val)
                            else:
                                st.code(f"result = {result_val}", language="text")
                        if trace.get("error"):
                            st.error(trace["error"])
                    if msg.get("evaluation"):
                        st.markdown("**Deterministic Verification Checks**")
                        st.json(msg["evaluation"])

    prompt = st.chat_input("Ask SpectrumBloom to filter, aggregate, test hypotheses, or plot...")
    if prompt:
        _submit_copilot_query(prompt)
        st.rerun()

with tab_ml:
    st.subheader("Leak-Safe Baseline ML Experiments")
    col_names = [cp["name"] for cp in profile["column_profiles"]]
    col_left, col_mid, col_right = st.columns([2, 1, 1])
    target_col = col_left.selectbox("Target Column", options=col_names, index=len(col_names) - 1)
    task_override = col_mid.selectbox("Task Type", options=["auto", "classification", "regression"])
    cv_folds = col_right.slider("CV Folds", min_value=2, max_value=10, value=5)

    if st.button("Run Baseline ML Experiment", type="primary"):
        try:
            sid = _ensure_session()
            with st.spinner("Training cross-validated scikit-learn pipelines..."):
                ml_res = _api_request(
                    "POST",
                    f"/api/v1/sessions/{sid}/ml/experiment",
                    json_body={
                        "target_column": target_col,
                        "task_type": task_override,
                        "cv_folds": cv_folds,
                    },
                )
            st.session_state.ml_result = ml_res
        except Exception as exc:
            st.error(f"ML experiment failed: {exc}")

    ml_res = st.session_state.ml_result
    if ml_res:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Detected Task", ml_res["task_type"].title())
        k2.metric("Best Model", ml_res["best_model"])
        k3.metric("Train / Test Rows", f"{ml_res['train_rows']} / {ml_res['test_rows']}")
        k4.metric("Usable Features", ml_res["feature_count"])

        for note in ml_res.get("notes", []):
            st.info(note)

        model_rows = []
        for m in ml_res["models"]:
            metric_label = f"CV {m['cv_primary_metric_name']} (mean ± std)"
            metric_val = f"{m['cv_primary_mean']:.4f} ± {m['cv_primary_std']:.4f}"
            row = {
                "Model": m["model_name"],
                metric_label: metric_val,
                **{f"Test {k}": v for k, v in m["test_metrics"].items()},
            }
            model_rows.append(row)
        st.dataframe(pd.DataFrame(model_rows), use_container_width=True, hide_index=True)

        _render_plotly_list(
            ml_res.get("figures", []), key_prefix="ml_lab", allow_html_download=True
        )

with tab_export:
    st.subheader("Export Analytical Artifacts & Results")
    ex1, ex2, ex3 = st.columns(3)

    with ex1:
        st.markdown("**Automated EDA Report (JSON)**")
        eda_export = json.dumps(
            {
                "profile": profile,
                "statistics": stats_rep,
                "figure_titles": [f.get("title") for f in eda_figures],
            },
            indent=2,
            default=str,
        )
        st.download_button(
            "Download EDA Report (JSON)",
            data=eda_export.encode("utf-8"),
            file_name=f"{Path(profile['filename']).stem}_eda_report.json",
            mime="application/json",
            use_container_width=True,
        )

    with ex2:
        st.markdown("**Dataset Preview (CSV)**")
        prev_df = pd.DataFrame((preview_data or {}).get("preview", []))
        st.download_button(
            "Download Preview Rows (CSV)",
            data=prev_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{Path(profile['filename']).stem}_preview.csv",
            mime="text/csv",
            use_container_width=True,
            disabled=prev_df.empty,
        )

    with ex3:
        st.markdown("**Copilot Chat & Tool Traces (JSON)**")
        trace_export = json.dumps(st.session_state.chat_messages, indent=2, default=str)
        st.download_button(
            "Download Conversation & Traces",
            data=trace_export.encode("utf-8"),
            file_name=f"{Path(profile['filename']).stem}_copilot_traces.json",
            mime="application/json",
            use_container_width=True,
            disabled=len(st.session_state.chat_messages) == 0,
        )

    if st.session_state.ml_result:
        st.divider()
        st.markdown("**ML Baseline Experiment Summary (JSON)**")
        st.download_button(
            "Download ML Experiment Results (JSON)",
            data=json.dumps(st.session_state.ml_result, indent=2, default=str).encode("utf-8"),
            file_name=f"{Path(profile['filename']).stem}_ml_results.json",
            mime="application/json",
        )
