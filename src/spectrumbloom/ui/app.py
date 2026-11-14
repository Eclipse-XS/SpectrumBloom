"""SpectrumBloom — Original Agentic EDA & ML Studio UI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import plotly.io as pio
import streamlit as st

DEFAULT_API_URL = os.getenv("SPECTRUMBLOOM_API_URL", "http://localhost:8000").rstrip("/")
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
        padding: 1.0rem 1.25rem;
        border-radius: 12px;
        background: linear-gradient(120deg, #0f766e 0%, #115e59 55%, #1e293b 100%);
        color: #f8fafc;
        margin-bottom: 1rem;
    }
    .sb-hero h1 { margin: 0; font-size: 1.75rem; font-weight: 700; }
    .sb-hero p { margin: 0.25rem 0 0; opacity: 0.9; font-size: 0.95rem; }
    .sb-badge {
        display: inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        background: rgba(255,255,255,0.16);
        margin-right: 0.35rem;
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
    timeout: float = 120.0,
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


def _render_plotly_list(figures: list[dict[str, Any]], key_prefix: str) -> None:
    for idx, fig_payload in enumerate(figures):
        fig_json = fig_payload.get("figure_json")
        if not fig_json:
            continue
        try:
            fig = pio.from_json(fig_json)
            st.plotly_chart(
                fig,
                use_container_width=True,
                key=f"{key_prefix}_{fig_payload.get('id', idx)}_{idx}",
            )
        except Exception as exc:
            st.warning(f"Unable to render figure '{fig_payload.get('title', idx)}': {exc}")


def _ensure_session() -> str:
    if "session_id" not in st.session_state or not st.session_state.session_id:
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


# -- Initialize UI session defaults --------------------------------------------
for key, default_val in (
    ("api_url", DEFAULT_API_URL),
    ("api_token", ""),
    ("provider", os.getenv("LLM_PROVIDER", "ollama")),
    ("model", os.getenv("LLM_MODEL", "qwen3:4b-instruct")),
    ("session_id", ""),
    ("eda_report", None),
    ("preview_data", None),
    ("chat_messages", []),
    ("ml_result", None),
):
    if key not in st.session_state:
        st.session_state[key] = default_val

st.markdown(
    """
    <div class="sb-hero">
      <h1>🌿 SpectrumBloom</h1>
      <p>Local-first Agentic Exploratory Data Analysis &amp; Sandboxed ML Studio</p>
      <div style="margin-top:0.5rem;">
        <span class="sb-badge">LangGraph</span>
        <span class="sb-badge">Quantized Qwen / Ollama</span>
        <span class="sb-badge">Fail-Closed Sandbox</span>
        <span class="sb-badge">OpenTelemetry + DeepEval</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -- Sidebar -------------------------------------------------------------------
with st.sidebar:
    st.subheader("🔌 Backend & Inference")
    st.session_state.api_url = st.text_input("FastAPI Endpoint", value=st.session_state.api_url)
    st.session_state.api_token = st.text_input(
        "Bearer Token (if enabled)", value=st.session_state.api_token, type="password"
    )

    try:
        health = _api_request("GET", "/health", timeout=3.0)
        st.caption(
            f"API: `{health['status']}` | Sandbox: `{health['sandbox_backend']}` | "
            f"LLM ({health['llm'].get('provider')}): `{health['llm'].get('status')}`"
        )
    except Exception as exc:
        st.error(f"Backend unreachable: {exc}")

    provider = st.selectbox(
        "LLM Provider",
        options=["ollama", "llamacpp", "openai", "gemini"],
        index=["ollama", "llamacpp", "openai", "gemini"].index(
            st.session_state.provider
            if st.session_state.provider in {"ollama", "llamacpp", "openai", "gemini"}
            else "ollama"
        ),
    )
    default_model_map = {
        "ollama": "qwen3:4b-instruct",
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
            st.success(f"Updated to {provider}:{model_name}")
        except Exception as exc:
            st.error(str(exc))

    st.divider()
    st.subheader("📂 Dataset Upload")
    uploaded = st.file_uploader(
        "Upload CSV, TSV, or Excel",
        type=["csv", "tsv", "xlsx", "xls", "xlsm", "ods"],
    )
    load_sample = st.button("Load Sample E-Commerce Dataset", use_container_width=True)

    upload_payload: tuple[str, bytes] | None = None
    if load_sample and SAMPLE_CSV_PATH.exists():
        upload_payload = (SAMPLE_CSV_PATH.name, SAMPLE_CSV_PATH.read_bytes())
    elif uploaded is not None:
        upload_payload = (uploaded.name, uploaded.getvalue())

    if upload_payload is not None and (
        st.session_state.eda_report is None
        or st.session_state.eda_report["profile"]["filename"] != upload_payload[0]
        or load_sample
    ):
        try:
            sid = _ensure_session()
            with st.spinner("Profiling dataset and computing statistical diagnostics..."):
                eda = _api_request(
                    "POST",
                    f"/api/v1/sessions/{sid}/upload",
                    files={
                        "file": (upload_payload[0], upload_payload[1], "application/octet-stream")
                    },
                )
                preview = _api_request("GET", f"/api/v1/sessions/{sid}/data/preview?rows=12")
            st.session_state.eda_report = eda
            st.session_state.preview_data = preview
            st.session_state.chat_messages = []
            st.session_state.ml_result = None
            st.success(f"Loaded `{upload_payload[0]}` ({eda['profile']['rows']} rows)")
        except Exception as exc:
            st.error(f"Upload failed: {exc}")

    if st.session_state.session_id:
        st.divider()
        st.caption(f"Session ID: `{st.session_state.session_id[:12]}...`")
        if st.button("Clear Conversation", use_container_width=True):
            try:
                _api_request("POST", f"/api/v1/sessions/{st.session_state.session_id}/clear")
                st.session_state.chat_messages = []
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

# -- Main Workspace ------------------------------------------------------------
eda_report = st.session_state.eda_report
preview_data = st.session_state.preview_data

if eda_report is None:
    st.info(
        "Upload a CSV/Excel dataset from the sidebar or click **Load Sample E-Commerce Dataset** "
        "to explore profiling, statistical diagnostics, agentic Python EDA, and ML baselines."
    )
    st.stop()

profile = eda_report["profile"]
stats_rep = eda_report["statistics"]
eda_figures = eda_report["figures"]

tab_profile, tab_stats, tab_copilot, tab_ml = st.tabs(
    ["📋 Profile & Preview", "📊 Statistical EDA & Charts", "🤖 Agentic Copilot", "🧪 ML Lab"]
)

with tab_profile:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", f"{profile['rows']:,}")
    c2.metric("Columns", profile["columns"])
    c3.metric("Completeness", f"{profile['completeness_pct']}%")
    c4.metric("Duplicates", profile["duplicate_rows"])
    c5.metric("Memory", f"{profile['memory_mb']} MB")

    if profile["warnings"]:
        for warn in profile["warnings"]:
            st.warning(warn)

    st.subheader("Column Schema & Semantic Profiling")
    col_rows = []
    for cp in profile["column_profiles"]:
        ns = cp.get("numeric_stats") or {}
        col_rows.append(
            {
                "Column": cp["name"],
                "Dtype": cp["dtype"],
                "Semantic Role": cp["semantic_type"],
                "Nulls (%)": f"{cp['null_count']} ({cp['null_pct']}%)",
                "Unique": cp["unique_count"],
                "Mean": ns.get("mean"),
                "Median": ns.get("median"),
                "Skew": ns.get("skew"),
            }
        )
    st.dataframe(pd.DataFrame(col_rows), use_container_width=True, hide_index=True)

    if preview_data:
        st.subheader("Data Preview (Formula-Safe)")
        st.dataframe(
            pd.DataFrame(preview_data["preview"]), use_container_width=True, hide_index=True
        )

with tab_stats:
    miss = stats_rep["missingness"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Missing Cells", f"{miss['total_missing_cells']:,}")
    m2.metric(
        "Rows with Missing",
        f"{miss['rows_with_any_missing']} ({miss['rows_with_any_missing_pct']}%)",
    )
    m3.metric("Multivariate Anomalies (IsoForest)", stats_rep["multivariate_anomaly_count"])

    if stats_rep["top_correlations"]:
        st.subheader("Strongest Numeric Correlations")
        st.dataframe(
            pd.DataFrame(stats_rep["top_correlations"]), use_container_width=True, hide_index=True
        )

    if stats_rep["outliers"]:
        st.subheader("Outlier Summary (IQR, Z-Score, Robust MAD)")
        st.dataframe(pd.DataFrame(stats_rep["outliers"]), use_container_width=True, hide_index=True)

    if stats_rep["normality_checks"]:
        st.subheader("Normality & Distribution Diagnostics")
        st.dataframe(
            pd.DataFrame(stats_rep["normality_checks"]), use_container_width=True, hide_index=True
        )

    if stats_rep["group_tests"]:
        st.subheader("Significant Group Differences (Kruskal-Wallis p < 0.05)")
        st.dataframe(
            pd.DataFrame(stats_rep["group_tests"]), use_container_width=True, hide_index=True
        )

    st.subheader("Automated Interactive Visualizations")
    _render_plotly_list(eda_figures, key_prefix="auto_eda")

with tab_copilot:
    st.caption(
        "Ask natural-language questions. Generated Python runs inside SpectrumBloom's "
        "fail-closed sandbox with AST checks, no network/fs writes, and correctness checks."
    )

    for idx, msg in enumerate(st.session_state.chat_messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("figures"):
                _render_plotly_list(msg["figures"], key_prefix=f"chat_{idx}")
            if msg.get("tool_traces"):
                with st.expander("Sandbox Execution Traces & Evaluation", expanded=False):
                    if msg.get("evaluation"):
                        st.json(msg["evaluation"])
                    for t_idx, trace in enumerate(msg["tool_traces"], start=1):
                        st.markdown(
                            f"**Step {t_idx}: `{trace['tool']}`** ({trace['duration_ms']} ms)"
                        )
                        if trace.get("code"):
                            st.code(trace["code"], language="python")
                        if trace.get("stdout"):
                            st.text(trace["stdout"])
                        if trace.get("error"):
                            st.error(trace["error"])

    prompt = st.chat_input("Ask SpectrumBloom to analyze, filter, test, or plot your data...")
    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        try:
            sid = _ensure_session()
            with st.spinner("Running LangGraph agent & sandboxed Python..."):
                res = _api_request(
                    "POST",
                    f"/api/v1/sessions/{sid}/query",
                    json_body={"query": prompt},
                )
            st.session_state.chat_messages.append(
                {
                    "role": "assistant",
                    "content": res["answer"],
                    "figures": res.get("figures", []),
                    "tool_traces": res.get("tool_traces", []),
                    "evaluation": res.get("evaluation"),
                }
            )
            st.rerun()
        except Exception as exc:
            st.error(f"Query failed: {exc}")

with tab_ml:
    st.subheader("Leak-Safe Baseline ML Experiments (Optional Module)")
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

        _render_plotly_list(ml_res.get("figures", []), key_prefix="ml_lab")
