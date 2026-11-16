"""Multi-node LangGraph EDA workflow with bounded self-repair and fallback parsing."""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from collections.abc import Callable
from typing import Any

import pandas as pd
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from spectrumbloom.agent.deterministic_tools import (
    format_statistical_focus,
    run_aggregate_data,
    run_statistical_test,
)
from spectrumbloom.agent.state import EDAGraphState
from spectrumbloom.agent.tools import (
    AggregateDataInput,
    MLBaselineInput,
    StandardChartInput,
    StatisticalSummaryInput,
    StatisticalTestInput,
    get_agent_tools,
)
from spectrumbloom.domain.exceptions import SandboxTimeoutError, SandboxUnavailableError
from spectrumbloom.domain.interfaces import CodeSandbox
from spectrumbloom.domain.schemas import (
    DatasetProfile,
    MLExperimentRequest,
    StatisticalReport,
    utc_now_iso,
)
from spectrumbloom.eda.charts import build_standard_chart, validate_plotly_figure_payload
from spectrumbloom.eda.profiler import profile_dataframe
from spectrumbloom.llm.prompts import build_eda_system_prompt
from spectrumbloom.ml.experiments import run_ml_experiment

MAX_TOOL_STEPS = 6
MAX_REPAIR_RETRIES = 2

_THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)
_FAKE_IMAGE_MD_RE = re.compile(r"!\[[^\]]*\]\((?:https?://|data:image/)[^\)]+\)\s*", re.IGNORECASE)
_ANY_IMAGE_MD_RE = re.compile(r"!\[[^\]]*\]\([^\)]+\)\s*")
_TOOL_CALL_TAG_RE = re.compile(r"<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>", re.IGNORECASE)
_JSON_FENCE_RE = re.compile(r"```json\s*\n(\{[\s\S]*?\})\s*\n```", re.IGNORECASE)
_PY_FENCE_RE = re.compile(r"```(?:python|py)\s*\n([\s\S]*?)\n```", re.IGNORECASE)
_REPEAT_QUERY_RE = re.compile(
    r"\b(?:repeat|restate|echo|say that again|previous figure)\b",
    re.IGNORECASE,
)
_PROFILE_QUERY_RE = re.compile(
    r"\b(?:profile|shape|missing|nulls?|nan|duplicates?|overview|row count|column count)\b",
    re.IGNORECASE,
)
_DATASET_CALC_RE = re.compile(
    r"\b(?:"
    r"mean|average|avg|median|std|standard deviation|variance|sum|total|count|rows?|columns?|"
    r"min(?:imum)?|max(?:imum)?|quantile|percentile|missing|nulls?|nan|duplicates?|"
    r"correlat\w*|p[- ]?value|t[- ]?test|anova|mann[- ]?whitney|kruskal|"
    r"chi[- ]?square|pearson|spearman|"
    r"profile|summar(?:y|ize)|distribut\w*|histogram|scatter|bar chart|box ?plot|"
    r"heatmap|line chart|"
    r"chart|plot|visualiz\w*|group|filter|subset|compar\w*|highest|lowest|largest|smallest|"
    r"top|bottom|ratio|difference|percent(?:age)?|"
    r"still filtering|instead of|follow[- ]?up|that group|that region|those records|that subset"
    r")\b",
    re.IGNORECASE,
)
_INVALID_COL_ANSWER_RE = re.compile(
    r"\b(?:does not exist|not exist|not found|not in the dataset|"
    r"no such column|invalid column|unknown column)\b",
    re.IGNORECASE,
)
_NUM_TOKEN_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def normalize_text(content: Any) -> str:
    """Extract clean plain text from LangChain message content and strip raw <think> blocks."""
    if isinstance(content, str):
        raw = content
    elif isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
        raw = "\n".join(parts)
    elif content is None:
        raw = ""
    else:
        raw = str(content)

    cleaned = _THINK_RE.sub("", raw)
    cleaned = _FAKE_IMAGE_MD_RE.sub("", cleaned).strip()
    return cleaned


def extract_tool_calls_with_fallback(message: BaseMessage) -> list[dict[str, Any]]:
    """Extract native tool calls, or fall back to XML <tool_call> or fenced Python blocks."""
    native = getattr(message, "tool_calls", None) or []
    if native:
        normalized: list[dict[str, Any]] = []
        for tc in native:
            if isinstance(tc, dict):
                normalized.append(
                    {
                        "id": str(tc.get("id") or uuid.uuid4()),
                        "name": str(tc.get("name") or ""),
                        "args": tc.get("args") or tc.get("arguments") or {},
                    }
                )
            else:
                normalized.append(
                    {
                        "id": str(getattr(tc, "id", None) or uuid.uuid4()),
                        "name": str(getattr(tc, "name", "")),
                        "args": getattr(tc, "args", None) or {},
                    }
                )
        return normalized

    text = normalize_text(getattr(message, "content", ""))
    if not text:
        return []

    stripped = text.strip()
    tag_matches = _TOOL_CALL_TAG_RE.findall(text) + _JSON_FENCE_RE.findall(text)
    if not tag_matches and stripped.startswith("{") and stripped.endswith("}"):
        tag_matches = [stripped]
    if tag_matches:
        parsed_calls: list[dict[str, Any]] = []
        for raw_json in tag_matches:
            try:
                obj = json.loads(raw_json)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and obj.get("name"):
                args = obj.get("arguments") or obj.get("args") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"code": args}
                parsed_calls.append(
                    {
                        "id": str(uuid.uuid4()),
                        "name": str(obj["name"]),
                        "args": args if isinstance(args, dict) else {},
                    }
                )
        if parsed_calls:
            return parsed_calls

    py_blocks = _PY_FENCE_RE.findall(text)
    if py_blocks:
        return [
            {
                "id": str(uuid.uuid4()),
                "name": "python_repl",
                "args": {"code": py_blocks[0].strip(), "thoughts": "fallback_code_block"},
            }
        ]

    return []


def _extract_numeric_tokens(text: str) -> list[str]:
    return [tok.replace(",", "").lstrip("+") for tok in _NUM_TOKEN_RE.findall(text or "")]


def _requires_tool_execution(
    query: str,
    answer_text: str,
    prior_ai_texts: list[str],
) -> bool:
    """Return True if a data/profile answer bypassed tool execution."""
    if _PROFILE_QUERY_RE.search(query):
        return True

    answer_nums = _extract_numeric_tokens(answer_text)
    if _REPEAT_QUERY_RE.search(query):
        prior_nums = set(_extract_numeric_tokens("\n".join(prior_ai_texts)))
        if not answer_nums or all(n in prior_nums for n in answer_nums):
            return False

    if _INVALID_COL_ANSWER_RE.search(answer_text) and not answer_nums:
        return False

    if answer_nums:
        return True

    return bool(_DATASET_CALC_RE.search(query))


def _format_boundary_regex(value: Any) -> str | None:
    if isinstance(value, bool):
        return None
    try:
        num = float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None
    if num.is_integer():
        return rf"{int(num)}(?:\.0+)?"
    return re.escape(f"{num:g}")


def _check_filter_operator_fidelity(query: str, args: dict[str, Any]) -> str | None:
    """Catch dropped equality (`>` vs `>=`, `<` vs `<=`) on explicit numeric boundary filters."""
    if not query:
        return None

    for col_k, op_k, val_k in (
        ("filter_column", "filter_op", "filter_value"),
        ("filter2_column", "filter2_op", "filter2_value"),
    ):
        col = args.get(col_k)
        op = str(args.get(op_k) or "").strip()
        val = args.get(val_k)
        if not col or op not in {">", "<"}:
            continue
        num_pat = _format_boundary_regex(val)
        if not num_pat:
            continue

        col_words = re.escape(str(col)).replace(r"_", r"[\s_]+")
        if op == ">":
            strict_pat = rf"\b{col_words}\b[^\n,;.]{{0,24}}?(?<![<>!=])>(?!=)\s*{num_pat}\b"
            incl_op = r"(?:>=|≥|at least|no less than|on or after)"
            incl_pat = rf"\b{col_words}\b[^\n,;.]{{0,24}}?{incl_op}\s*{num_pat}\b"
            expected_op = ">="
        else:
            strict_pat = rf"\b{col_words}\b[^\n,;.]{{0,24}}?(?<![<>!=])<(?!=)\s*{num_pat}\b"
            incl_op = r"(?:<=|≤|at most|no more than|on or before)"
            incl_pat = rf"\b{col_words}\b[^\n,;.]{{0,24}}?{incl_op}\s*{num_pat}\b"
            expected_op = "<="

        if re.search(incl_pat, query, re.IGNORECASE) and not re.search(
            strict_pat, query, re.IGNORECASE
        ):
            return (
                f"Filter operator mismatch in {op_k}: the query specifies an inclusive boundary "
                f"('{expected_op}' for '{col}' at {val}), not strict '{op}'. "
                f"Re-run aggregate_data with {op_k}='{expected_op}'."
            )
    return None


def _check_spurious_filter(
    messages: list[BaseMessage],
    active_query: str,
    args: dict[str, Any],
) -> str | None:
    """Reject hallucinated row filters (from example profile values) never requested in chat."""
    conv_parts: list[str] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            conv_parts.append(normalize_text(msg.content))
        elif isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            conv_parts.append(normalize_text(msg.content))
    conv_low = re.sub(r"[^a-z0-9]+", " ", " ".join(conv_parts).lower())
    active_low = (active_query or "").lower()

    generic_tokens = {
        "mean",
        "total",
        "rate",
        "score",
        "value",
        "count",
        "type",
        "name",
        "date",
        "time",
        "year",
    }

    for col_k, op_k, val_k in (
        ("filter_column", "filter_op", "filter_value"),
        ("filter2_column", "filter2_op", "filter2_value"),
    ):
        col = args.get(col_k)
        op = str(args.get(op_k) or "").strip().lower()
        val = args.get(val_k)
        if not col:
            continue

        col_norm = re.sub(r"[^a-z0-9]+", " ", str(col).lower()).strip()
        col_mentioned = bool(col_norm and col_norm in conv_low)
        if not col_mentioned:
            for tok in col_norm.split():
                if len(tok) >= 4 and tok not in generic_tokens and tok in conv_low:
                    col_mentioned = True
                    break

        vals_to_check = val if isinstance(val, (list, tuple, set)) else [val]
        val_mentioned = False
        for item in vals_to_check:
            if item is None:
                continue
            item_norm = re.sub(r"[^a-z0-9]+", " ", str(item).lower()).strip()
            if len(item_norm) >= 2 and re.search(rf"\b{re.escape(item_norm)}\b", conv_low):
                val_mentioned = True
                break

        null_filter_requested = op in {"isna", "notna"} and bool(
            re.search(
                r"\b(?:null|na|nan|missing|incomplete|non[- ]?null|valid|exclude|drop)\b",
                active_low,
            )
        )

        if not (col_mentioned or val_mentioned or null_filter_requested):
            return (
                f"Spurious filter rejected ({col_k}='{col}', {op_k}='{op}'): neither '{col}' "
                "nor those filter values were requested by the user. Re-run `aggregate_data` on "
                "the full dataset with `filter_column=None, filter_op=None, filter_value=None` "
                "(and `filter2_*=None`)."
            )
    return None


class EDAAgentGraph:
    """Compiled LangGraph agent orchestrating sandboxed code execution, stats, and ML tools."""

    def __init__(
        self,
        *,
        chat_model: Any,
        sandbox: CodeSandbox,
        df_provider: Callable[[], pd.DataFrame],
        dataset_context_provider: Callable[[], str],
        stats_provider: Callable[[], StatisticalReport],
        profile_provider: Callable[[], DatasetProfile] | None = None,
        checkpointer: Any = None,
    ) -> None:
        self.sandbox = sandbox
        self.allow_python_repl = sandbox.backend_name != "disabled"
        self.chat_model = chat_model
        self.tools = get_agent_tools(include_python_repl=self.allow_python_repl)
        self.model_with_tools = chat_model.bind_tools(self.tools)
        self.df_provider = df_provider
        self.dataset_context_provider = dataset_context_provider
        self.stats_provider = stats_provider
        self.profile_provider = profile_provider or (
            lambda: profile_dataframe(self.df_provider(), filename="dataset")
        )
        self.memory = checkpointer or MemorySaver()
        self.compiled = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(EDAGraphState)
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", self._tools_node)
        workflow.add_node("finalize", self._finalize_node)

        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            self._route_after_agent,
            {"tools": "tools", "finalize": "finalize"},
        )
        workflow.add_conditional_edges(
            "tools",
            self._route_after_tools,
            {"agent": "agent", "finalize": "finalize"},
        )
        workflow.add_edge("finalize", END)

        return workflow.compile(checkpointer=self.memory)

    def _compact_history(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        """Keep complete prior Q/A pairs compact while preserving the active turn in full."""
        last_human_idx = -1
        for idx in range(len(messages) - 1, -1, -1):
            if isinstance(messages[idx], HumanMessage):
                last_human_idx = idx
                break

        if last_human_idx <= 0:
            return messages

        prior_qa: list[BaseMessage] = []
        pending_human: HumanMessage | None = None
        for msg in messages[:last_human_idx]:
            if isinstance(msg, HumanMessage):
                pending_human = msg
            elif isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                text = normalize_text(msg.content)
                if pending_human is not None and text:
                    prior_qa.append(pending_human)
                    prior_qa.append(AIMessage(content=text[:500]))
                    pending_human = None

        # Keep at most the last 2 completed Q/A pairs (~4 messages) plus full active turn
        return [
            *[
                HumanMessage(content=normalize_text(m.content))
                if isinstance(m, HumanMessage)
                else AIMessage(content=normalize_text(m.content)[:400])
                for m in prior_qa[-2:]
            ],
            *messages[last_human_idx:],
        ]

    def _agent_node(self, state: EDAGraphState) -> dict[str, Any]:
        messages = list(state.get("messages", []))
        system_prompt = build_eda_system_prompt(
            self.dataset_context_provider(),
            allow_python_repl=self.allow_python_repl,
        )

        # Keep single up-to-date SystemMessage at the front and compact prior tool traces
        filtered = [m for m in messages if not isinstance(m, SystemMessage)]
        compacted = self._compact_history(filtered)
        prompt_messages = [SystemMessage(content=system_prompt), *compacted]

        # Extract active query & prior AI texts for tool-requirement gating
        last_human_idx = -1
        for idx in range(len(filtered) - 1, -1, -1):
            if isinstance(filtered[idx], HumanMessage):
                last_human_idx = idx
                break
        active_query = (
            normalize_text(filtered[last_human_idx].content) if last_human_idx >= 0 else ""
        )
        prior_ai_texts = [
            normalize_text(m.content)
            for m in (filtered[:last_human_idx] if last_human_idx > 0 else [])
            if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None)
        ]

        # If previous tool step failed, inject concise repair guidance
        last_error = state.get("last_error")
        if last_error:
            prompt_messages.append(
                HumanMessage(
                    content=(
                        f"Your previous tool call failed with:\n{last_error}\n"
                        "Prefer deterministic tools (`aggregate_data`, "
                        "`compute_statistical_test`, `create_standard_chart`, "
                        "`get_statistical_summary`) or fix the column name / Python syntax "
                        "and try again."
                    )
                )
            )

        if state.get("step_count", 0) >= MAX_TOOL_STEPS:
            synthesis_prompt = [
                *prompt_messages,
                HumanMessage(
                    content=(
                        "Tool budget reached. Provide your concise final answer directly "
                        "using ONLY the exact tool outputs already computed above."
                    )
                ),
            ]
            response = self.chat_model.invoke(synthesis_prompt)
            return {
                "messages": [AIMessage(content=normalize_text(getattr(response, "content", "")))],
                "last_error": None,
            }

        response = self.model_with_tools.invoke(prompt_messages)

        # Ensure fallback tool calls get attached as structured tool_calls if missing,
        # so LangGraph ToolMessages remain aligned with an AIMessage that declares tool_calls.
        if not getattr(response, "tool_calls", None):
            extracted = extract_tool_calls_with_fallback(response)
            if extracted:
                response = AIMessage(
                    content=normalize_text(getattr(response, "content", "")),
                    tool_calls=extracted,
                )

        # Enforce tool execution when the model tries to answer analytical/follow-up/profile
        # questions directly from prior text or memory without running a tool this turn.
        turn_traces = state.get("tool_traces") or []
        if not getattr(response, "tool_calls", None) and not turn_traces and not last_error:
            draft_text = normalize_text(getattr(response, "content", ""))
            if _requires_tool_execution(active_query, draft_text, prior_ai_texts):
                nudge_messages = [
                    *prompt_messages,
                    AIMessage(content=draft_text or ""),
                    HumanMessage(
                        content=(
                            "You have not called any tool for the current question yet. "
                            "Never answer dataset calculations, filtered counts, or follow-up "
                            "metrics (or statistical tests, charts, missing values, and "
                            "duplicate counts) from memory or prior turns. Call the appropriate "
                            "tool (`aggregate_data`, `compute_statistical_test`, "
                            "`get_statistical_summary`, `create_standard_chart`, or "
                            "`python_repl`) right now."
                        )
                    ),
                ]
                retry_resp = self.model_with_tools.invoke(nudge_messages)
                if not getattr(retry_resp, "tool_calls", None):
                    retry_extracted = extract_tool_calls_with_fallback(retry_resp)
                    if retry_extracted:
                        retry_resp = AIMessage(
                            content=normalize_text(getattr(retry_resp, "content", "")),
                            tool_calls=retry_extracted,
                        )
                response = retry_resp

        # If the user explicitly asked to create/plot a chart/visualization and the model called a
        # non-chart tool (e.g. get_statistical_summary) and is about to finish with 0 figures,
        # nudge once to invoke create_standard_chart.
        turn_figs = state.get("turn_figures") or []
        if (
            not getattr(response, "tool_calls", None)
            and turn_traces
            and not turn_figs
            and not last_error
            and not _INVALID_COL_ANSWER_RE.search(normalize_text(getattr(response, "content", "")))
            and re.search(
                r"\b(?:create|plot|generate|build|draw)\b.*\b(?:chart|plot|histogram|heatmap|"
                r"scatter|box\s*plot|violin|bar\s*chart|line\s*chart)\b",
                active_query,
                re.IGNORECASE,
            )
        ):
            chart_nudge = [
                *prompt_messages,
                AIMessage(content=normalize_text(getattr(response, "content", "")) or ""),
                HumanMessage(
                    content=(
                        "The user requested a Plotly chart, but no chart has been generated yet. "
                        "Call `create_standard_chart` right now with the requested `chart_type` "
                        "(e.g. `heatmap`, `histogram`, `bar`, `box`, `scatter`, `line`)."
                    )
                ),
            ]
            retry_chart = self.model_with_tools.invoke(chart_nudge)
            if not getattr(retry_chart, "tool_calls", None):
                chart_extracted = extract_tool_calls_with_fallback(retry_chart)
                if chart_extracted:
                    retry_chart = AIMessage(
                        content=normalize_text(getattr(retry_chart, "content", "")),
                        tool_calls=chart_extracted,
                    )
            response = retry_chart

        # If the user explicitly asked for a difference ("difference", "minus") or ratio ("ratio")
        # and the draft final answer after tool execution omitted stating the difference or ratio,
        # nudge once so the requested comparison value is explicitly calculated and reported.
        if not getattr(response, "tool_calls", None) and turn_traces and not last_error:
            draft_text = normalize_text(getattr(response, "content", ""))
            q_low = active_query.lower()
            d_low = draft_text.lower()
            asks_diff = bool(re.search(r"\b(?:difference|minus)\b", q_low))
            asks_ratio = bool(re.search(r"\b(?:ratio)\b", q_low))
            has_diff_word = bool(
                re.search(r"\b(?:difference|minus|higher by|lower by|diff)\b", d_low)
            )
            has_ratio_word = bool(re.search(r"\b(?:ratio|times|x\s+higher|x\s+as)\b", d_low))
            if (asks_diff and not has_diff_word) or (asks_ratio and not has_ratio_word):
                missing_kind = "difference" if asks_diff else "ratio"
                calc_nudge = [
                    *prompt_messages,
                    AIMessage(content=draft_text or ""),
                    HumanMessage(
                        content=(
                            f"The user explicitly requested the numeric {missing_kind}. "
                            f"Do not call any more tools. Using the exact values already in the "
                            f"tool outputs above, compute and state both individual values AND "
                            f"their exact numeric {missing_kind} clearly in your final response."
                        )
                    ),
                ]
                retry_calc = self.chat_model.invoke(calc_nudge)
                response = AIMessage(content=normalize_text(getattr(retry_calc, "content", "")))

        return {
            "messages": [response],
            "last_error": None,
        }

    def _route_after_agent(self, state: EDAGraphState) -> str:
        messages = state.get("messages", [])
        if not messages:
            return "finalize"
        last = messages[-1]
        if getattr(last, "tool_calls", None):
            if state.get("step_count", 0) >= MAX_TOOL_STEPS:
                return "finalize"
            return "tools"
        return "finalize"

    def _route_after_tools(self, state: EDAGraphState) -> str:
        if state.get("terminal_error"):
            return "finalize"
        if state.get("retries", 0) > MAX_REPAIR_RETRIES and state.get("last_error"):
            return "finalize"
        return "agent"

    def _execute_tool_call(
        self,
        name: str,
        args: dict[str, Any],
        df: pd.DataFrame,
    ) -> tuple[str, Any, list[dict[str, Any]], str | None, str | None, bool, bool]:
        """Dispatch one tool call cleanly with Pydantic validation and fail-closed isolation."""
        if name == "python_repl":
            code_str = str(args.get("code", ""))
            try:
                exec_res = self.sandbox.execute(code_str, df)
            except (SandboxTimeoutError, SandboxUnavailableError) as exc:
                err = str(exc)
                return "", None, [], err, err, isinstance(exc, SandboxTimeoutError), True

            valid_figs: list[dict[str, Any]] = []
            fig_errors: list[str] = []
            for fig in exec_res.figures:
                ok, reason = validate_plotly_figure_payload(fig)
                if ok:
                    valid_figs.append(fig)
                elif reason:
                    fig_errors.append(reason)
            err = exec_res.error
            if fig_errors and not err:
                err = f"Generated Plotly figure failed validation: {'; '.join(fig_errors)}"
            return exec_res.stdout, exec_res.result, valid_figs, err, None, False, bool(err)

        try:
            if name == "get_statistical_summary":
                validated_stat = StatisticalSummaryInput.model_validate(args)
                res_dict = format_statistical_focus(
                    df,
                    self.profile_provider(),
                    self.stats_provider(),
                    focus=validated_stat.focus,
                    columns=validated_stat.columns,
                )
                return json.dumps(res_dict, default=str), res_dict, [], None, None, False, False

            if name == "aggregate_data":
                validated_agg = AggregateDataInput.model_validate(args)
                res_dict = run_aggregate_data(df, **validated_agg.model_dump())
                return json.dumps(res_dict, default=str), res_dict, [], None, None, False, False

            if name == "compute_statistical_test":
                validated_test = StatisticalTestInput.model_validate(args)
                res_dict = run_statistical_test(df, **validated_test.model_dump())
                return json.dumps(res_dict, default=str), res_dict, [], None, None, False, False

            if name == "create_standard_chart":
                validated_chart = StandardChartInput.model_validate(args)
                fig_payload = build_standard_chart(df, **validated_chart.model_dump())
                ok, reason = validate_plotly_figure_payload(fig_payload)
                if not ok:
                    raise ValueError(reason or "Chart specification failed validation.")
                res_dict: dict[str, Any] = {
                    "status": "chart_created",
                    "title": fig_payload["title"],
                    "chart_kind": fig_payload["chart_kind"],
                    "x": validated_chart.x or None,
                    "y": validated_chart.y,
                    "rows_total": int(fig_payload.get("rows_total", len(df))),
                    "rows_plotted": int(fig_payload.get("rows_plotted", len(df))),
                }
                if "x_min" in fig_payload:
                    res_dict["x_range"] = [fig_payload["x_min"], fig_payload["x_max"]]
                return (
                    json.dumps(res_dict, default=str),
                    res_dict,
                    [fig_payload],
                    None,
                    None,
                    False,
                    False,
                )

            if name == "run_ml_baseline":
                validated_ml = MLBaselineInput.model_validate(args)
                ml_res = run_ml_experiment(
                    df,
                    MLExperimentRequest(
                        target_column=validated_ml.target_column,
                        task_type=validated_ml.task_type,
                    ),
                )
                valid_figs = [
                    fig for fig in ml_res.figures if validate_plotly_figure_payload(fig)[0]
                ]
                res_dict = {
                    "target_column": ml_res.target_column,
                    "task_type": ml_res.task_type,
                    "best_model": ml_res.best_model,
                    "models": [m.model_dump() for m in ml_res.models],
                    "top_features": ml_res.feature_importances[:5],
                }
                return (
                    json.dumps(res_dict, default=str),
                    res_dict,
                    valid_figs,
                    None,
                    None,
                    False,
                    False,
                )
        except (ValidationError, Exception) as exc:
            return "", None, [], str(exc), None, False, True

        return "", None, [], f"Unknown tool '{name}'.", None, False, True

    def _tools_node(self, state: EDAGraphState) -> dict[str, Any]:
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None) or []

        tool_messages: list[ToolMessage] = []
        traces: list[dict[str, Any]] = []
        new_figures: list[dict[str, Any]] = []
        new_sigs: list[str] = []
        seen_sigs = set(state.get("executed_signatures", []))
        retries = int(state.get("retries", 0))
        step_count = int(state.get("step_count", 0))
        last_error: str | None = None
        terminal_error: str | None = state.get("terminal_error")
        timed_out = bool(state.get("timed_out", False))

        df = self.df_provider()
        active_query = ""
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, HumanMessage):
                active_query = normalize_text(msg.content)
                break

        for tc in tool_calls:
            step_count += 1
            name = str(tc.get("name", ""))
            tc_id = str(tc.get("id") or uuid.uuid4())
            args = tc.get("args") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"code": args}
            if not isinstance(args, dict):
                args = {}

            sig_payload = json.dumps({"tool": name, "args": args}, sort_keys=True, default=str)
            sig = hashlib.sha256(sig_payload.encode("utf-8")).hexdigest()[:24]
            ok_sig = f"ok:{sig}"
            err_sig = f"err:{sig}"

            if ok_sig in seen_sigs:
                tool_messages.append(
                    ToolMessage(
                        content=(
                            f"Duplicate tool call '{name}' with identical arguments was skipped "
                            "because it already succeeded in this turn. Summarize the prior result "
                            "directly without calling more tools."
                        ),
                        tool_call_id=tc_id,
                        name=name,
                    )
                )
                continue

            started = time.perf_counter()
            code_str = str(args.get("code", "")) if name == "python_repl" else ""
            if err_sig in seen_sigs:
                error_msg = (
                    f"Duplicate failing tool call '{name}' with identical arguments suppressed. "
                    "Change the tool, column names, or Python code."
                )
                stdout, result_val, figs = "", None, []
                retries += 1
                last_error = error_msg
            else:
                op_fidelity_err = (
                    _check_filter_operator_fidelity(active_query, args)
                    or _check_spurious_filter(state.get("messages", []), active_query, args)
                    if name == "aggregate_data"
                    else None
                )
                if op_fidelity_err:
                    stdout, result_val, figs = "", None, []
                    error_msg = op_fidelity_err
                    retries += 1
                    last_error = error_msg
                    seen_sigs.add(err_sig)
                    new_sigs.append(err_sig)
                else:
                    (
                        stdout,
                        result_val,
                        figs,
                        error_msg,
                        term_err,
                        call_timed_out,
                        count_retry,
                    ) = self._execute_tool_call(name, args, df)
                    if term_err:
                        terminal_error = term_err
                    if call_timed_out:
                        timed_out = True
                    if error_msg:
                        if count_retry:
                            retries += 1
                        last_error = error_msg
                        seen_sigs.add(err_sig)
                        new_sigs.append(err_sig)
                    else:
                        last_error = None
                        seen_sigs.add(ok_sig)
                        new_sigs.append(ok_sig)

            duration_ms = round((time.perf_counter() - started) * 1000.0, 2)
            new_figures.extend(figs)
            traces.append(
                {
                    "tool": name,
                    "code": code_str,
                    "stdout": stdout,
                    "result": result_val,
                    "error": error_msg,
                    "figure_count": len(figs),
                    "duration_ms": duration_ms,
                    "timestamp": utc_now_iso(),
                }
            )

            parts: list[str] = []
            if stdout:
                parts.append(f"STDOUT:\n{stdout}")
            if result_val is not None:
                parts.append(f"RESULT:\n{json.dumps(result_val, default=str)}")
            if figs:
                labels = ", ".join(f.get("title", "Figure") for f in figs)
                parts.append(f"FIGURES_GENERATED: {labels}")
            elif name == "python_repl":
                parts.append(
                    "FIGURES_GENERATED: none (only claim a chart was generated if "
                    "FIGURES_GENERATED lists it)"
                )
            if error_msg:
                parts.append(f"ERROR:\n{error_msg}")

            content = "\n\n".join(parts) if parts else "Execution completed with no output."
            if len(content) > 2200:
                content = f"{content[:2150]}\n...[truncated for context budget]"
            tool_messages.append(ToolMessage(content=content, tool_call_id=tc_id, name=name))

        return {
            "messages": tool_messages,
            "tool_traces": traces,
            "turn_figures": new_figures,
            "executed_signatures": new_sigs,
            "retries": retries,
            "step_count": step_count,
            "last_error": last_error,
            "terminal_error": terminal_error,
            "timed_out": timed_out,
        }

    def _finalize_node(self, state: EDAGraphState) -> dict[str, Any]:
        messages = state.get("messages", [])
        last_human_idx = -1
        for idx in range(len(messages) - 1, -1, -1):
            if isinstance(messages[idx], HumanMessage):
                last_human_idx = idx
                break
        turn_messages = messages[last_human_idx + 1 :] if last_human_idx >= 0 else messages

        answer = ""
        for msg in reversed(turn_messages):
            if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                candidate = normalize_text(msg.content)
                if candidate:
                    answer = candidate
                    break

        traces = state.get("tool_traces", [])
        figs = state.get("turn_figures", [])
        terminal_error = state.get("terminal_error")
        timed_out = bool(state.get("timed_out", False))
        status = "success"

        if terminal_error:
            status = "failed"
            prefix = "Execution timed out" if timed_out else "Execution failed"
            answer = f"{prefix}: {terminal_error}"
        elif traces and all(t.get("error") for t in traces) and not figs:
            status = "failed"
            terminal_error = str(traces[-1].get("error") or "All tool executions failed.")
            answer = (
                "The analysis could not be completed because all tool executions failed. "
                f"Last error: {terminal_error}"
            )
        elif not answer:
            if figs:
                titles = ", ".join(f.get("title", "Chart") for f in figs)
                answer = f"Generated {len(figs)} interactive visualization(s): {titles}."
            elif traces:
                ok_traces = [t for t in traces if not t.get("error")]
                last_out = (
                    (ok_traces[-1].get("stdout") or ok_traces[-1].get("result"))
                    if ok_traces
                    else None
                )
                answer = (
                    f"Analysis result:\n\n```\n{last_out}\n```"
                    if last_out
                    else "Analysis complete."
                )
            else:
                status = "failed"
                terminal_error = "No response was generated by the model."
                answer = terminal_error

        if not figs and answer:
            answer = _ANY_IMAGE_MD_RE.sub("", answer).strip()

        return {
            "final_answer": answer,
            "status": status,
            "terminal_error": terminal_error,
            "timed_out": timed_out,
        }

    def seed_history_if_empty(self, thread_id: str, history_messages: list[BaseMessage]) -> bool:
        """Seed the LangGraph checkpointer with persisted messages if the thread is empty."""
        if not history_messages:
            return False
        config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
        try:
            snap = self.compiled.get_state(config)
            existing = list((snap.values or {}).get("messages") or []) if snap else []
            if existing:
                return False
            self.compiled.update_state(config, {"messages": history_messages})
            return True
        except Exception:
            return False

    def invoke(
        self,
        *,
        query: str,
        thread_id: str,
        callbacks: list[Any] | None = None,
    ) -> dict[str, Any]:
        config: dict[str, Any] = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 30,
        }
        if callbacks:
            config["callbacks"] = callbacks

        initial_state = {
            "messages": [HumanMessage(content=query)],
            "tool_traces": None,
            "turn_figures": None,
            "executed_signatures": None,
            "retries": 0,
            "step_count": 0,
            "last_error": None,
            "terminal_error": None,
            "timed_out": False,
            "status": "success",
            "final_answer": "",
        }
        return self.compiled.invoke(initial_state, config=config)
