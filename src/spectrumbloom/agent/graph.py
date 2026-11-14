"""Multi-node LangGraph EDA workflow with bounded self-repair and fallback parsing."""

from __future__ import annotations

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

from spectrumbloom.agent.state import EDAGraphState
from spectrumbloom.agent.tools import get_agent_tools
from spectrumbloom.domain.interfaces import CodeSandbox
from spectrumbloom.domain.schemas import (
    MLExperimentRequest,
    StatisticalReport,
    utc_now_iso,
)
from spectrumbloom.llm.prompts import build_eda_system_prompt
from spectrumbloom.ml.experiments import run_ml_experiment

MAX_TOOL_STEPS = 6
MAX_REPAIR_RETRIES = 2

_THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)
_FAKE_IMAGE_MD_RE = re.compile(r"!\[[^\]]*\]\((?:https?://|data:image/)[^\)]+\)\s*", re.IGNORECASE)
_TOOL_CALL_TAG_RE = re.compile(r"<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>", re.IGNORECASE)
_JSON_FENCE_RE = re.compile(r"```json\s*\n(\{[\s\S]*?\})\s*\n```", re.IGNORECASE)
_PY_FENCE_RE = re.compile(r"```(?:python|py)\s*\n([\s\S]*?)\n```", re.IGNORECASE)


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
        checkpointer: Any = None,
    ) -> None:
        self.tools = get_agent_tools()
        self.model_with_tools = chat_model.bind_tools(self.tools)
        self.sandbox = sandbox
        self.df_provider = df_provider
        self.dataset_context_provider = dataset_context_provider
        self.stats_provider = stats_provider
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
        """Keep prior Q/A summaries compact while preserving the active turn in full."""
        last_human_idx = -1
        for idx in range(len(messages) - 1, -1, -1):
            if isinstance(messages[idx], HumanMessage):
                last_human_idx = idx
                break

        if last_human_idx <= 0:
            return messages

        prior_qa: list[BaseMessage] = []
        for msg in messages[:last_human_idx]:
            if isinstance(msg, HumanMessage):
                prior_qa.append(msg)
            elif isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                text = normalize_text(msg.content)
                if text:
                    prior_qa.append(AIMessage(content=text[:500]))

        # Keep at most the last 2 completed Q/A turns (~4 messages) plus full active turn
        return [*prior_qa[-4:], *messages[last_human_idx:]]

    def _agent_node(self, state: EDAGraphState) -> dict[str, Any]:
        messages = list(state.get("messages", []))
        system_prompt = build_eda_system_prompt(self.dataset_context_provider())

        # Keep single up-to-date SystemMessage at the front and compact prior tool traces
        filtered = [m for m in messages if not isinstance(m, SystemMessage)]
        compacted = self._compact_history(filtered)
        prompt_messages = [SystemMessage(content=system_prompt), *compacted]

        # If previous tool step failed, inject concise repair guidance
        last_error = state.get("last_error")
        if last_error:
            prompt_messages.append(
                HumanMessage(
                    content=(
                        f"Your previous tool call failed with:\n{last_error}\n"
                        "Fix the Python syntax, column name, or disallowed API call and try again."
                    )
                )
            )

        response = self.model_with_tools.invoke(prompt_messages)

        # Ensure fallback tool calls get attached as structured tool_calls if missing,
        # so LangGraph ToolMessages remain aligned with an AIMessage that declares tool_calls.
        extracted = extract_tool_calls_with_fallback(response)
        if extracted and not getattr(response, "tool_calls", None):
            response = AIMessage(
                content=normalize_text(getattr(response, "content", "")),
                tool_calls=extracted,
            )

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
        if state.get("step_count", 0) >= MAX_TOOL_STEPS:
            return "finalize"
        if state.get("retries", 0) > MAX_REPAIR_RETRIES and state.get("last_error"):
            return "finalize"
        return "agent"

    def _format_stats_focus(self, focus: str) -> str:
        report = self.stats_provider()
        data = report.model_dump()
        key_map = {
            "missingness": "missingness",
            "correlations": "top_correlations",
            "top_correlations": "top_correlations",
            "outliers": "outliers",
            "normality": "normality_checks",
            "normality_checks": "normality_checks",
            "group_tests": "group_tests",
        }
        if focus == "all":
            compact = {
                "missing_cols": [
                    {k: c[k] for k in ("column", "null_count", "null_pct") if k in c}
                    for c in data["missingness"].get("columns_with_missing", [])[:5]
                ],
                "top_correlations": data["top_correlations"][:4],
                "outliers": data["outliers"][:4],
                "multivariate_anomalies": data["multivariate_anomaly_count"],
                "normality_checks": data["normality_checks"][:4],
                "group_tests": data["group_tests"][:3],
            }
            encoded = json.dumps(compact, default=str)
            return encoded[:1200]

        section_key = key_map.get(focus, focus)
        section_val = data.get(section_key, data.get("top_correlations", []))
        if isinstance(section_val, list):
            section_val = section_val[:8]
        encoded = json.dumps(section_val, default=str)
        return encoded[:1200]

    def _tools_node(self, state: EDAGraphState) -> dict[str, Any]:
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None) or []

        tool_messages: list[ToolMessage] = []
        traces: list[dict[str, Any]] = []
        new_figures: list[dict[str, Any]] = []
        retries = int(state.get("retries", 0))
        step_count = int(state.get("step_count", 0))
        last_error: str | None = None

        df = self.df_provider()

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

            started = time.perf_counter()
            stdout = ""
            result_val: Any = None
            error_msg: str | None = None
            figs: list[dict[str, Any]] = []
            code_str = ""

            if name == "python_repl":
                code_str = str(args.get("code", ""))
                exec_res = self.sandbox.execute(code_str, df)
                stdout = exec_res.stdout
                result_val = exec_res.result
                error_msg = exec_res.error
                figs = exec_res.figures
                if error_msg:
                    retries += 1
                    last_error = error_msg
                else:
                    last_error = None
            elif name == "get_statistical_summary":
                focus = str(args.get("focus", "all"))
                try:
                    stdout = self._format_stats_focus(focus)
                    result_val = json.loads(stdout)
                except Exception as exc:
                    error_msg = str(exc)
                    last_error = error_msg
            elif name == "run_ml_baseline":
                target = str(args.get("target_column", ""))
                task_type = str(args.get("task_type", "auto"))
                if task_type not in {"auto", "classification", "regression"}:
                    task_type = "auto"
                try:
                    ml_res = run_ml_experiment(
                        df,
                        MLExperimentRequest(
                            target_column=target,
                            task_type=task_type,  # type: ignore[arg-type]
                        ),
                    )
                    figs = ml_res.figures
                    result_val = {
                        "target_column": ml_res.target_column,
                        "task_type": ml_res.task_type,
                        "best_model": ml_res.best_model,
                        "models": [m.model_dump() for m in ml_res.models],
                        "top_features": ml_res.feature_importances[:5],
                    }
                    stdout = json.dumps(result_val)
                except Exception as exc:
                    error_msg = str(exc)
                    retries += 1
                    last_error = error_msg
            else:
                error_msg = f"Unknown tool '{name}'."
                last_error = error_msg

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
            if error_msg:
                parts.append(f"ERROR:\n{error_msg}")

            content = "\n\n".join(parts) if parts else "Execution completed with no output."
            if len(content) > 1800:
                content = f"{content[:1750]}\n...[truncated for context budget]"
            tool_messages.append(ToolMessage(content=content, tool_call_id=tc_id, name=name))

        return {
            "messages": tool_messages,
            "tool_traces": traces,
            "turn_figures": new_figures,
            "retries": retries,
            "step_count": step_count,
            "last_error": last_error,
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

        if not answer:
            traces = state.get("tool_traces", [])
            figs = state.get("turn_figures", [])
            if traces and traces[-1].get("error") and not figs:
                answer = (
                    "The analysis encountered an error during sandboxed execution and reached "
                    f"the retry limit: {traces[-1]['error']}"
                )
            elif figs:
                titles = ", ".join(f.get("title", "Chart") for f in figs)
                answer = f"Generated {len(figs)} interactive visualization(s): {titles}."
            elif traces:
                last_out = traces[-1].get("stdout") or traces[-1].get("result")
                answer = (
                    f"Analysis result:\n\n```\n{last_out}\n```"
                    if last_out
                    else "Analysis complete."
                )
            else:
                answer = "No response was generated by the model."

        return {"final_answer": answer}

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
            "retries": 0,
            "step_count": 0,
            "last_error": None,
            "final_answer": "",
        }
        return self.compiled.invoke(initial_state, config=config)
