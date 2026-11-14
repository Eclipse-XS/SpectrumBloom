"""Typed LangGraph state and per-turn accumulators."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from langgraph.graph import MessagesState


def _replace_or_append(left: list[Any] | None, right: list[Any] | None) -> list[Any]:
    if right is None:
        return []
    if not left:
        return list(right)
    return [*left, *right]


class EDAGraphState(MessagesState):
    """Conversation + per-turn tool execution state."""

    tool_traces: Annotated[list[dict[str, Any]], _replace_or_append]
    turn_figures: Annotated[list[dict[str, Any]], _replace_or_append]
    executed_signatures: Annotated[list[str], _replace_or_append]
    retries: int
    step_count: int
    last_error: str | None
    terminal_error: str | None
    timed_out: bool
    status: Literal["success", "failed"]
    final_answer: str
