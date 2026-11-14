"""Typed LangGraph state and per-turn accumulators."""

from __future__ import annotations

from typing import Annotated, Any

from langgraph.graph import MessagesState


def _replace_or_append(
    left: list[dict[str, Any]] | None, right: list[dict[str, Any]] | None
) -> list[dict[str, Any]]:
    if right is None:
        return []
    if not left:
        return list(right)
    return [*left, *right]


class EDAGraphState(MessagesState):
    """Conversation + per-turn tool execution state."""

    tool_traces: Annotated[list[dict[str, Any]], _replace_or_append]
    turn_figures: Annotated[list[dict[str, Any]], _replace_or_append]
    retries: int
    step_count: int
    last_error: str | None
    final_answer: str
