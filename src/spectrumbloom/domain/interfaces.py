"""Domain repository and infrastructure service protocols."""

from __future__ import annotations

from typing import Any, Protocol

import pandas as pd

from spectrumbloom.domain.schemas import SandboxExecutionResult, SessionRecord


class SessionRepository(Protocol):
    """Persistence contract for session state and history."""

    def create(self, record: SessionRecord) -> SessionRecord: ...

    def get(self, session_id: str) -> SessionRecord | None: ...

    def save(self, record: SessionRecord) -> SessionRecord: ...

    def list_all(self) -> list[SessionRecord]: ...

    def delete(self, session_id: str) -> bool: ...

    def ping(self) -> bool: ...


class DatasetStore(Protocol):
    """Storage backend contract for uploaded tabular dataset bytes."""

    backend_name: str

    def put_bytes(self, session_id: str, filename: str, content: bytes) -> str: ...

    def get_bytes(self, storage_ref: str) -> bytes: ...

    def delete(self, storage_ref: str) -> None: ...

    def ping(self) -> bool: ...


class CodeSandbox(Protocol):
    """Fail-closed sandbox execution contract for LLM-generated Python."""

    backend_name: str

    def execute(self, code: str, df: pd.DataFrame) -> SandboxExecutionResult: ...


class LLMProvider(Protocol):
    """Provider factory contract returning a LangChain-compatible chat model."""

    provider_name: str

    def build_chat_model(
        self,
        *,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.1,
        num_ctx: int = 4096,
        max_tokens: int = 1536,
    ) -> Any: ...
