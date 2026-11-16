"""Application service orchestrating sessions, EDA profiling, ML experiments, and agent turns."""

from __future__ import annotations

import contextlib
import hashlib
import threading
import time
import uuid
from typing import Any

import httpx
import pandas as pd
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from spectrumbloom.agent.graph import EDAAgentGraph, normalize_text
from spectrumbloom.core.config import Settings
from spectrumbloom.core.security import neutralize_formula_value, sanitize_json_payload
from spectrumbloom.core.telemetry import TelemetryRuntime
from spectrumbloom.domain.exceptions import (
    DatasetNotFoundError,
    LLMInvocationError,
    LLMTimeoutError,
    SessionNotFoundError,
)
from spectrumbloom.domain.interfaces import CodeSandbox, DatasetStore, SessionRepository
from spectrumbloom.domain.schemas import (
    ChatMessageRecord,
    CreateSessionRequest,
    CreateSessionResponse,
    DatasetPreviewResponse,
    DatasetProfile,
    FullEDAReportResponse,
    MLExperimentRequest,
    MLExperimentResponse,
    QueryResponse,
    SessionRecord,
    SessionStatusResponse,
    StatisticalReport,
    ToolTraceItem,
    UpdateProviderRequest,
    UploadedDatasetMeta,
    utc_now_iso,
)
from spectrumbloom.eda.charts import build_eda_figures
from spectrumbloom.eda.loader import parse_tabular_bytes
from spectrumbloom.eda.profiler import format_profile_for_prompt, profile_dataframe
from spectrumbloom.eda.statistics import compute_statistics
from spectrumbloom.llm.providers import UnifiedLLMProvider
from spectrumbloom.ml.experiments import run_ml_experiment
from spectrumbloom.services.eval_service import evaluate_turn_deterministically


class _SessionRuntimeCache:
    """In-process cache of DataFrame, EDA computations, graph, and optional runtime API key."""

    def __init__(self, checkpointer: Any = None) -> None:
        self.lock = threading.RLock()
        self.last_accessed: float = time.monotonic()
        self.dataset_sha256: str | None = None
        self.dataset_storage_ref: str | None = None
        self.df: pd.DataFrame | None = None
        self.profile: DatasetProfile | None = None
        self.statistics: StatisticalReport | None = None
        self.eda_figures: list[dict[str, Any]] | None = None
        self.api_key: str | None = None
        self.graph: EDAAgentGraph | None = None
        self.checkpointer: Any = checkpointer if checkpointer is not None else MemorySaver()

    def touch(self) -> None:
        self.last_accessed = time.monotonic()

    def clear_dataset_resources(self) -> None:
        self.dataset_sha256 = None
        self.dataset_storage_ref = None
        self.df = None
        self.profile = None
        self.statistics = None
        self.eda_figures = None
        self.graph = None

    def release_all(self) -> None:
        self.clear_dataset_resources()
        self.api_key = None
        self.checkpointer = None


class SpectrumBloomService:
    """Core service layer decoupling FastAPI routes from persistence, EDA, ML, and LangGraph."""

    def __init__(
        self,
        *,
        settings: Settings,
        session_repo: SessionRepository,
        dataset_store: DatasetStore,
        sandbox: CodeSandbox,
        llm_provider: UnifiedLLMProvider,
        telemetry: TelemetryRuntime,
        shared_checkpointer: Any = None,
    ) -> None:
        self.settings = settings
        self.session_repo = session_repo
        self.dataset_store = dataset_store
        self.sandbox = sandbox
        self.llm_provider = llm_provider
        self.telemetry = telemetry
        self.shared_checkpointer = shared_checkpointer
        self._cache_lock = threading.RLock()
        self._runtime_cache: dict[str, _SessionRuntimeCache] = {}

    def _evict_expired_or_excess_caches_locked(
        self, *, preserve_session_id: str | None = None
    ) -> None:
        now = time.monotonic()
        ttl = max(60, int(self.settings.session_ttl_seconds))
        max_entries = max(1, int(self.settings.max_cached_sessions))

        to_evict: list[str] = []
        for sid, entry in list(self._runtime_cache.items()):
            if sid == preserve_session_id:
                continue
            if (now - entry.last_accessed) >= ttl:
                to_evict.append(sid)

        for sid in to_evict:
            evicted = self._runtime_cache.pop(sid, None)
            if evicted is not None and evicted.lock.acquire(blocking=False):
                try:
                    evicted.release_all()
                finally:
                    evicted.lock.release()

        if len(self._runtime_cache) > max_entries:
            candidates = sorted(
                (
                    (sid, entry)
                    for sid, entry in self._runtime_cache.items()
                    if sid != preserve_session_id
                ),
                key=lambda item: item[1].last_accessed,
            )
            excess = len(self._runtime_cache) - max_entries
            for sid, entry in candidates:
                if excess <= 0:
                    break
                if entry.lock.acquire(blocking=False):
                    try:
                        self._runtime_cache.pop(sid, None)
                        entry.release_all()
                        excess -= 1
                    finally:
                        entry.lock.release()

    def _get_cache(self, session_id: str) -> _SessionRuntimeCache:
        with self._cache_lock:
            cache = self._runtime_cache.get(session_id)
            if cache is None:
                cache = _SessionRuntimeCache(checkpointer=self.shared_checkpointer)
                self._runtime_cache[session_id] = cache
            else:
                cache.touch()
            self._evict_expired_or_excess_caches_locked(preserve_session_id=session_id)
            return cache

    def _require_session(self, session_id: str) -> SessionRecord:
        record = self.session_repo.get(session_id)
        if record is None:
            with self._cache_lock:
                stale = self._runtime_cache.pop(session_id, None)
                if stale is not None:
                    stale.release_all()
            raise SessionNotFoundError(f"Session '{session_id}' not found.")
        return record

    def _sync_cache_with_record(self, record: SessionRecord, cache: _SessionRuntimeCache) -> None:
        cache.touch()
        rec_sha = record.dataset.sha256 if record.dataset is not None else None
        rec_ref = record.dataset.storage_ref if record.dataset is not None else None
        if cache.dataset_sha256 != rec_sha or cache.dataset_storage_ref != rec_ref:
            cache.clear_dataset_resources()
            cache.dataset_sha256 = rec_sha
            cache.dataset_storage_ref = rec_ref

    def _ensure_df(self, record: SessionRecord, cache: _SessionRuntimeCache) -> pd.DataFrame:
        self._sync_cache_with_record(record, cache)
        if cache.df is not None:
            return cache.df
        if record.dataset is None:
            raise DatasetNotFoundError("No dataset has been uploaded to this session yet.")
        raw_bytes = self.dataset_store.get_bytes(record.dataset.storage_ref)
        df = parse_tabular_bytes(
            raw_bytes,
            record.dataset.filename,
            max_upload_bytes=self.settings.max_upload_bytes,
        )
        cache.df = df
        cache.dataset_sha256 = record.dataset.sha256
        cache.dataset_storage_ref = record.dataset.storage_ref
        return df

    def _ensure_eda(
        self, record: SessionRecord, cache: _SessionRuntimeCache
    ) -> tuple[pd.DataFrame, DatasetProfile, StatisticalReport, list[dict[str, Any]]]:
        df = self._ensure_df(record, cache)
        if cache.profile is None:
            cache.profile = profile_dataframe(
                df, filename=record.dataset.filename if record.dataset else "dataset"
            )
        if cache.statistics is None:
            cache.statistics = compute_statistics(df, cache.profile)
        if cache.eda_figures is None:
            cache.eda_figures = build_eda_figures(df, cache.profile, cache.statistics)
        return df, cache.profile, cache.statistics, cache.eda_figures

    def _restore_graph_history_if_needed(self, record: SessionRecord, graph: EDAAgentGraph) -> None:
        if not record.messages:
            return
        lc_messages: list[BaseMessage] = []
        for msg in record.messages:
            if msg.role == "user":
                lc_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant" and msg.status == "success":
                summary = msg.content
                if msg.tool_traces:
                    trace_snippets: list[str] = []
                    for tr in msg.tool_traces:
                        if not tr.error and (tr.result is not None or tr.stdout):
                            val = tr.result if tr.result is not None else tr.stdout
                            trace_snippets.append(f"{tr.tool}: {str(val)[:600]}")
                    if trace_snippets:
                        summary = f"{summary}\n[Prior tool outputs: {'; '.join(trace_snippets)}]"
                lc_messages.append(AIMessage(content=summary))
        if lc_messages:
            graph.seed_history_if_empty(record.thread_id, lc_messages)

    def _ensure_graph(self, record: SessionRecord, cache: _SessionRuntimeCache) -> EDAAgentGraph:
        if cache.checkpointer is None:
            cache.checkpointer = (
                self.shared_checkpointer if self.shared_checkpointer is not None else MemorySaver()
            )
        if cache.graph is not None:
            self._restore_graph_history_if_needed(record, cache.graph)
            return cache.graph

        chat_model = self.llm_provider.build_chat_model(
            provider=record.provider,
            model=record.model,
            api_key=cache.api_key,
            request_timeout_seconds=self.settings.llm_request_timeout_seconds,
        )

        def df_provider() -> pd.DataFrame:
            latest = self._require_session(record.session_id)
            return self._ensure_df(latest, cache)

        def context_provider() -> str:
            latest = self._require_session(record.session_id)
            df, prof, _, _ = self._ensure_eda(latest, cache)
            return format_profile_for_prompt(prof, df)

        def stats_provider() -> StatisticalReport:
            latest = self._require_session(record.session_id)
            _, _, stats_rep, _ = self._ensure_eda(latest, cache)
            return stats_rep

        def profile_provider() -> DatasetProfile:
            latest = self._require_session(record.session_id)
            _, prof, _, _ = self._ensure_eda(latest, cache)
            return prof

        cache.graph = EDAAgentGraph(
            chat_model=chat_model,
            sandbox=self.sandbox,
            df_provider=df_provider,
            dataset_context_provider=context_provider,
            stats_provider=stats_provider,
            profile_provider=profile_provider,
            checkpointer=cache.checkpointer,
        )
        self._restore_graph_history_if_needed(record, cache.graph)
        return cache.graph

    # -- Public API methods ---------------------------------------------------

    def create_session(self, req: CreateSessionRequest) -> CreateSessionResponse:
        session_id = uuid.uuid4().hex
        thread_id = f"sb_{uuid.uuid4().hex[:12]}"
        provider = req.provider or self.settings.llm_provider
        model = req.model or self.settings.llm_model

        # Validate provider configuration eagerly before persisting session state
        self.llm_provider.build_chat_model(
            provider=provider,
            model=model,
            api_key=req.api_key,
        )

        record = SessionRecord(
            session_id=session_id,
            thread_id=thread_id,
            provider=provider,
            model=model,
        )
        self.session_repo.create(record)
        cache = self._get_cache(session_id)
        if req.api_key:
            cache.api_key = req.api_key
        return CreateSessionResponse(
            session_id=session_id,
            thread_id=thread_id,
            provider=provider,
            model=model,
            has_data=False,
        )

    def get_session_status(self, session_id: str) -> SessionStatusResponse:
        record = self._require_session(session_id)
        with self._cache_lock:
            existing = self._runtime_cache.get(session_id)
            if existing is not None:
                existing.touch()
        return SessionStatusResponse(
            session_id=record.session_id,
            thread_id=record.thread_id,
            provider=record.provider,
            model=record.model,
            has_data=record.dataset is not None,
            message_count=len(record.messages),
            figure_count=len(record.figures),
            dataset=record.dataset,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )

    def get_session_history(self, session_id: str) -> dict[str, Any]:
        record = self._require_session(session_id)
        with self._cache_lock:
            existing = self._runtime_cache.get(session_id)
            if existing is not None:
                existing.touch()
        return {
            "session_id": record.session_id,
            "messages": [m.model_dump() for m in record.messages],
            "last_tool_traces": [t.model_dump() for t in record.last_tool_traces],
        }

    def list_sessions(self) -> list[SessionStatusResponse]:
        records = self.session_repo.list_all()
        active_ids = {r.session_id for r in records}
        with self._cache_lock:
            self._evict_expired_or_excess_caches_locked()
            for sid in list(self._runtime_cache.keys()):
                if sid not in active_ids:
                    entry = self._runtime_cache.get(sid)
                    if entry is not None and entry.lock.acquire(blocking=False):
                        try:
                            if self.session_repo.get(sid) is None:
                                self._runtime_cache.pop(sid, None)
                                entry.release_all()
                        finally:
                            entry.lock.release()
        return [
            SessionStatusResponse(
                session_id=r.session_id,
                thread_id=r.thread_id,
                provider=r.provider,
                model=r.model,
                has_data=r.dataset is not None,
                message_count=len(r.messages),
                figure_count=len(r.figures),
                dataset=r.dataset,
                created_at=r.created_at,
                updated_at=r.updated_at,
            )
            for r in records
        ]

    def update_provider(self, session_id: str, req: UpdateProviderRequest) -> SessionStatusResponse:
        cache = self._get_cache(session_id)
        with cache.lock:
            record = self._require_session(session_id)
            api_key = req.api_key if req.api_key is not None else cache.api_key
            self.llm_provider.build_chat_model(
                provider=req.provider,
                model=req.model,
                api_key=api_key,
            )
            record.provider = req.provider
            record.model = req.model
            record.updated_at = utc_now_iso()
            saved = self.session_repo.save(record, expected_version=record.version)
            cache.api_key = api_key
            cache.graph = None
            cache.touch()
            return SessionStatusResponse(
                session_id=saved.session_id,
                thread_id=saved.thread_id,
                provider=saved.provider,
                model=saved.model,
                has_data=saved.dataset is not None,
                message_count=len(saved.messages),
                figure_count=len(saved.figures),
                dataset=saved.dataset,
                created_at=saved.created_at,
                updated_at=saved.updated_at,
            )

    def clear_session_chat(self, session_id: str) -> SessionStatusResponse:
        cache = self._get_cache(session_id)
        with cache.lock:
            record = self._require_session(session_id)
            record.thread_id = f"sb_{uuid.uuid4().hex[:12]}"
            record.messages = []
            record.figures = []
            record.last_tool_traces = []
            record.updated_at = utc_now_iso()
            saved = self.session_repo.save(record, expected_version=record.version)
            if self.shared_checkpointer is None:
                cache.checkpointer = MemorySaver()
            cache.graph = None
            cache.touch()
            return SessionStatusResponse(
                session_id=saved.session_id,
                thread_id=saved.thread_id,
                provider=saved.provider,
                model=saved.model,
                has_data=saved.dataset is not None,
                message_count=len(saved.messages),
                figure_count=len(saved.figures),
                dataset=saved.dataset,
                created_at=saved.created_at,
                updated_at=saved.updated_at,
            )

    def delete_session(self, session_id: str) -> None:
        cache = self._get_cache(session_id)
        with cache.lock:
            record = self._require_session(session_id)
            old_ref = record.dataset.storage_ref if record.dataset is not None else None
            self.session_repo.delete(session_id)
            if old_ref:
                with contextlib.suppress(Exception):
                    self.dataset_store.delete(old_ref)
            cache.release_all()
            with self._cache_lock:
                self._runtime_cache.pop(session_id, None)

    def upload_dataset(
        self, session_id: str, filename: str, content: bytes
    ) -> FullEDAReportResponse:
        # Verify session exists before doing heavy work
        self._require_session(session_id)
        cache = self._get_cache(session_id)

        with self.telemetry.span(
            "eda.upload_and_profile",
            {"session.id": session_id, "dataset.filename": filename},
        ):
            # 1. Stage and compute all derived artifacts BEFORE mutating storage or session state.
            df = parse_tabular_bytes(
                content,
                filename,
                max_upload_bytes=self.settings.max_upload_bytes,
            )
            digest = hashlib.sha256(content).hexdigest()
            new_profile = profile_dataframe(df, filename=filename)
            new_statistics = compute_statistics(df, new_profile)
            new_figures = build_eda_figures(df, new_profile, new_statistics)

            with cache.lock:
                record = self._require_session(session_id)
                old_storage_ref = record.dataset.storage_ref if record.dataset is not None else None
                staged_storage_ref: str | None = None
                try:
                    staged_storage_ref = self.dataset_store.put_bytes(session_id, filename, content)
                    updated_record = record.model_copy(deep=True)
                    updated_record.dataset = UploadedDatasetMeta(
                        filename=filename,
                        size_bytes=len(content),
                        sha256=digest,
                        rows=int(df.shape[0]),
                        columns=int(df.shape[1]),
                        storage_backend=self.dataset_store.backend_name,
                        storage_ref=staged_storage_ref,
                    )
                    updated_record.thread_id = f"sb_{uuid.uuid4().hex[:12]}"
                    updated_record.messages = []
                    updated_record.figures = []
                    updated_record.last_tool_traces = []
                    updated_record.updated_at = utc_now_iso()

                    self.session_repo.save(updated_record, expected_version=record.version)
                except Exception:
                    if staged_storage_ref and staged_storage_ref != old_storage_ref:
                        with contextlib.suppress(Exception):
                            self.dataset_store.delete(staged_storage_ref)
                    raise

                # Commit runtime cache state and clean up old storage artifact after save succeeded
                cache.df = df
                cache.dataset_sha256 = digest
                cache.dataset_storage_ref = staged_storage_ref
                cache.profile = new_profile
                cache.statistics = new_statistics
                cache.eda_figures = new_figures
                if self.shared_checkpointer is None:
                    cache.checkpointer = MemorySaver()
                cache.graph = None
                cache.touch()

                if old_storage_ref and old_storage_ref != staged_storage_ref:
                    with contextlib.suppress(Exception):
                        self.dataset_store.delete(old_storage_ref)

                return FullEDAReportResponse(
                    profile=new_profile,
                    statistics=new_statistics,
                    figures=new_figures,
                )

    def get_preview(
        self, session_id: str, rows: int = 10, sanitize_formulas: bool = True
    ) -> DatasetPreviewResponse:
        cache = self._get_cache(session_id)
        with cache.lock:
            record = self._require_session(session_id)
            df = self._ensure_df(record, cache)
            head_records = sanitize_json_payload(df.head(rows).to_dict(orient="records"))
            if sanitize_formulas:
                head_records = [
                    {k: neutralize_formula_value(v) for k, v in row.items()} for row in head_records
                ]
            return DatasetPreviewResponse(
                filename=record.dataset.filename if record.dataset else "dataset",
                shape=[int(df.shape[0]), int(df.shape[1])],
                columns=[str(c) for c in df.columns],
                dtypes={str(col): str(dt) for col, dt in df.dtypes.items()},
                preview=head_records,
            )

    def get_full_eda(self, session_id: str) -> FullEDAReportResponse:
        cache = self._get_cache(session_id)
        with self.telemetry.span("eda.full_report", {"session.id": session_id}):
            with cache.lock:
                record = self._require_session(session_id)
                _, prof, stats_rep, figs = self._ensure_eda(record, cache)
                return FullEDAReportResponse(
                    profile=prof,
                    statistics=stats_rep,
                    figures=figs,
                )

    def run_ml(self, session_id: str, request: MLExperimentRequest) -> MLExperimentResponse:
        cache = self._get_cache(session_id)
        with self.telemetry.span(
            "ml.experiment",
            {"session.id": session_id, "ml.target": request.target_column},
        ):
            with cache.lock:
                record = self._require_session(session_id)
                df = self._ensure_df(record, cache)
                result = run_ml_experiment(df, request)
                self.telemetry.ml_counter.add(1, {"task_type": result.task_type})
                return result

    def run_query(
        self,
        session_id: str,
        query: str,
        *,
        client_turn_id: str | None = None,
    ) -> QueryResponse:
        cache = self._get_cache(session_id)
        started = time.perf_counter()

        with cache.lock:
            record = self._require_session(session_id)
            normalized_turn_id = (client_turn_id or "").strip() or None
            if normalized_turn_id:
                for msg in reversed(record.messages):
                    if msg.role == "assistant" and msg.client_turn_id == normalized_turn_id:
                        return QueryResponse(
                            session_id=session_id,
                            answer=msg.content,
                            status=msg.status,
                            error=msg.error,
                            timed_out=False,
                            figures=list(msg.figures),
                            tool_traces=list(msg.tool_traces),
                            retries_used=int(msg.retries_used or 0),
                            latency_ms=float(msg.latency_ms or 0.0),
                            evaluation=msg.evaluation,
                        )

            with self.telemetry.span(
                "agent.query",
                {
                    "session.id": session_id,
                    "llm.provider": record.provider,
                    "llm.model": record.model,
                },
            ) as span:
                df, prof, _, _ = self._ensure_eda(record, cache)
                profile_context = format_profile_for_prompt(prof, df)
                graph = self._ensure_graph(record, cache)
                lf_cb = self.telemetry.create_langfuse_callback()
                callbacks = [lf_cb] if lf_cb is not None else None

                try:
                    raw_result = graph.invoke(
                        query=query,
                        thread_id=record.thread_id,
                        callbacks=callbacks,
                    )
                except LLMTimeoutError:
                    raise
                except (httpx.TimeoutException, TimeoutError) as exc:
                    raise LLMTimeoutError(
                        f"LLM inference timed out after "
                        f"{self.settings.llm_request_timeout_seconds}s: {exc}"
                    ) from exc
                except Exception as exc:
                    msg = str(exc)
                    if "timed out" in msg.lower() or "timeout" in msg.lower():
                        raise LLMTimeoutError(
                            f"LLM inference timed out after "
                            f"{self.settings.llm_request_timeout_seconds}s: {exc}"
                        ) from exc
                    raise LLMInvocationError(f"Agent query failed: {exc}") from exc
                finally:
                    self.telemetry.flush()

                traces = [
                    ToolTraceItem.model_validate(item)
                    for item in (raw_result.get("tool_traces") or [])
                ]
                figures = list(raw_result.get("turn_figures") or [])
                answer = normalize_text(raw_result.get("final_answer", ""))
                status = "failed" if raw_result.get("status") == "failed" else "success"
                terminal_error = raw_result.get("terminal_error")
                timed_out = bool(raw_result.get("timed_out", False))
                retries_used = int(raw_result.get("retries", 0))
                latency_ms = round((time.perf_counter() - started) * 1000.0, 2)

                prior_traces = [
                    trace_item
                    for prior_msg in record.messages
                    if prior_msg.role == "assistant"
                    for trace_item in prior_msg.tool_traces
                    if not trace_item.error
                ]
                eval_summary = evaluate_turn_deterministically(
                    query=query,
                    answer=answer,
                    tool_traces=traces,
                    figures=figures,
                    profile_context=profile_context,
                    status=status,
                    prior_tool_traces=prior_traces,
                )

                for trace_item in traces:
                    self.telemetry.sandbox_counter.add(
                        1,
                        {
                            "tool": trace_item.tool,
                            "status": "error" if trace_item.error else "ok",
                        },
                    )
                    self.telemetry.sandbox_duration_ms.record(trace_item.duration_ms)

                self.telemetry.query_counter.add(
                    1,
                    {"provider": record.provider, "model": record.model, "status": status},
                )
                self.telemetry.query_latency_ms.record(latency_ms)
                span.set_attribute("agent.retries", retries_used)
                span.set_attribute("agent.figure_count", len(figures))
                span.set_attribute("agent.status", status)
                span.set_attribute("eval.passed", bool(eval_summary["passed"]))

                record.messages.append(
                    ChatMessageRecord(
                        role="user",
                        content=query,
                        client_turn_id=normalized_turn_id,
                    )
                )
                record.messages.append(
                    ChatMessageRecord(
                        role="assistant",
                        content=answer,
                        figures=figures,
                        tool_traces=traces,
                        evaluation=eval_summary,
                        status=status,
                        error=terminal_error,
                        latency_ms=latency_ms,
                        retries_used=retries_used,
                        client_turn_id=normalized_turn_id,
                    )
                )
                record.figures.extend(figures)
                record.last_tool_traces = traces
                record.updated_at = utc_now_iso()
                self.session_repo.save(record, expected_version=record.version)
                cache.touch()

                return QueryResponse(
                    session_id=session_id,
                    answer=answer,
                    status=status,
                    error=terminal_error,
                    timed_out=timed_out,
                    figures=figures,
                    tool_traces=traces,
                    retries_used=retries_used,
                    latency_ms=latency_ms,
                    evaluation=eval_summary,
                )
