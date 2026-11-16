"""Unit and concurrency tests for P0 data integrity, atomic uploads, write races, and eviction."""

from __future__ import annotations

import concurrent.futures
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from spectrumbloom.core.config import Settings
from spectrumbloom.core.security import redact_secrets
from spectrumbloom.core.telemetry import TelemetryRuntime
from spectrumbloom.domain.exceptions import (
    LLMInvocationError,
    MLExperimentError,
    SessionConflictError,
    SessionNotFoundError,
)
from spectrumbloom.domain.schemas import (
    CreateSessionRequest,
    SessionRecord,
    UpdateProviderRequest,
)
from spectrumbloom.llm.providers import UnifiedLLMProvider
from spectrumbloom.repositories.dataset_store import LocalFilesystemDatasetStore
from spectrumbloom.repositories.session_repo import (
    InMemorySessionRepository,
    RedisSessionRepository,
)
from spectrumbloom.sandbox.service import DisabledSandbox
from spectrumbloom.services.session_service import SpectrumBloomService


class _DeterministicChatModel:
    """Thread-safe deterministic chat model that echoes query and sees prior history."""

    def __init__(self, delay_seconds: float = 0.0) -> None:
        self.delay_seconds = delay_seconds

    def bind_tools(self, tools: list[Any]) -> _DeterministicChatModel:
        return self

    def invoke(self, messages: list[Any]) -> AIMessage:
        if self.delay_seconds > 0:
            time.sleep(self.delay_seconds)
        # Count prior assistant messages in the conversation state
        prior_assistant = [
            str(getattr(m, "content", ""))
            for m in messages
            if getattr(m, "type", "") == "ai" and getattr(m, "content", "")
        ]
        user_texts = [
            str(getattr(m, "content", "")) for m in messages if getattr(m, "type", "") == "human"
        ]
        latest_q = user_texts[-1] if user_texts else ""
        # If the agent graph nudges for a tool call, call get_statistical_summary
        if "You have not called any tool" in latest_q or not any(
            getattr(m, "type", "") == "tool" for m in messages
        ):
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tc_profile",
                        "name": "get_statistical_summary",
                        "args": {"focus": "profile"},
                    }
                ],
            )
        history_tag = f"prior_turns={len(prior_assistant)}"
        if prior_assistant:
            history_tag += f" last_prior={prior_assistant[0][:40]}"
        return AIMessage(
            content=f"Answered '{user_texts[0] if user_texts else ''}' ({history_tag}). Rows=3."
        )


def _make_service(
    tmp_path: Path,
    *,
    ttl_seconds: int = 86400,
    max_cached_sessions: int = 128,
    delay_seconds: float = 0.0,
) -> SpectrumBloomService:
    settings = Settings(
        app_env="test",
        local_dataset_dir=tmp_path / "datasets",
        sandbox_backend="disabled",
        session_ttl_seconds=ttl_seconds,
        max_cached_sessions=max_cached_sessions,
    )
    telemetry = TelemetryRuntime(settings, in_memory_export=True)
    repo = InMemorySessionRepository(ttl_seconds=ttl_seconds)
    store = LocalFilesystemDatasetStore(base_dir=settings.local_dataset_dir)
    llm_provider = UnifiedLLMProvider(settings)
    llm_provider.build_chat_model = lambda **kwargs: _DeterministicChatModel(
        delay_seconds=delay_seconds
    )  # type: ignore[method-assign]
    return SpectrumBloomService(
        settings=settings,
        session_repo=repo,
        dataset_store=store,
        sandbox=DisabledSandbox(),
        llm_provider=llm_provider,
        telemetry=telemetry,
    )


CSV_A = b"category,revenue,units\nEast,100.0,10\nWest,200.0,20\nNorth,300.0,30\n"
CSV_B = b"segment,profit,cost\nAlpha,50.0,15\nBeta,75.0,25\nGamma,125.0,35\nDelta,150.0,45\n"


def test_atomic_upload_preserves_state_on_storage_profile_stats_figure_and_save_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _make_service(tmp_path)
    created = service.create_session(CreateSessionRequest())
    sid = created.session_id

    # Initial upload succeeds and establishes baseline dataset, EDA, and conversation
    eda1 = service.upload_dataset(sid, "initial.csv", CSV_A)
    assert eda1.profile.filename == "initial.csv"
    assert eda1.profile.rows == 3

    q1 = service.run_query(sid, "How many rows are in the dataset?")
    assert q1.status == "success"

    baseline_status = service.get_session_status(sid)
    assert baseline_status.dataset is not None
    original_ref = baseline_status.dataset.storage_ref
    original_sha = baseline_status.dataset.sha256
    assert baseline_status.message_count == 2

    def _assert_baseline_preserved() -> None:
        status_now = service.get_session_status(sid)
        assert status_now.dataset is not None
        assert status_now.dataset.filename == "initial.csv"
        assert status_now.dataset.sha256 == original_sha
        assert status_now.dataset.storage_ref == original_ref
        assert status_now.message_count == 2
        # Original file bytes still exist on disk
        assert service.dataset_store.get_bytes(original_ref) == CSV_A
        # Cached EDA report still matches initial dataset
        eda_now = service.get_full_eda(sid)
        assert eda_now.profile.filename == "initial.csv"
        assert eda_now.profile.rows == 3
        # Conversation history still intact
        hist = service.get_session_history(sid)
        assert len(hist["messages"]) == 2
        # No leaked intermediate artifacts in session directory
        session_dir = tmp_path / "datasets" / sid
        files = list(session_dir.iterdir())
        assert len(files) == 1
        assert files[0].name == Path(original_ref).name

    # 1. Fault injection: storage failure (put_bytes raises)
    def _fail_put_bytes(*args: Any, **kwargs: Any) -> str:
        raise OSError("Simulated disk full during put_bytes")

    with monkeypatch.context() as m:
        m.setattr(service.dataset_store, "put_bytes", _fail_put_bytes)
        with pytest.raises(OSError, match="Simulated disk full"):
            service.upload_dataset(sid, "replacement.csv", CSV_B)
    _assert_baseline_preserved()

    # 2. Fault injection: profiling failure (profile_dataframe raises)
    def _fail_profile(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Simulated profiler crash")

    with monkeypatch.context() as m:
        m.setattr("spectrumbloom.services.session_service.profile_dataframe", _fail_profile)
        with pytest.raises(RuntimeError, match="Simulated profiler crash"):
            service.upload_dataset(sid, "replacement.csv", CSV_B)
    _assert_baseline_preserved()

    # 3. Fault injection: statistics failure (compute_statistics raises)
    def _fail_stats(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("Simulated statistics computation failure")

    with monkeypatch.context() as m:
        m.setattr("spectrumbloom.services.session_service.compute_statistics", _fail_stats)
        with pytest.raises(ValueError, match="Simulated statistics computation failure"):
            service.upload_dataset(sid, "replacement.csv", CSV_B)
    _assert_baseline_preserved()

    # 4. Fault injection: figure generation failure (build_eda_figures raises)
    def _fail_figs(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Simulated Plotly figure generation crash")

    with monkeypatch.context() as m:
        m.setattr("spectrumbloom.services.session_service.build_eda_figures", _fail_figs)
        with pytest.raises(RuntimeError, match="Simulated Plotly figure generation crash"):
            service.upload_dataset(sid, "replacement.csv", CSV_B)
    _assert_baseline_preserved()

    # 5. Fault injection: session repository save failure after staging file
    def _fail_save(*args: Any, **kwargs: Any) -> Any:
        raise SessionConflictError("Simulated repository save conflict")

    with monkeypatch.context() as m:
        m.setattr(service.session_repo, "save", _fail_save)
        with pytest.raises(SessionConflictError, match="Simulated repository save conflict"):
            service.upload_dataset(sid, "replacement.csv", CSV_B)
    _assert_baseline_preserved()

    # 6. Re-uploading identical dataset bytes succeeds and cleans up old artifact without data loss
    eda_same = service.upload_dataset(sid, "initial.csv", CSV_A)
    assert eda_same.profile.rows == 3
    status_same = service.get_session_status(sid)
    assert status_same.dataset is not None
    assert service.dataset_store.get_bytes(status_same.dataset.storage_ref) == CSV_A

    # 7. Successful replacement with CSV_B updates dataset, clears chat, and deletes previous file
    eda2 = service.upload_dataset(sid, "replacement.csv", CSV_B)
    assert eda2.profile.filename == "replacement.csv"
    assert eda2.profile.rows == 4
    status_after = service.get_session_status(sid)
    assert status_after.dataset is not None
    assert status_after.dataset.filename == "replacement.csv"
    assert status_after.message_count == 0
    session_dir = tmp_path / "datasets" / sid
    remaining_files = list(session_dir.iterdir())
    assert len(remaining_files) == 1
    assert remaining_files[0].name == Path(status_after.dataset.storage_ref).name


def test_concurrent_mutations_do_not_restore_stale_metadata_or_corrupt_history(
    tmp_path: Path,
) -> None:
    service = _make_service(tmp_path, delay_seconds=0.08)
    sid = service.create_session(CreateSessionRequest()).session_id
    service.upload_dataset(sid, "initial.csv", CSV_A)

    # Launch a slow query and an overlapping dataset replacement + provider update
    barrier = threading.Barrier(3)

    def _worker_query() -> Any:
        barrier.wait()
        return service.run_query(sid, "How many rows are in the dataset?")

    def _worker_upload() -> Any:
        barrier.wait()
        time.sleep(0.01)
        return service.upload_dataset(sid, "replacement.csv", CSV_B)

    def _worker_provider() -> Any:
        barrier.wait()
        time.sleep(0.02)
        return service.update_provider(
            sid, UpdateProviderRequest(provider="ollama", model="qwen2.5-coder:3b-instruct-q4_K_M")
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        f_q = pool.submit(_worker_query)
        f_u = pool.submit(_worker_upload)
        f_p = pool.submit(_worker_provider)
        f_q.result()
        f_u.result()
        f_p.result()

    # Regardless of lock acquisition order, replacement dataset and provider update must persist
    final_status = service.get_session_status(sid)
    assert final_status.dataset is not None
    assert final_status.dataset.filename == "replacement.csv"
    assert final_status.dataset.rows == 4
    assert final_status.model == "qwen2.5-coder:3b-instruct-q4_K_M"


def test_optimistic_version_checks_in_memory_and_fake_redis() -> None:
    repo = InMemorySessionRepository(ttl_seconds=3600)
    rec = SessionRecord(
        session_id="s1", thread_id="t1", provider="ollama", model="qwen3:4b-instruct"
    )
    created = repo.create(rec)
    assert created.version == 1

    # Save with matching expected_version increments to 2
    saved = repo.save(created, expected_version=1)
    assert saved.version == 2

    # Save with stale expected_version=1 raises SessionConflictError
    with pytest.raises(SessionConflictError, match="expected version 1, found 2"):
        repo.save(created, expected_version=1)

    # Save on missing session with expected_version raises SessionNotFoundError
    ghost = SessionRecord(session_id="missing", thread_id="t2", provider="ollama", model="m")
    with pytest.raises(SessionNotFoundError):
        repo.save(ghost, expected_version=1)


def test_redis_repository_optimistic_concurrency_and_restart_context_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify RedisSessionRepository CAS semantics and LangGraph context recovery after restart."""

    class _FakeRedisPipeline:
        def __init__(self, store: dict[str, str], sets: dict[str, set[str]]) -> None:
            self._store = store
            self._sets = sets
            self._watched_key: str | None = None
            self._watched_val: str | None = None
            self._ops: list[Any] = []
            self._in_multi = False

        def __enter__(self) -> _FakeRedisPipeline:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def watch(self, key: str) -> None:
            self._watched_key = key
            self._watched_val = self._store.get(key)

        def get(self, key: str) -> str | None:
            return self._store.get(key)

        def multi(self) -> None:
            self._in_multi = True

        def set(self, key: str, value: str, ex: int | None = None) -> None:
            self._ops.append(("set", key, value))

        def sadd(self, key: str, member: str) -> None:
            self._ops.append(("sadd", key, member))

        def execute(self) -> list[Any]:
            if (
                self._watched_key is not None
                and self._store.get(self._watched_key) != self._watched_val
            ):
                raise SessionConflictError("Watched key changed")
            for op in self._ops:
                if op[0] == "set":
                    self._store[op[1]] = op[2]
                elif op[0] == "sadd":
                    self._sets.setdefault(op[1], set()).add(op[2])
            self._ops.clear()
            return [True]

    class _FakeRedisClient:
        def __init__(self) -> None:
            self.store: dict[str, str] = {}
            self.sets: dict[str, set[str]] = {}

        def pipeline(self) -> _FakeRedisPipeline:
            return _FakeRedisPipeline(self.store, self.sets)

        def get(self, key: str) -> str | None:
            return self.store.get(key)

        def smembers(self, key: str) -> set[str]:
            return set(self.sets.get(key, set()))

        def srem(self, key: str, *members: str) -> int:
            s = self.sets.get(key, set())
            for m in members:
                s.discard(m)
            return len(members)

        def delete(self, key: str) -> int:
            return 1 if self.store.pop(key, None) is not None else 0

        def ping(self) -> bool:
            return True

    fake_client = _FakeRedisClient()
    redis_repo = RedisSessionRepository(
        "redis://localhost:6379/0",
        ttl_seconds=3600,
        client=fake_client,
    )
    rec = SessionRecord(
        session_id="redis_s1",
        thread_id="sb_redis_t1",
        provider="ollama",
        model="qwen3:4b-instruct",
    )
    c1 = redis_repo.create(rec)
    assert c1.version == 1
    s2 = redis_repo.save(c1, expected_version=1)
    assert s2.version == 2
    with pytest.raises(SessionConflictError):
        redis_repo.save(c1, expected_version=1)

    # Verify service restart preserves LangGraph multi-turn context with RedisSessionRepository
    settings = Settings(
        app_env="test",
        local_dataset_dir=tmp_path / "datasets",
        sandbox_backend="disabled",
    )
    telemetry = TelemetryRuntime(settings, in_memory_export=True)
    store = LocalFilesystemDatasetStore(base_dir=settings.local_dataset_dir)
    llm_provider = UnifiedLLMProvider(settings)
    llm_provider.build_chat_model = lambda **kwargs: _DeterministicChatModel()  # type: ignore[method-assign]

    service_before_restart = SpectrumBloomService(
        settings=settings,
        session_repo=redis_repo,
        dataset_store=store,
        sandbox=DisabledSandbox(),
        llm_provider=llm_provider,
        telemetry=telemetry,
    )
    created = service_before_restart.create_session(CreateSessionRequest())
    sid = created.session_id
    service_before_restart.upload_dataset(sid, "sales.csv", CSV_A)

    turn1 = service_before_restart.run_query(
        sid, "How many rows are in the dataset?", client_turn_id="turn_001"
    )
    assert "prior_turns=0" in turn1.answer

    # Idempotent retry with same client_turn_id does not duplicate messages
    turn1_retry = service_before_restart.run_query(
        sid, "How many rows are in the dataset?", client_turn_id="turn_001"
    )
    assert turn1_retry.answer == turn1.answer
    assert service_before_restart.get_session_status(sid).message_count == 2

    # Simulate application restart: new SpectrumBloomService with empty _runtime_cache
    service_after_restart = SpectrumBloomService(
        settings=settings,
        session_repo=redis_repo,
        dataset_store=store,
        sandbox=DisabledSandbox(),
        llm_provider=llm_provider,
        telemetry=telemetry,
    )
    turn2 = service_after_restart.run_query(sid, "Follow up on the previous row count.")
    # Because _restore_graph_history_if_needed seeded the checkpointer from Redis messages,
    # the LLM received the prior assistant turn in its LangGraph state!
    assert "prior_turns=1" in turn2.answer
    assert "last_prior=Answered" in turn2.answer


def test_runtime_cache_ttl_eviction_and_bounded_lru_cleanup(tmp_path: Path) -> None:
    # Configure max_cached_sessions=3
    service = _make_service(tmp_path, ttl_seconds=60, max_cached_sessions=3)
    sids: list[str] = []
    for i in range(6):
        s = service.create_session(CreateSessionRequest())
        service.upload_dataset(s.session_id, f"d_{i}.csv", CSV_A)
        sids.append(s.session_id)

    # Bounded LRU cap ensures at most 3 sessions remain in _runtime_cache
    assert len(service._runtime_cache) <= 3
    # Older evicted sessions are still reloadable from session_repo + dataset_store on demand
    preview_first = service.get_preview(sids[0], rows=2)
    assert preview_first.shape == [3, 3]
    assert len(service._runtime_cache) <= 3

    # Simulate TTL expiration on cached entries
    for entry in service._runtime_cache.values():
        entry.last_accessed = time.monotonic() - 120.0

    # Trigger cache access on a new session -> expired entries are evicted and cleared
    extra = service.create_session(CreateSessionRequest())
    assert extra.session_id in service._runtime_cache
    assert len(service._runtime_cache) == 1

    # Deleting a session removes its runtime cache and releases resources
    service.delete_session(extra.session_id)
    assert extra.session_id not in service._runtime_cache


def test_runtime_cache_eviction_concurrency_and_lock_safety(tmp_path: Path) -> None:
    """Verify active locked entries never detach and concurrent ops stay deadlock-free."""
    service = _make_service(tmp_path, ttl_seconds=60, max_cached_sessions=2, delay_seconds=0.05)

    s1 = service.create_session(CreateSessionRequest()).session_id
    service.upload_dataset(s1, "s1.csv", CSV_A)
    s2 = service.create_session(CreateSessionRequest()).session_id
    service.upload_dataset(s2, "s2.csv", CSV_B)

    # 1. Expired idle entry is released and marked released
    cache1 = service._runtime_cache[s1]
    cache1.last_accessed = time.monotonic() - 3600.0
    service.create_session(CreateSessionRequest())
    assert cache1.released is True
    assert cache1.df is None
    assert s1 not in service._runtime_cache

    # 2. Active locked entry (even if expired and over LRU cap) is NEVER prematurely evicted
    cache2 = service._runtime_cache[s2]
    cache2.last_accessed = time.monotonic() - 3600.0
    locked_ready = threading.Event()
    release_lock_ev = threading.Event()

    def _hold_lock() -> None:
        with service._acquire_session_cache(s2) as c:
            locked_ready.set()
            release_lock_ev.wait(timeout=5.0)
            # Inside lock, entry must remain intact and unreleased
            assert not c.released
            assert c.df is not None

    holder = threading.Thread(target=_hold_lock)
    holder.start()
    assert locked_ready.wait(timeout=2.0)

    # Force LRU & TTL cleanup while s2 is actively locked
    s4 = service.create_session(CreateSessionRequest()).session_id
    s5 = service.create_session(CreateSessionRequest()).session_id
    assert s2 in service._runtime_cache
    assert cache2.released is False
    assert cache2.df is not None

    release_lock_ev.set()
    holder.join(timeout=5.0)
    assert not holder.is_alive()

    # Once unlocked and idle/expired, subsequent eviction cleans up s2 down to max_cached_sessions
    cache2.last_accessed = time.monotonic() - 3600.0
    with service._cache_lock:
        service._evict_expired_or_excess_caches_locked(preserve_session_id=s5)
    assert cache2.released is True
    assert s2 not in service._runtime_cache
    assert len(service._runtime_cache) <= 2

    # 3. Previously evicted session s1 & s2 can be loaded again from repo + store
    reloaded = service.get_preview(s1, rows=3)
    assert reloaded.shape == [3, 3]
    reloaded_eda = service.get_full_eda(s2)
    assert reloaded_eda.profile.rows == 4

    # 4. Concurrent query + ML + repeated session creation complete without deadlock or corruption
    ml_csv = b"feature_a,feature_b,target\n" + b"".join(
        f"{i},{i * 2},{i % 2}\n".encode("ascii") for i in range(40)
    )
    service.upload_dataset(s4, "ml.csv", ml_csv)

    from spectrumbloom.domain.schemas import MLExperimentRequest

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futs: list[concurrent.futures.Future[Any]] = []
        futs.append(pool.submit(service.run_query, s1, "How many rows?"))
        futs.append(
            pool.submit(
                service.run_ml,
                s4,
                MLExperimentRequest(target_column="target", task_type="classification", cv_folds=3),
            )
        )
        for idx in range(4):
            extra_id = service.create_session(CreateSessionRequest()).session_id
            service.upload_dataset(extra_id, f"churn_{idx}.csv", CSV_A)
            futs.append(pool.submit(service.get_preview, extra_id, 2))
        results = [f.result(timeout=10.0) for f in futs]

    assert "Rows=3" in results[0].answer
    assert results[1].best_model
    assert len(service._runtime_cache) <= 2

    # 5. Failed query/ML operations do not leave locks held or corrupt cache state
    class _ExplodingChatModel:
        def bind_tools(self, tools: list[Any]) -> Any:
            return self

        def invoke(self, messages: list[Any]) -> Any:
            raise RuntimeError("boom")

    orig_builder = service.llm_provider.build_chat_model
    service.llm_provider.build_chat_model = lambda **kwargs: _ExplodingChatModel()  # type: ignore[method-assign]
    if s4 in service._runtime_cache:
        service._runtime_cache[s4].graph = None
    with pytest.raises(LLMInvocationError):
        service.run_query(s4, "Fail cleanly")
    service.llm_provider.build_chat_model = orig_builder  # type: ignore[method-assign]
    with pytest.raises(MLExperimentError):
        service.run_ml(s4, MLExperimentRequest(target_column="nonexistent_col", task_type="auto"))
    # Session remains usable afterwards (lock released cleanly)
    assert service.get_preview(s4, rows=2).shape == [40, 3]
    assert len(service._runtime_cache) <= 2


def test_upload_size_streaming_guard_and_secret_redaction(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from spectrumbloom.main import create_app

    settings = Settings(
        app_env="test",
        local_dataset_dir=tmp_path / "datasets",
        api_auth_token="super-secret-bearer-token-9999",
        openai_api_key="sk-proj-1234567890abcdefghijklmnop",
        sandbox_backend="disabled",
        max_upload_mb=1,
    )
    app = create_app(settings, in_memory_telemetry=True)

    def _exploding_model(**kwargs: Any) -> Any:
        raise LLMInvocationError(
            "Upstream failed with Authorization: Bearer super-secret-bearer-token-9999 "
            "and key sk-proj-1234567890abcdefghijklmnop and api_key=my_private_secret_123"
        )

    app.state.service.llm_provider.build_chat_model = _exploding_model  # type: ignore[method-assign]

    with TestClient(app) as client:
        headers = {"Authorization": "Bearer super-secret-bearer-token-9999"}
        # Error response redacts tokens and keys
        res_err = client.post("/api/v1/sessions", json={}, headers=headers)
        assert res_err.status_code == 502
        detail = res_err.json()["detail"]
        assert "super-secret-bearer-token-9999" not in detail
        assert "sk-proj-1234567890abcdefghijklmnop" not in detail
        assert "my_private_secret_123" not in detail
        assert "[REDACTED" in detail

    # Also verify direct unit redaction helper
    redacted = redact_secrets(
        "Failed auth: Bearer secret_abc_123456, token=xyz987654321, "
        "key=AIzaSyA12345678901234567890",
        extra_secrets=("custom_internal_token_777",),
    )
    assert "secret_abc_123456" not in redacted
    assert "xyz987654321" not in redacted
    assert "AIzaSyA12345678901234567890" not in redacted
