"""Thread-safe session persistence repositories (in-memory and optional Redis)."""

from __future__ import annotations

import threading
import time
from typing import Any

from spectrumbloom.domain.exceptions import SessionConflictError, SessionNotFoundError
from spectrumbloom.domain.schemas import SessionRecord


class InMemorySessionRepository:
    """Thread-safe in-memory session repository with TTL cleanup and optimistic concurrency."""

    def __init__(self, ttl_seconds: int = 86400) -> None:
        self._ttl_seconds = max(60, ttl_seconds)
        self._lock = threading.RLock()
        self._records: dict[str, SessionRecord] = {}
        self._expires_at: dict[str, float] = {}

    def _evict_expired_locked(self) -> None:
        now = time.monotonic()
        expired = [sid for sid, exp in self._expires_at.items() if exp <= now]
        for sid in expired:
            self._records.pop(sid, None)
            self._expires_at.pop(sid, None)

    def create(self, record: SessionRecord) -> SessionRecord:
        with self._lock:
            self._evict_expired_locked()
            to_store = record.model_copy(deep=True)
            to_store.version = max(1, int(to_store.version))
            self._records[to_store.session_id] = to_store
            self._expires_at[to_store.session_id] = time.monotonic() + self._ttl_seconds
            return to_store.model_copy(deep=True)

    def get(self, session_id: str) -> SessionRecord | None:
        with self._lock:
            self._evict_expired_locked()
            record = self._records.get(session_id)
            return record.model_copy(deep=True) if record is not None else None

    def save(
        self,
        record: SessionRecord,
        *,
        expected_version: int | None = None,
    ) -> SessionRecord:
        with self._lock:
            self._evict_expired_locked()
            existing = self._records.get(record.session_id)
            if expected_version is not None:
                if existing is None:
                    raise SessionNotFoundError(
                        f"Session '{record.session_id}' does not exist or has expired."
                    )
                if existing.version != expected_version:
                    raise SessionConflictError(
                        f"Concurrent modification detected for session '{record.session_id}' "
                        f"(expected version {expected_version}, found {existing.version})."
                    )
                next_version = expected_version + 1
            else:
                next_version = (
                    (existing.version + 1) if existing is not None else max(1, record.version)
                )
            to_store = record.model_copy(deep=True)
            to_store.version = next_version
            self._records[to_store.session_id] = to_store
            self._expires_at[to_store.session_id] = time.monotonic() + self._ttl_seconds
            return to_store.model_copy(deep=True)

    def list_all(self) -> list[SessionRecord]:
        with self._lock:
            self._evict_expired_locked()
            return [rec.model_copy(deep=True) for rec in self._records.values()]

    def delete(self, session_id: str) -> bool:
        with self._lock:
            existed = session_id in self._records
            self._records.pop(session_id, None)
            self._expires_at.pop(session_id, None)
            return existed

    def ping(self) -> bool:
        return True


class RedisSessionRepository:
    """Optional Redis-backed session repository (no mandatory dependency)."""

    def __init__(
        self,
        redis_url: str,
        ttl_seconds: int = 86400,
        *,
        client: Any = None,
    ) -> None:
        self._ttl_seconds = max(60, ttl_seconds)
        self._index_key = "spectrumbloom:sessions"
        if client is not None:
            self._client: Any = client
            return
        try:
            import redis
        except ImportError as exc:
            raise RuntimeError(
                "Redis support requested but 'redis' package is not installed. "
                "Install with `uv pip install -e '.[redis]'`."
            ) from exc

        self._client = redis.Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_timeout=3.0,
            socket_connect_timeout=3.0,
        )

    def _key(self, session_id: str) -> str:
        return f"spectrumbloom:session:{session_id}"

    def create(self, record: SessionRecord) -> SessionRecord:
        to_store = record.model_copy(deep=True)
        to_store.version = max(1, int(to_store.version))
        payload = to_store.model_dump_json()
        pipe = self._client.pipeline()
        pipe.set(self._key(to_store.session_id), payload, ex=self._ttl_seconds)
        pipe.sadd(self._index_key, to_store.session_id)
        pipe.execute()
        return to_store.model_copy(deep=True)

    def get(self, session_id: str) -> SessionRecord | None:
        raw = self._client.get(self._key(session_id))
        if not raw:
            return None
        return SessionRecord.model_validate_json(raw)

    def save(
        self,
        record: SessionRecord,
        *,
        expected_version: int | None = None,
    ) -> SessionRecord:
        key = self._key(record.session_id)
        if expected_version is None:
            to_store = record.model_copy(deep=True)
            existing = self.get(record.session_id)
            to_store.version = (
                (existing.version + 1) if existing is not None else max(1, to_store.version)
            )
            payload = to_store.model_dump_json()
            pipe = self._client.pipeline()
            pipe.set(key, payload, ex=self._ttl_seconds)
            pipe.sadd(self._index_key, to_store.session_id)
            pipe.execute()
            return to_store.model_copy(deep=True)

        try:
            import redis

            watch_err_cls: type[Exception] = redis.WatchError
        except ImportError:

            class _FallbackWatchError(Exception):
                pass

            watch_err_cls = _FallbackWatchError

        with self._client.pipeline() as pipe:
            try:
                pipe.watch(key)
                raw = pipe.get(key)
                if not raw:
                    raise SessionNotFoundError(
                        f"Session '{record.session_id}' does not exist or has expired."
                    )
                current = SessionRecord.model_validate_json(raw)
                if current.version != expected_version:
                    raise SessionConflictError(
                        f"Concurrent modification detected for session '{record.session_id}' "
                        f"(expected version {expected_version}, found {current.version})."
                    )
                to_store = record.model_copy(deep=True)
                to_store.version = expected_version + 1
                payload = to_store.model_dump_json()
                pipe.multi()
                pipe.set(key, payload, ex=self._ttl_seconds)
                pipe.sadd(self._index_key, to_store.session_id)
                pipe.execute()
                return to_store.model_copy(deep=True)
            except (SessionNotFoundError, SessionConflictError):
                raise
            except watch_err_cls as exc:
                raise SessionConflictError(
                    f"Concurrent Redis modification detected for session '{record.session_id}'."
                ) from exc

    def list_all(self) -> list[SessionRecord]:
        session_ids = sorted(self._client.smembers(self._index_key))
        records: list[SessionRecord] = []
        stale: list[str] = []
        for sid in session_ids:
            record = self.get(sid)
            if record is not None:
                records.append(record)
            else:
                stale.append(sid)
        if stale:
            self._client.srem(self._index_key, *stale)
        return records

    def delete(self, session_id: str) -> bool:
        deleted = int(self._client.delete(self._key(session_id)))
        self._client.srem(self._index_key, session_id)
        return deleted > 0

    def ping(self) -> bool:
        return bool(self._client.ping())
