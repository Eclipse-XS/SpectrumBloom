"""Thread-safe session persistence repositories (in-memory and optional Redis)."""

from __future__ import annotations

import threading
import time
from typing import Any

from spectrumbloom.domain.schemas import SessionRecord


class InMemorySessionRepository:
    """Thread-safe in-memory session repository with TTL cleanup."""

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
            self._records[record.session_id] = record.model_copy(deep=True)
            self._expires_at[record.session_id] = time.monotonic() + self._ttl_seconds
            return record.model_copy(deep=True)

    def get(self, session_id: str) -> SessionRecord | None:
        with self._lock:
            self._evict_expired_locked()
            record = self._records.get(session_id)
            return record.model_copy(deep=True) if record is not None else None

    def save(self, record: SessionRecord) -> SessionRecord:
        with self._lock:
            self._evict_expired_locked()
            self._records[record.session_id] = record.model_copy(deep=True)
            self._expires_at[record.session_id] = time.monotonic() + self._ttl_seconds
            return record.model_copy(deep=True)

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

    def __init__(self, redis_url: str, ttl_seconds: int = 86400) -> None:
        try:
            import redis
        except ImportError as exc:
            raise RuntimeError(
                "Redis support requested but 'redis' package is not installed. "
                "Install with `uv pip install -e '.[redis]'`."
            ) from exc

        self._ttl_seconds = max(60, ttl_seconds)
        self._client: Any = redis.Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_timeout=3.0,
            socket_connect_timeout=3.0,
        )
        self._index_key = "spectrumbloom:sessions"

    def _key(self, session_id: str) -> str:
        return f"spectrumbloom:session:{session_id}"

    def create(self, record: SessionRecord) -> SessionRecord:
        return self.save(record)

    def get(self, session_id: str) -> SessionRecord | None:
        raw = self._client.get(self._key(session_id))
        if not raw:
            return None
        return SessionRecord.model_validate_json(raw)

    def save(self, record: SessionRecord) -> SessionRecord:
        payload = record.model_dump_json()
        pipe = self._client.pipeline()
        pipe.set(self._key(record.session_id), payload, ex=self._ttl_seconds)
        pipe.sadd(self._index_key, record.session_id)
        pipe.execute()
        return record.model_copy(deep=True)

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
