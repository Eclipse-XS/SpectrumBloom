"""Dataset storage backends (local filesystem default + optional S3-compatible)."""

from __future__ import annotations

import hashlib
import mimetypes
from pathlib import Path
from typing import Any

from spectrumbloom.core.security import sanitize_filename
from spectrumbloom.domain.exceptions import DatasetNotFoundError, DatasetValidationError


class LocalFilesystemDatasetStore:
    """Stores uploaded datasets safely inside a confined local directory."""

    backend_name = "local"

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir.resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_safe_path(self, relative_ref: str) -> Path:
        target = (self.base_dir / relative_ref).resolve()
        if not target.is_relative_to(self.base_dir):
            raise DatasetValidationError("Invalid dataset storage path.")
        return target

    def put_bytes(self, session_id: str, filename: str, content: bytes) -> str:
        safe_session = sanitize_filename(session_id)
        safe_name = sanitize_filename(filename)
        digest = hashlib.sha256(content).hexdigest()[:12]
        rel_ref = f"{safe_session}/{digest}_{safe_name}"
        target = self._resolve_safe_path(rel_ref)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return rel_ref

    def get_bytes(self, storage_ref: str) -> bytes:
        target = self._resolve_safe_path(storage_ref)
        if not target.exists() or not target.is_file():
            raise DatasetNotFoundError(f"Dataset artifact '{storage_ref}' not found.")
        return target.read_bytes()

    def delete(self, storage_ref: str) -> None:
        target = self._resolve_safe_path(storage_ref)
        if target.exists() and target.is_file():
            target.unlink()
            try:
                if target.parent != self.base_dir and not any(target.parent.iterdir()):
                    target.parent.rmdir()
            except OSError:
                pass

    def ping(self) -> bool:
        return self.base_dir.exists() and self.base_dir.is_dir()


class S3CompatibleDatasetStore:
    """Optional S3-compatible object storage (MinIO, Cloudflare R2, GCS interop, S3)."""

    backend_name = "s3"

    def __init__(
        self,
        *,
        bucket: str,
        prefix: str = "datasets",
        endpoint_url: str | None = None,
        region: str | None = None,
        delete_on_session_delete: bool = True,
        client: Any = None,
    ) -> None:
        if not bucket:
            raise ValueError("S3 bucket name is required.")
        self.bucket = bucket
        self.prefix = prefix.strip("/") or "datasets"
        self.delete_on_session_delete = delete_on_session_delete

        if client is not None:
            self._client = client
        else:
            try:
                import boto3
            except ImportError as exc:
                raise RuntimeError(
                    "S3 storage requested but 'boto3' is not installed. "
                    "Install with `uv pip install -e '.[s3]'`."
                ) from exc
            kwargs: dict[str, Any] = {}
            if endpoint_url:
                kwargs["endpoint_url"] = endpoint_url
            if region:
                kwargs["region_name"] = region
            self._client = boto3.client("s3", **kwargs)

    def put_bytes(self, session_id: str, filename: str, content: bytes) -> str:
        safe_session = sanitize_filename(session_id)
        safe_name = sanitize_filename(filename)
        digest = hashlib.sha256(content).hexdigest()[:12]
        key = f"{self.prefix}/{safe_session}/{digest}_{safe_name}"
        content_type = mimetypes.guess_type(safe_name)[0] or "application/octet-stream"
        self._client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=content,
            ContentType=content_type,
        )
        return key

    def get_bytes(self, storage_ref: str) -> bytes:
        try:
            response = self._client.get_object(Bucket=self.bucket, Key=storage_ref)
            return bytes(response["Body"].read())
        except Exception as exc:
            raise DatasetNotFoundError(f"Unable to load S3 object '{storage_ref}'.") from exc

    def delete(self, storage_ref: str) -> None:
        if not self.delete_on_session_delete:
            return
        try:
            self._client.delete_object(Bucket=self.bucket, Key=storage_ref)
        except Exception:
            return

    def ping(self) -> bool:
        try:
            self._client.head_bucket(Bucket=self.bucket)
            return True
        except Exception:
            return False
