"""Safe CSV/TSV/Excel ingestion with size, encoding, and zip-bomb protections."""

from __future__ import annotations

import csv
import zipfile
from io import BytesIO, StringIO
from pathlib import Path

import pandas as pd

from spectrumbloom.domain.exceptions import DatasetValidationError

EXCEL_ENGINES: dict[str, str] = {
    ".xlsx": "openpyxl",
    ".xlsm": "openpyxl",
    ".xls": "xlrd",
    ".ods": "odf",
}
TEXT_EXTENSIONS = {".csv", ".tsv"}
SUPPORTED_EXTENSIONS = tuple(sorted([*TEXT_EXTENSIONS, *EXCEL_ENGINES.keys()]))

MAX_ZIP_ENTRIES = 100
MAX_UNCOMPRESSED_RATIO = 20
MAX_UNCOMPRESSED_BYTES = 150 * 1024 * 1024  # 150 MB safety cap


def _guard_zip_spreadsheet(content: bytes) -> None:
    """Reject malformed or zip-bomb OOXML/ODF archives before parsing."""
    try:
        with zipfile.ZipFile(BytesIO(content)) as zf:
            infos = zf.infolist()
            if len(infos) > MAX_ZIP_ENTRIES:
                raise DatasetValidationError(
                    f"Spreadsheet archive contains too many entries ({len(infos)})."
                )
            total_uncompressed = sum(info.file_size for info in infos)
            max_allowed = min(
                MAX_UNCOMPRESSED_BYTES,
                max(25 * 1024 * 1024, len(content) * MAX_UNCOMPRESSED_RATIO),
            )
            if total_uncompressed > max_allowed:
                raise DatasetValidationError(
                    "Spreadsheet uncompressed size exceeds safety limit (possible zip bomb)."
                )
    except zipfile.BadZipFile as exc:
        raise DatasetValidationError("Invalid or corrupted spreadsheet archive.") from exc


def _decode_csv_text(content: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise DatasetValidationError("Unable to decode text file using UTF-8 or Latin encodings.")


def _sniff_delimiter(sample_text: str, ext: str) -> str:
    if ext == ".tsv":
        return "\t"
    try:
        dialect = csv.Sniffer().sniff(sample_text[:8192], delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        return ","


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    seen: dict[str, int] = {}
    cleaned_cols: list[str] = []
    for idx, col in enumerate(df.columns):
        base = str(col).strip() or f"column_{idx}"
        count = seen.get(base, 0)
        seen[base] = count + 1
        cleaned_cols.append(base if count == 0 else f"{base}_{count}")
    out = df.copy()
    out.columns = cleaned_cols
    return out


def parse_tabular_bytes(
    content: bytes,
    filename: str,
    *,
    max_upload_bytes: int = 25 * 1024 * 1024,
    sheet_name: str | int = 0,
) -> pd.DataFrame:
    """Validate and parse uploaded CSV/TSV/Excel bytes into a normalized DataFrame."""
    if not content:
        raise DatasetValidationError("Uploaded file is empty.")
    if len(content) > max_upload_bytes:
        limit_mb = max_upload_bytes / (1024 * 1024)
        raise DatasetValidationError(
            f"Uploaded file exceeds maximum allowed size ({limit_mb:.1f} MB)."
        )

    ext = Path(filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise DatasetValidationError(
            f"Unsupported file extension '{ext}'. Allowed: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    try:
        if ext in TEXT_EXTENSIONS:
            text = _decode_csv_text(content)
            sep = _sniff_delimiter(text, ext)
            df = pd.read_csv(StringIO(text), sep=sep)
        else:
            if ext in {".xlsx", ".xlsm", ".ods"}:
                _guard_zip_spreadsheet(content)
            df = pd.read_excel(
                BytesIO(content),
                sheet_name=sheet_name,
                engine=EXCEL_ENGINES[ext],
            )
    except DatasetValidationError:
        raise
    except Exception as exc:
        raise DatasetValidationError(f"Failed to parse '{filename}': {exc}") from exc

    if df.empty or len(df.columns) == 0:
        raise DatasetValidationError("Parsed dataset contains no rows or columns.")

    return _normalize_columns(df)
