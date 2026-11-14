FROM python:3.12-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

COPY --from=ghcr.io/astral-sh/uv:0.6.10 /uv /uvx /bin/

WORKDIR /app

# Copy packaging metadata and source
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY examples ./examples

RUN uv pip install --system --no-cache --compile-bytecode ".[redis,eval]" && \
    python -m compileall -q /app/src && \
    groupadd -r spectrumbloom && \
    useradd -r -g spectrumbloom -d /app -s /sbin/nologin spectrumbloom && \
    mkdir -p /app/data/datasets && \
    chown -R spectrumbloom:spectrumbloom /app

USER spectrumbloom

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "spectrumbloom.main:app", "--host", "0.0.0.0", "--port", "8000"]
