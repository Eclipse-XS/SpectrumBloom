FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SPECTRUMBLOOM_API_URL=http://127.0.0.1:8000 \
    LOCAL_DATASET_DIR=/tmp/spectrumbloom/datasets \
    SANDBOX_BACKEND=disabled

COPY --from=ghcr.io/astral-sh/uv:0.6.10 /uv /bin/

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY examples ./examples
COPY deploy/hf-spaces/start.sh /app/start.sh

RUN uv pip install --system --no-cache ".[eval]" && \
    chmod +x /app/start.sh && \
    mkdir -p /tmp/spectrumbloom/datasets && \
    useradd -m -u 1000 user && \
    chown -R user:user /app /tmp/spectrumbloom

USER user
EXPOSE 7860

CMD ["/app/start.sh"]
