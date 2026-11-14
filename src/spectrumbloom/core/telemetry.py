"""OpenTelemetry tracing/metrics and optional Langfuse observability."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from urllib.parse import urlparse

from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from spectrumbloom.core.config import Settings

logger = logging.getLogger(__name__)


class TelemetryRuntime:
    """Manages OpenTelemetry tracer/meter and optional Langfuse callbacks."""

    def __init__(self, settings: Settings, in_memory_export: bool = False) -> None:
        self.settings = settings
        self.in_memory_exporter: InMemorySpanExporter | None = None
        self.tracer_provider: TracerProvider | None = None
        self.meter_provider: MeterProvider | None = None
        self.langfuse_enabled: bool = False
        self.langfuse_client: Any = None
        self.langfuse_handler_cls: Any = None

        resource = Resource.create(
            {
                "service.name": settings.otel_service_name,
                "service.version": "0.1.0",
                "deployment.environment": settings.app_env,
            }
        )

        self.tracer_provider = TracerProvider(resource=resource)
        if in_memory_export or settings.app_env == "test":
            self.in_memory_exporter = InMemorySpanExporter()
            self.tracer_provider.add_span_processor(SimpleSpanProcessor(self.in_memory_exporter))

        if settings.otel_console_export:
            self.tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

        otlp_endpoint = (settings.otel_exporter_otlp_endpoint or "").strip()
        if otlp_endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                    OTLPSpanExporter,
                )

                exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
                self.tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
            except Exception:
                logger.exception("Failed to configure OTLP span exporter.")

        current_tp = trace.get_tracer_provider()
        if not isinstance(current_tp, TracerProvider):
            trace.set_tracer_provider(self.tracer_provider)
        self.tracer = self.tracer_provider.get_tracer("spectrumbloom")

        # Configure OpenTelemetry Metrics
        metric_readers = []
        if otlp_endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
                    OTLPMetricExporter,
                )

                metric_readers.append(
                    PeriodicExportingMetricReader(OTLPMetricExporter(endpoint=otlp_endpoint))
                )
            except Exception:
                logger.exception("Failed to configure OTLP metric exporter.")

        self.meter_provider = MeterProvider(resource=resource, metric_readers=metric_readers)
        current_mp = metrics.get_meter_provider()
        if not isinstance(current_mp, MeterProvider):
            metrics.set_meter_provider(self.meter_provider)
        meter = self.meter_provider.get_meter("spectrumbloom")

        self.query_counter = meter.create_counter(
            "spectrumbloom.agent.queries",
            description="Total agentic EDA queries processed",
        )
        self.query_latency_ms = meter.create_histogram(
            "spectrumbloom.agent.query_duration_ms",
            unit="ms",
            description="End-to-end agent query latency",
        )
        self.sandbox_counter = meter.create_counter(
            "spectrumbloom.sandbox.executions",
            description="Total sandboxed Python executions",
        )
        self.sandbox_duration_ms = meter.create_histogram(
            "spectrumbloom.sandbox.duration_ms",
            unit="ms",
            description="Sandboxed Python execution latency",
        )
        self.ml_counter = meter.create_counter(
            "spectrumbloom.ml.experiments",
            description="Total ML baseline experiments executed",
        )

        self._init_langfuse()

    def _init_langfuse(self) -> None:
        public_key = (self.settings.langfuse_public_key or "").strip()
        secret_key = (self.settings.langfuse_secret_key or "").strip()
        base_url = (self.settings.langfuse_base_url or "").strip()

        if not public_key or not secret_key:
            return

        if base_url:
            parsed = urlparse(base_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                logger.warning("Ignoring invalid LANGFUSE_BASE_URL: %s", base_url)
                return
            os.environ["LANGFUSE_BASE_URL"] = base_url
            os.environ["LANGFUSE_HOST"] = base_url

        os.environ["LANGFUSE_PUBLIC_KEY"] = public_key
        os.environ["LANGFUSE_SECRET_KEY"] = secret_key

        try:
            from langfuse import Langfuse
            from langfuse.langchain import CallbackHandler

            self._langfuse_public_key = public_key
            self.langfuse_client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                base_url=base_url or None,
                tracer_provider=self.tracer_provider,
            )
            self.langfuse_handler_cls = CallbackHandler
            self.langfuse_enabled = True
        except Exception:
            logger.exception("Langfuse initialization failed; continuing without Langfuse.")
            self.langfuse_enabled = False

    def create_langfuse_callback(self) -> Any | None:
        if not self.langfuse_enabled or self.langfuse_handler_cls is None:
            return None
        try:
            return self.langfuse_handler_cls(public_key=self._langfuse_public_key)
        except Exception:
            logger.exception("Failed to create Langfuse CallbackHandler.")
            return None

    def flush(self) -> None:
        if self.langfuse_client is not None:
            try:
                self.langfuse_client.flush()
            except Exception:
                logger.debug("Langfuse flush failed", exc_info=True)

    @contextmanager
    def span(self, name: str, attributes: dict[str, Any] | None = None) -> Iterator[Any]:
        with self.tracer.start_as_current_span(name) as current_span:
            if attributes:
                for key, value in attributes.items():
                    if value is not None:
                        current_span.set_attribute(key, value)
            yield current_span

    def instrument_fastapi(self, app: Any) -> None:
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

            FastAPIInstrumentor.instrument_app(app, tracer_provider=self.tracer_provider)
        except Exception:
            logger.exception("Failed to instrument FastAPI with OpenTelemetry.")

    def shutdown(self) -> None:
        self.flush()
        if self.tracer_provider is not None:
            self.tracer_provider.shutdown()
        if self.meter_provider is not None:
            self.meter_provider.shutdown()
