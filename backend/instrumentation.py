"""OpenTelemetry instrumentation for des-chatbot.

Per agentic_workflow rollout plan W2.2. Fail-open:
 - No opentelemetry packages installed → every helper is a no-op.
 - No OTEL_EXPORTER_OTLP_ENDPOINT set → default providers (also no-op).

Metrics exposed when enabled:
 - des_chatbot.chat.duration_ms (histogram) — end-to-end /api/chat latency
 - des_chatbot.chat.requests_total (counter) — request count per status
 - des_chatbot.rag.query_duration_ms (histogram) — RAG pipeline duration
 - des_chatbot.rag.retrieval_results (histogram) — results per query
 - des_chatbot.llm.tokens_total (counter) — streaming tokens emitted

Environment variables (read by the OTel SDK):
    OTEL_EXPORTER_OTLP_ENDPOINT   — e.g. http://otel-collector:4317
    OTEL_SERVICE_NAME             — defaults to "des-chatbot"
"""
from __future__ import annotations

import contextlib
import os
from typing import Any, Iterator

_OTEL_AVAILABLE = False
_tracer = None
_meter = None
_chat_histogram = None
_chat_counter = None
_rag_histogram = None
_retrieval_histogram = None
_tokens_counter = None

try:
    from opentelemetry import metrics, trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
                OTLPMetricExporter,
            )

            service_name = os.environ.get("OTEL_SERVICE_NAME", "des-chatbot")
            resource = Resource.create({"service.name": service_name})

            trace_provider = TracerProvider(resource=resource)
            trace_provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter())
            )
            trace.set_tracer_provider(trace_provider)

            meter_provider = MeterProvider(
                resource=resource,
                metric_readers=[
                    PeriodicExportingMetricReader(OTLPMetricExporter())
                ],
            )
            metrics.set_meter_provider(meter_provider)

            _OTEL_AVAILABLE = True
        except ImportError:
            pass

    _tracer = trace.get_tracer(__name__)
    _meter = metrics.get_meter(__name__)

    _chat_histogram = _meter.create_histogram(
        name="des_chatbot.chat.duration_ms",
        description="End-to-end /api/chat request duration",
        unit="ms",
    )
    _chat_counter = _meter.create_counter(
        name="des_chatbot.chat.requests_total",
        description="Cumulative count of /api/chat requests",
    )
    _rag_histogram = _meter.create_histogram(
        name="des_chatbot.rag.query_duration_ms",
        description="RAG pipeline query duration",
        unit="ms",
    )
    _retrieval_histogram = _meter.create_histogram(
        name="des_chatbot.rag.retrieval_results",
        description="Number of retrieval hits per query",
    )
    _tokens_counter = _meter.create_counter(
        name="des_chatbot.llm.tokens_total",
        description="Cumulative count of streaming tokens emitted",
    )
except ImportError:
    pass


def is_enabled() -> bool:
    """True when OTel is both installed and configured with an endpoint."""
    return _OTEL_AVAILABLE


@contextlib.contextmanager
def span(name: str, **attrs: Any) -> Iterator[None]:
    """Start a trace span. No-op when OTel isn't wired up."""
    if _tracer is None:
        yield
        return
    with _tracer.start_as_current_span(name) as s:
        for k, v in attrs.items():
            try:
                if isinstance(v, (str, int, float, bool)):
                    s.set_attribute(k, v)
                else:
                    s.set_attribute(k, str(v))
            except Exception:
                pass
        yield


def record_chat_ms(ms: float, status: str = "success") -> None:
    """Record one /api/chat request's end-to-end duration."""
    if _chat_histogram is not None:
        try:
            _chat_histogram.record(ms, attributes={"status": status})
        except Exception:
            pass
    if _chat_counter is not None:
        try:
            _chat_counter.add(1, attributes={"status": status})
        except Exception:
            pass


def record_rag_ms(ms: float, **labels: str) -> None:
    """Record one RAG pipeline query duration."""
    if _rag_histogram is None:
        return
    try:
        _rag_histogram.record(ms, attributes=labels or None)
    except Exception:
        pass


def record_retrieval_hits(count: int, **labels: str) -> None:
    """Record how many retrieval hits a query returned."""
    if _retrieval_histogram is None:
        return
    try:
        _retrieval_histogram.record(count, attributes=labels or None)
    except Exception:
        pass


def record_tokens(n: int, **labels: str) -> None:
    """Increment the streaming-tokens counter."""
    if _tokens_counter is None:
        return
    try:
        _tokens_counter.add(n, attributes=labels or None)
    except Exception:
        pass
