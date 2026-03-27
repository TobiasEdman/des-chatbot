"""
FastAPI application for the Digital Earth Sweden chatbot.

Provides a streaming chat endpoint backed by a RAG pipeline with
Qdrant vector search and vLLM language model.
"""

import json
import logging
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config import ALLOWED_ORIGINS, LOG_LEVEL, MAX_HISTORY_MESSAGES, RATE_LIMIT_PER_MINUTE
from rag import rag_pipeline

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory stores
# ---------------------------------------------------------------------------
# Session conversation history: session_id -> list of {role, content}
session_store: dict[str, list[dict[str, str]]] = {}

# Rate limiting: ip -> list of request timestamps
rate_limit_store: dict[str, list[float]] = defaultdict(list)

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup / shutdown lifecycle."""
    logger.info("Digital Earth Sweden Chatbot starting up...")
    # Eagerly load the embedding model at startup
    try:
        _ = rag_pipeline.embedding_model
        logger.info("Embedding model ready.")
    except Exception:
        logger.exception("Failed to load embedding model at startup")
    yield
    logger.info("Digital Earth Sweden Chatbot shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Digital Earth Sweden Chatbot API",
    description="RAG-powered chatbot for Digital Earth Sweden",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Rate limiting helper
# ---------------------------------------------------------------------------
def _check_rate_limit(client_ip: str) -> None:
    """
    Enforce per-IP rate limiting.

    Raises HTTPException 429 if the client exceeds RATE_LIMIT_PER_MINUTE
    requests within the last 60 seconds.
    """
    now = time.time()
    window_start = now - 60.0

    # Prune old entries
    timestamps = rate_limit_store[client_ip]
    rate_limit_store[client_ip] = [t for t in timestamps if t > window_start]

    if len(rate_limit_store[client_ip]) >= RATE_LIMIT_PER_MINUTE:
        logger.warning("Rate limit exceeded for IP: %s", client_ip)
        raise HTTPException(
            status_code=429,
            detail="For manga forfragan. Forsok igen om en minut.",
        )

    rate_limit_store[client_ip].append(now)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    """Incoming chat request."""

    message: str = Field(
        ..., min_length=1, max_length=2000, description="User message"
    )
    session_id: str | None = Field(
        default=None,
        description="Session ID for conversation continuity. "
        "A new one is generated if omitted.",
    )


class ChatMetadata(BaseModel):
    """Metadata returned alongside the streamed response."""

    session_id: str
    sources: list[dict] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/api/chat")
async def chat(request: Request, body: ChatRequest) -> StreamingResponse:
    """
    Chat endpoint with Server-Sent Events (SSE) streaming.

    Accepts a user message and optional session_id. Returns a streaming
    response where each SSE event contains a text chunk from the LLM.
    """
    # Rate limiting
    client_ip = request.client.host if request.client else "unknown"
    _check_rate_limit(client_ip)

    # Session management
    session_id = body.session_id or str(uuid.uuid4())
    history = session_store.get(session_id, [])

    logger.info(
        "Chat request from %s | session=%s | message_length=%d",
        client_ip,
        session_id,
        len(body.message),
    )

    async def event_stream() -> AsyncGenerator[str, None]:
        """Generate SSE events from the RAG pipeline."""
        full_response: list[str] = []

        # Send session metadata as the first event
        meta = ChatMetadata(session_id=session_id)
        yield f"event: metadata\ndata: {meta.model_dump_json()}\n\n"

        try:
            async for chunk in rag_pipeline.query(body.message, history):
                full_response.append(chunk)
                # SSE format: each event is "data: <payload>\n\n"
                payload = json.dumps({"text": chunk}, ensure_ascii=False)
                yield f"data: {payload}\n\n"
        except Exception:
            logger.exception("Error during response generation")
            error_msg = (
                "Ett fel uppstod vid generering av svar. Forsok igen."
            )
            payload = json.dumps({"text": error_msg}, ensure_ascii=False)
            yield f"data: {payload}\n\n"
            full_response.append(error_msg)

        # Signal stream end — include sources used for this response
        sources = rag_pipeline.last_sources
        done_payload = json.dumps(
            {"sources": sources}, ensure_ascii=False
        )
        yield f"event: done\ndata: {done_payload}\n\n"

        # Update session history
        assistant_text = "".join(full_response)
        history.append({"role": "user", "content": body.message})
        history.append({"role": "assistant", "content": assistant_text})

        # Keep only the last N messages (N = MAX_HISTORY_MESSAGES * 2
        # because each turn has user + assistant)
        max_entries = MAX_HISTORY_MESSAGES * 2
        session_store[session_id] = history[-max_entries:]

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
