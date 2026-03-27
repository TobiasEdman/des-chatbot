"""
RAG pipeline for Digital Earth Sweden chatbot.

Handles vector retrieval from Qdrant and response generation via vLLM.
"""

import json
import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint
from sentence_transformers import SentenceTransformer

from config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    MODEL_NAME,
    QDRANT_URL,
    RETRIEVAL_SCORE_THRESHOLD,
    RETRIEVAL_TOP_K,
    VLLM_URL,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt — strict, fact-based, no hallucination
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "Du är DES Chatbot, en domänexpert inom Digital Earth Sweden (DES).\n"
    "Du svarar på svenska om inte användaren skriver på ett annat språk.\n\n"
    "Du svarar enbart på frågor som rör Digital Earth Swedens datamängder, "
    "tjänster, openEO API, STAC, tutorials, guider, satellitdata, "
    "fjärranalys och jordobservation — baserat på tillhandahållen kontext "
    "och källor.\n\n"
    "REGLER:\n"
    "- Ge faktabaserade, kortfattade och seriösa svar.\n"
    "- Ställ INGA uppföljningsfrågor och be INTE om förtydliganden.\n"
    "- Förklara eller diskutera INTE saker som inte explicit efterfrågats.\n"
    "- Spekulera eller gissa INTE om något saknas i kontexten.\n"
    "- Inkludera INTE information från externa eller ej angivna källor.\n"
    "- Använd INTE sociala medier, forum eller generella AI-kunskaper "
    "som källa.\n"
    "- Om kontexten inte innehåller tillräcklig information, svara: "
    '"Jag har ingen information om detta i tillgänglig kontext."\n'
    "- Ange alltid källa till svaret där det är möjligt, i formatet: "
    '"Källa: [länk]"\n'
    "- Avsluta svaret efter att frågan är besvarad. Lägg INTE till "
    "ytterligare förklaringar, sammanfattningar eller förslag.\n"
)

NO_CONTEXT_RESPONSE = "Jag har ingen information om detta i tillgänglig kontext."


@dataclass
class RetrievedChunk:
    """A chunk retrieved from the vector store."""

    text: str
    source: str
    score: float
    metadata: dict = field(default_factory=dict)


@dataclass
class RAGResult:
    """Result from the RAG pipeline including sources used."""

    sources: list[dict]


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline using Qdrant and vLLM."""

    def __init__(self) -> None:
        logger.info("Initializing RAG pipeline...")
        self._embedding_model: SentenceTransformer | None = None
        self._qdrant: QdrantClient | None = None

    @property
    def embedding_model(self) -> SentenceTransformer:
        """Lazy-load the embedding model."""
        if self._embedding_model is None:
            logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
            self._embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info("Embedding model loaded successfully.")
        return self._embedding_model

    @property
    def qdrant(self) -> QdrantClient:
        """Lazy-load the Qdrant client."""
        if self._qdrant is None:
            logger.info("Connecting to Qdrant at %s", QDRANT_URL)
            self._qdrant = QdrantClient(url=QDRANT_URL)
        return self._qdrant

    def embed_query(self, text: str) -> list[float]:
        """Encode a text query into an embedding vector."""
        return self.embedding_model.encode(text).tolist()

    def retrieve(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
        score_threshold: float = RETRIEVAL_SCORE_THRESHOLD,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the most relevant document chunks for a given query.

        Applies score filtering to remove low-relevance results.

        Args:
            query: The user's search query.
            top_k: Number of top results to fetch from Qdrant.
            score_threshold: Minimum cosine similarity score to keep.

        Returns:
            List of RetrievedChunk objects above the score threshold.
        """
        try:
            query_vector = self.embed_query(query)
            results: list[ScoredPoint] = self.qdrant.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                limit=top_k,
            )
            chunks = []
            for point in results:
                if point.score < score_threshold:
                    logger.debug(
                        "Skipping chunk (score=%.3f < threshold=%.3f): %s",
                        point.score,
                        score_threshold,
                        (point.payload or {}).get("source", "?"),
                    )
                    continue
                payload = point.payload or {}
                chunks.append(
                    RetrievedChunk(
                        text=payload.get("text", ""),
                        source=payload.get("source", "unknown"),
                        score=point.score,
                        metadata={
                            k: v
                            for k, v in payload.items()
                            if k not in ("text", "source")
                        },
                    )
                )
            logger.info(
                "Retrieved %d chunks for query (top score: %.3f, "
                "filtered from %d, threshold=%.2f)",
                len(chunks),
                chunks[0].score if chunks else 0.0,
                len(results),
                score_threshold,
            )
            return chunks
        except Exception:
            logger.exception("Error retrieving chunks from Qdrant")
            return []

    def _build_messages(
        self,
        query: str,
        context_chunks: list[RetrievedChunk],
        history: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """
        Build the message list for the LLM.

        Mistral v0.2 does not support a native system role, so the system
        prompt and context are embedded in the first user message using
        the [INST] template implicitly via the chat format.
        """
        # Build context block with numbered sources for citation
        if context_chunks:
            context_parts = []
            for i, chunk in enumerate(context_chunks, 1):
                title = chunk.metadata.get("title", "")
                source_label = title if title else chunk.source
                context_parts.append(
                    f"[{i}] Källa: {chunk.source}\n"
                    f"    Titel: {source_label}\n"
                    f"    Relevans: {chunk.score:.2f}\n"
                    f"    {chunk.text}"
                )
            context_text = "\n\n---\n\n".join(context_parts)
        else:
            context_text = ""

        # Compose the full system + context block
        system_content = SYSTEM_PROMPT
        if context_text:
            system_content += (
                "\n--- KONTEXT ---\n"
                "Använd ENBART följande information för att svara:\n\n"
                f"{context_text}\n\n"
                "--- SLUT KONTEXT ---\n"
            )
        else:
            system_content += (
                "\n--- KONTEXT ---\n"
                "Ingen relevant kontext hittades.\n"
                "--- SLUT KONTEXT ---\n"
            )

        # Mistral v0.2: embed system in first user message
        # The model expects alternating user/assistant turns.
        messages: list[dict[str, str]] = []

        # Add conversation history
        for msg in history:
            messages.append(msg)

        # Current query with system prompt baked in
        user_content = f"{system_content}\n\nAnvändarens fråga: {query}"
        messages.append({"role": "user", "content": user_content})

        return messages

    async def generate_response(
        self,
        query: str,
        context_chunks: list[RetrievedChunk],
        history: list[dict[str, str]],
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response from vLLM given query, context, and history.

        If no context chunks passed the score filter, yields a no-context
        response immediately without calling the LLM.

        Yields text chunks as they arrive from the LLM.
        """
        # If no relevant context, don't waste GPU — answer directly
        if not context_chunks:
            logger.info("No context chunks above threshold — returning no-context response")
            yield NO_CONTEXT_RESPONSE
            return

        messages = self._build_messages(query, context_chunks, history)
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "max_tokens": LLM_MAX_TOKENS,
            "temperature": LLM_TEMPERATURE,
            "stream": True,
            # Encourage shorter, focused answers
            "stop": ["\n\nFråga:", "\n\nUser:", "\n\nAnvändare:"],
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                async with client.stream(
                    "POST",
                    f"{VLLM_URL}/chat/completions",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[len("data: "):]
                        if data.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            delta = (
                                chunk.get("choices", [{}])[0]
                                .get("delta", {})
                                .get("content", "")
                            )
                            if delta:
                                yield delta
                        except (ValueError, IndexError, KeyError):
                            logger.warning(
                                "Failed to parse streaming chunk: %s", data
                            )
                            continue
            except httpx.HTTPStatusError as exc:
                logger.error(
                    "vLLM returned HTTP %d: %s",
                    exc.response.status_code,
                    exc.response.text[:500],
                )
                yield (
                    "Jag kunde inte generera ett svar just nu. "
                    "Försök igen om en stund."
                )
            except httpx.RequestError as exc:
                logger.error("Failed to connect to vLLM: %s", exc)
                yield (
                    "Jag kan inte nå språkmodellen just nu. "
                    "Försök igen senare."
                )

    async def query(
        self,
        user_query: str,
        history: list[dict[str, str]] | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Full RAG pipeline: retrieve context and generate a streaming response.

        After the LLM response, appends source references.

        Args:
            user_query: The user's question.
            history: Previous conversation messages.

        Yields:
            Text chunks of the generated response, followed by source refs.
        """
        if history is None:
            history = []

        chunks = self.retrieve(user_query)

        # Store sources for post-response citation
        self._last_sources = [
            {
                "url": c.source,
                "title": c.metadata.get("title", ""),
                "score": round(c.score, 3),
            }
            for c in chunks
        ]

        # Stream the LLM response
        response_parts: list[str] = []
        async for token in self.generate_response(user_query, chunks, history):
            response_parts.append(token)
            yield token

        # Append source citations if we had context and the response
        # is not the no-context fallback
        full_response = "".join(response_parts)
        if chunks and full_response != NO_CONTEXT_RESPONSE:
            # Only add sources if the response doesn't already contain them
            if "Källa:" not in full_response:
                yield "\n\n---\n**Källor:**\n"
                seen_sources: set[str] = set()
                for chunk in chunks:
                    if chunk.source in seen_sources:
                        continue
                    seen_sources.add(chunk.source)
                    title = chunk.metadata.get("title", chunk.source)
                    yield f"- [{title}]({chunk.source})\n"

    @property
    def last_sources(self) -> list[dict]:
        """Return the sources used in the most recent query."""
        return getattr(self, "_last_sources", [])


# Module-level singleton
rag_pipeline = RAGPipeline()
