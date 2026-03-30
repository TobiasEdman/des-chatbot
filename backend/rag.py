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
    "Du är DES Chatbot för Digital Earth Sweden.\n"
    "Svara ALLTID på svenska. ALDRIG på engelska.\n"
    "Svara kort (max 3-4 meningar). Inkludera INTE engelska översättningar.\n\n"
    "OM DIGITAL EARTH SWEDEN:\n"
    "Digital Earth Sweden (DES) är en nationell plattform för "
    "jordobservationsdata, utvecklad av RISE och Rymdstyrelsen. "
    "DES tillhandahåller:\n"
    "- Analysredo Sentinel-2 satellitdata (optisk) för hela Sverige\n"
    "- openEO API för programmatisk dataåtkomst och processering\n"
    "- STAC-katalog för att söka och ladda ner geodata\n"
    "- Karttjänst för visualisering i webbläsaren\n"
    "- IMINT Engine för bildanalys (marktäcke, vegetation, kustlinje)\n\n"
    "DES fokuserar på optisk data (Sentinel-2), inte SAR eller väder.\n"
    "Men du kan svara på forskningsfrågor om alla EO-ämnen "
    "(SAR, fjärranalys, klimat etc.) baserat på kontexten.\n\n"
    "REGLER:\n"
    "- Basera svaret ENBART på kontexten nedan.\n"
    "- Ställ INGA uppföljningsfrågor.\n"
    "- Spekulera INTE. Hitta INTE på information.\n"
    "- Ange INGA källhänvisningar i svaret.\n"
    "- Om kontexten saknar svar: \"Jag har ingen information om detta.\"\n"
    "- Vid hälsningar (hej, hallå, god dag): svara vänligt och berätta "
    "kort vad du kan hjälpa till med.\n"
)

GREETING_RESPONSE = (
    "Hej! Jag är DES Chatbot och kan svara på frågor om "
    "Digital Earth Sweden, satellitdata, openEO, STAC och fjärranalys. "
    "Vad vill du veta?"
)

GREETING_WORDS = {"hej", "hallå", "hello", "hi", "tjena", "god dag", "hejsan", "tja"}

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

    # Keywords that indicate a research/science question
    _RESEARCH_KEYWORDS = {
        "forskning", "forskare", "studie", "artikel", "publikation",
        "abstract", "vetenskap", "akademi", "universitet",
        "research", "study", "paper", "scientist",
    }

    def _is_research_query(self, query: str) -> bool:
        """Detect if user is asking about research/publications."""
        q_lower = query.lower()
        return any(kw in q_lower for kw in self._RESEARCH_KEYWORDS)

    def retrieve(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
        score_threshold: float = RETRIEVAL_SCORE_THRESHOLD,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the most relevant document chunks for a given query.

        For research questions, fetches more results and prioritizes
        publication chunks over WordPress content.

        Args:
            query: The user's search query.
            top_k: Number of top results to fetch from Qdrant.
            score_threshold: Minimum cosine similarity score to keep.

        Returns:
            List of RetrievedChunk objects above the score threshold.
        """
        is_research = self._is_research_query(query)
        fetch_k = top_k * 3 if is_research else top_k

        try:
            query_vector = self.embed_query(query)
            results: list[ScoredPoint] = self.qdrant.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                limit=fetch_k,
            )
            chunks = []
            for point in results:
                if point.score < score_threshold:
                    continue
                payload = point.payload or {}
                chunk_type = payload.get("type", "")
                # Boost publication scores for research queries
                effective_score = point.score
                if is_research and chunk_type == "publication":
                    effective_score += 0.15  # Strong boost for publications
                elif is_research and chunk_type != "publication":
                    effective_score -= 0.05  # Demote non-publications
                chunks.append(
                    RetrievedChunk(
                        text=payload.get("text", ""),
                        source=payload.get("source", "unknown"),
                        score=effective_score,
                        metadata={
                            k: v
                            for k, v in payload.items()
                            if k not in ("text", "source")
                        },
                    )
                )

            # Re-sort by effective score and take top_k
            chunks.sort(key=lambda c: c.score, reverse=True)
            chunks = chunks[:top_k]
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

        # Use non-streaming vLLM call (httpx 0.28 streaming + vLLM has
        # compatibility issues). The full response is returned at once and
        # then yielded to the SSE stream in main.py.
        payload["stream"] = False

        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                response = await client.post(
                    f"{VLLM_URL}/chat/completions",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                data = response.json()
                content = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                if content:
                    yield content
                else:
                    logger.warning("vLLM returned empty content")
                    yield NO_CONTEXT_RESPONSE
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

        # Handle greetings without LLM call
        query_lower = user_query.strip().lower().rstrip("!?.,")
        if query_lower in GREETING_WORDS:
            self._last_sources = []
            yield GREETING_RESPONSE
            return

        chunks = self.retrieve(user_query)

        # GraphRAG: add Neo4j results for research queries
        try:
            from graph_rag import classify_query, query_graph
            query_type = classify_query(user_query)
            if query_type == "research":
                graph_results = query_graph(user_query)
                for gq in graph_results:
                    context_str = gq.to_context_string()
                    if context_str:
                        # Insert graph data as a high-priority chunk
                        chunks.insert(0, RetrievedChunk(
                            text=context_str,
                            source="knowledge_graph",
                            score=1.0,  # Highest priority
                            metadata={"type": "graph", "query_type": gq.query_type},
                        ))
                logger.info("GraphRAG: %d graph results for query type '%s'",
                           len(graph_results), query_type)
        except Exception as e:
            logger.warning("GraphRAG unavailable: %s", e)

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

        # Source citations removed — source URLs from indexer are not
        # user-facing (e.g. "wordpress" instead of actual URLs).
        # Sources are still available in the SSE done event for debugging.

    @property
    def last_sources(self) -> list[dict]:
        """Return the sources used in the most recent query."""
        return getattr(self, "_last_sources", [])


# Module-level singleton
rag_pipeline = RAGPipeline()
