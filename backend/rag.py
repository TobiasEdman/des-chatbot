"""
RAG pipeline for Digital Earth Sweden chatbot.

Handles vector retrieval from Qdrant and response generation via vLLM.
"""

import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass

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
    VLLM_URL,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Du ar en hjalpsam AI-assistent for Digital Earth Sweden (DES). "
    "Du svarar pa svenska om inte anvandaren skriver pa ett annat sprak. "
    "Ditt expertomrade ar:\n"
    "- Digital Earth Swedens datamangder och tjanster\n"
    "- openEO API for geospatial databearbetning\n"
    "- STAC (SpatioTemporal Asset Catalog) for att hitta och komma at geodata\n"
    "- Tutorials och guider for DES-plattformen\n"
    "- Satellitdata, fjarranalys och jordobservation\n\n"
    "Svara BARA pa fragor som ar relaterade till Digital Earth Sweden, "
    "geodata, fjarranalys, openEO, STAC eller liknande amnen. "
    "Om anvandaren fragar om nagot orelaterat, forklara artigt att du "
    "bara kan hjalpa till med fragor om Digital Earth Sweden och dess tjanster.\n\n"
    "Basera dina svar pa den kontext som ges. Om kontexten inte innehaller "
    "tillracklig information, saga det istallet for att hitta pa svar."
)


@dataclass
class RetrievedChunk:
    """A chunk retrieved from the vector store."""

    text: str
    source: str
    score: float
    metadata: dict


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

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """
        Retrieve the most relevant document chunks for a given query.

        Args:
            query: The user's search query.
            top_k: Number of top results to return.

        Returns:
            List of RetrievedChunk objects sorted by relevance.
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
                "Retrieved %d chunks for query (top score: %.3f)",
                len(chunks),
                chunks[0].score if chunks else 0.0,
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
        Build the message list for the LLM, including system prompt,
        retrieved context, conversation history, and the current query.
        """
        context_text = "\n\n---\n\n".join(
            f"[Kalla: {chunk.source}]\n{chunk.text}" for chunk in context_chunks
        )

        system_content = SYSTEM_PROMPT
        if context_text:
            system_content += (
                "\n\n--- KONTEXT ---\n"
                "Anvand foljande information for att svara pa fragor:\n\n"
                f"{context_text}"
            )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_content}
        ]
        messages.extend(history)
        messages.append({"role": "user", "content": query})
        return messages

    async def generate_response(
        self,
        query: str,
        context_chunks: list[RetrievedChunk],
        history: list[dict[str, str]],
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response from vLLM given query, context, and history.

        Yields text chunks as they arrive from the LLM.
        """
        messages = self._build_messages(query, context_chunks, history)
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "max_tokens": LLM_MAX_TOKENS,
            "temperature": LLM_TEMPERATURE,
            "stream": True,
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
                            import json

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
                    "Forsok igen om en stund."
                )
            except httpx.RequestError as exc:
                logger.error("Failed to connect to vLLM: %s", exc)
                yield (
                    "Jag kan inte na sprakmodellen just nu. "
                    "Forsok igen senare."
                )

    async def query(
        self,
        user_query: str,
        history: list[dict[str, str]] | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Full RAG pipeline: retrieve context and generate a streaming response.

        Args:
            user_query: The user's question.
            history: Previous conversation messages.

        Yields:
            Text chunks of the generated response.
        """
        if history is None:
            history = []

        chunks = self.retrieve(user_query)
        async for token in self.generate_response(user_query, chunks, history):
            yield token


# Module-level singleton
rag_pipeline = RAGPipeline()
