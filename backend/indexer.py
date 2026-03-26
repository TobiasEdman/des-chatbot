"""
Content indexer for Digital Earth Sweden chatbot.

Crawls WordPress pages, STAC catalogs, and local Markdown files,
then chunks and stores them in Qdrant for vector search.
"""

import logging
import re
from pathlib import Path

import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DES_STAC_URL,
    DES_WORDPRESS_URL,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    QDRANT_URL,
)

logger = logging.getLogger(__name__)


class ContentIndexer:
    """Indexes content from various sources into Qdrant."""

    def __init__(self) -> None:
        logger.info("Initializing ContentIndexer...")
        self.qdrant = QdrantClient(url=QDRANT_URL)
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create the Qdrant collection if it does not exist."""
        collections = [
            c.name for c in self.qdrant.get_collections().collections
        ]
        if COLLECTION_NAME not in collections:
            logger.info("Creating Qdrant collection: %s", COLLECTION_NAME)
            self.qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIMENSION,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Collection created successfully.")
        else:
            logger.info("Collection %s already exists.", COLLECTION_NAME)

    @staticmethod
    def _strip_html(html: str) -> str:
        """Remove HTML tags and decode common entities."""
        text = re.sub(r"<[^>]+>", " ", html)
        text = (
            text.replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", '"')
            .replace("&#8217;", "'")
            .replace("&#8220;", '"')
            .replace("&#8221;", '"')
            .replace("&nbsp;", " ")
        )
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _chunk_text(
        text: str,
        chunk_size: int = CHUNK_SIZE,
        overlap: int = CHUNK_OVERLAP,
    ) -> list[str]:
        """
        Split text into overlapping chunks by approximate token count.

        Uses whitespace-based tokenization (1 token ~ 1 word) as an
        approximation for chunk sizing.
        """
        words = text.split()
        if not words:
            return []

        chunks: list[str] = []
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])
            if chunk.strip():
                chunks.append(chunk.strip())
            start += chunk_size - overlap

        return chunks

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings."""
        return self.embedding_model.encode(texts).tolist()

    def _upsert_chunks(
        self,
        chunks: list[str],
        source: str,
        metadata: dict | None = None,
    ) -> int:
        """
        Embed and upsert text chunks into Qdrant.

        Args:
            chunks: Text chunks to index.
            source: Source identifier (URL, file path, etc.).
            metadata: Additional metadata to attach to each point.

        Returns:
            Number of points upserted.
        """
        if not chunks:
            return 0

        if metadata is None:
            metadata = {}

        embeddings = self._embed_texts(chunks)

        # Generate deterministic IDs based on source and chunk index
        import hashlib

        points: list[PointStruct] = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = hashlib.md5(
                f"{source}::{i}::{chunk[:50]}".encode()
            ).hexdigest()
            # Qdrant expects UUID or unsigned int; use first 16 hex chars as int
            numeric_id = int(point_id[:16], 16)
            points.append(
                PointStruct(
                    id=numeric_id,
                    vector=embedding,
                    payload={
                        "text": chunk,
                        "source": source,
                        "chunk_index": i,
                        **metadata,
                    },
                )
            )

        # Upsert in batches of 100
        batch_size = 100
        for batch_start in range(0, len(points), batch_size):
            batch = points[batch_start : batch_start + batch_size]
            self.qdrant.upsert(
                collection_name=COLLECTION_NAME,
                points=batch,
            )

        logger.info(
            "Upserted %d chunks from source: %s", len(points), source
        )
        return len(points)

    def index_wordpress(
        self, url: str = DES_WORDPRESS_URL, max_pages: int = 100
    ) -> int:
        """
        Crawl a WordPress site via its REST API and index page/post content.

        Args:
            url: Base URL of the WordPress site.
            max_pages: Maximum number of pages to fetch per content type.

        Returns:
            Total number of chunks indexed.
        """
        total_chunks = 0
        wp_api = f"{url.rstrip('/')}/wp-json/wp/v2"

        for content_type in ("pages", "posts"):
            page_num = 1
            while page_num <= max_pages:
                try:
                    response = httpx.get(
                        f"{wp_api}/{content_type}",
                        params={
                            "per_page": 50,
                            "page": page_num,
                            "status": "publish",
                        },
                        timeout=30.0,
                    )
                    if response.status_code == 400:
                        # No more pages
                        break
                    response.raise_for_status()
                    items = response.json()
                except httpx.HTTPError:
                    logger.exception(
                        "Error fetching WordPress %s (page %d)",
                        content_type,
                        page_num,
                    )
                    break

                if not items:
                    break

                for item in items:
                    title = self._strip_html(
                        item.get("title", {}).get("rendered", "")
                    )
                    content = self._strip_html(
                        item.get("content", {}).get("rendered", "")
                    )
                    full_text = f"{title}\n\n{content}" if title else content

                    if not full_text.strip():
                        continue

                    item_url = item.get("link", f"{url}/{content_type}/{item.get('id', '')}")
                    chunks = self._chunk_text(full_text)
                    count = self._upsert_chunks(
                        chunks,
                        source=item_url,
                        metadata={
                            "type": "wordpress",
                            "content_type": content_type,
                            "title": title,
                        },
                    )
                    total_chunks += count

                page_num += 1

        logger.info(
            "WordPress indexing complete. Total chunks: %d", total_chunks
        )
        return total_chunks

    def index_stac(self, stac_url: str = DES_STAC_URL) -> int:
        """
        Index datasets from a STAC catalog.

        Args:
            stac_url: Base URL of the STAC catalog.

        Returns:
            Total number of chunks indexed.
        """
        total_chunks = 0

        try:
            # Fetch the root catalog
            response = httpx.get(stac_url, timeout=30.0)
            response.raise_for_status()
            catalog = response.json()
        except httpx.HTTPError:
            logger.exception("Error fetching STAC catalog at %s", stac_url)
            return 0

        # Find collection links
        child_links = [
            link
            for link in catalog.get("links", [])
            if link.get("rel") in ("child", "collection")
        ]

        # If this is a catalog with a /collections endpoint, try that too
        try:
            collections_resp = httpx.get(
                f"{stac_url.rstrip('/')}/collections",
                timeout=30.0,
            )
            if collections_resp.status_code == 200:
                collections_data = collections_resp.json()
                collections = collections_data.get("collections", [])
                for collection in collections:
                    total_chunks += self._index_stac_collection(
                        collection, stac_url
                    )
                if collections:
                    logger.info(
                        "STAC indexing complete. Total chunks: %d",
                        total_chunks,
                    )
                    return total_chunks
        except httpx.HTTPError:
            logger.debug("No /collections endpoint, falling back to links.")

        # Fall back to crawling child links
        for link in child_links:
            href = link.get("href", "")
            if not href:
                continue

            # Resolve relative URLs
            if href.startswith("/"):
                from urllib.parse import urlparse

                parsed = urlparse(stac_url)
                href = f"{parsed.scheme}://{parsed.netloc}{href}"
            elif not href.startswith("http"):
                href = f"{stac_url.rstrip('/')}/{href}"

            try:
                coll_resp = httpx.get(href, timeout=30.0)
                coll_resp.raise_for_status()
                collection = coll_resp.json()
                total_chunks += self._index_stac_collection(
                    collection, stac_url
                )
            except httpx.HTTPError:
                logger.warning("Failed to fetch STAC collection: %s", href)
                continue

        logger.info(
            "STAC indexing complete. Total chunks: %d", total_chunks
        )
        return total_chunks

    def _index_stac_collection(
        self, collection: dict, stac_url: str
    ) -> int:
        """Index a single STAC collection."""
        coll_id = collection.get("id", "unknown")
        title = collection.get("title", coll_id)
        description = collection.get("description", "")
        keywords = ", ".join(collection.get("keywords", []))
        license_info = collection.get("license", "")

        # Build spatial/temporal extent info
        extent_info = ""
        extent = collection.get("extent", {})
        spatial = extent.get("spatial", {})
        temporal = extent.get("temporal", {})
        if spatial.get("bbox"):
            bbox = spatial["bbox"][0]
            extent_info += f"Rumslig utstrackning (bbox): {bbox}\n"
        if temporal.get("interval"):
            interval = temporal["interval"][0]
            extent_info += f"Tidsutstrackning: {interval[0]} till {interval[1]}\n"

        # Compose document text
        parts = [f"Dataset: {title}"]
        if description:
            parts.append(f"Beskrivning: {description}")
        if keywords:
            parts.append(f"Nyckelord: {keywords}")
        if license_info:
            parts.append(f"Licens: {license_info}")
        if extent_info:
            parts.append(extent_info)

        # Include summaries if available
        summaries = collection.get("summaries", {})
        if summaries:
            summary_lines = []
            for key, value in summaries.items():
                if isinstance(value, list):
                    summary_lines.append(f"{key}: {', '.join(str(v) for v in value)}")
                elif isinstance(value, dict):
                    summary_lines.append(
                        f"{key}: {value.get('minimum', '')} - {value.get('maximum', '')}"
                    )
            if summary_lines:
                parts.append("Sammanfattning:\n" + "\n".join(summary_lines))

        full_text = "\n\n".join(parts)
        chunks = self._chunk_text(full_text)

        source_url = f"{stac_url.rstrip('/')}/collections/{coll_id}"
        return self._upsert_chunks(
            chunks,
            source=source_url,
            metadata={
                "type": "stac",
                "collection_id": coll_id,
                "title": title,
            },
        )

    def index_markdown(self, dir_path: str) -> int:
        """
        Index local Markdown files from a directory (recursively).

        Args:
            dir_path: Path to directory containing .md files.

        Returns:
            Total number of chunks indexed.
        """
        total_chunks = 0
        directory = Path(dir_path)

        if not directory.is_dir():
            logger.error("Directory does not exist: %s", dir_path)
            return 0

        md_files = list(directory.rglob("*.md"))
        logger.info("Found %d Markdown files in %s", len(md_files), dir_path)

        for md_file in md_files:
            try:
                content = md_file.read_text(encoding="utf-8")
            except OSError:
                logger.warning("Could not read file: %s", md_file)
                continue

            if not content.strip():
                continue

            # Extract title from first heading if present
            title = ""
            title_match = re.match(r"^#\s+(.+)$", content, re.MULTILINE)
            if title_match:
                title = title_match.group(1).strip()

            chunks = self._chunk_text(content)
            source = str(md_file.resolve())
            count = self._upsert_chunks(
                chunks,
                source=source,
                metadata={
                    "type": "markdown",
                    "title": title,
                    "filename": md_file.name,
                },
            )
            total_chunks += count

        logger.info(
            "Markdown indexing complete. Total chunks: %d", total_chunks
        )
        return total_chunks

    def index_all(self) -> dict[str, int]:
        """
        Run all indexers with default settings.

        Returns:
            Dict mapping source type to number of chunks indexed.
        """
        results: dict[str, int] = {}

        logger.info("Starting full indexing run...")

        results["wordpress"] = self.index_wordpress()
        results["stac"] = self.index_stac()

        logger.info("Full indexing complete: %s", results)
        return results


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    indexer = ContentIndexer()

    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "wordpress":
            url = sys.argv[2] if len(sys.argv) > 2 else DES_WORDPRESS_URL
            indexer.index_wordpress(url)
        elif command == "stac":
            url = sys.argv[2] if len(sys.argv) > 2 else DES_STAC_URL
            indexer.index_stac(url)
        elif command == "markdown":
            if len(sys.argv) < 3:
                print("Usage: python indexer.py markdown <directory>")
                sys.exit(1)
            indexer.index_markdown(sys.argv[2])
        elif command == "all":
            indexer.index_all()
        else:
            print(f"Unknown command: {command}")
            print("Usage: python indexer.py [wordpress|stac|markdown|all]")
            sys.exit(1)
    else:
        indexer.index_all()
