"""
GraphRAG pipeline for DES chatbot.

Combines Neo4j graph queries with Qdrant vector search for
hybrid retrieval-augmented generation.

Query classification:
  - Platform questions → Qdrant vector search
  - Research questions → Neo4j Cypher + Qdrant vector search
  - Greeting → direct response (no retrieval)
"""

import logging
from typing import Optional

from neo4j import GraphDatabase

from config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    QDRANT_URL,
    RETRIEVAL_SCORE_THRESHOLD,
    RETRIEVAL_TOP_K,
)
from cypher_templates import (
    RESEARCHERS_BY_THEME,
    RESEARCHERS_BY_INSTITUTION,
    TOP_RESEARCHERS_BY_DOMAIN,
    PUBLICATIONS_BY_THEME,
    INSTITUTIONS_BY_DOMAIN,
    DOMAIN_KEYWORDS,
    THEME_KEYWORDS,
)

logger = logging.getLogger(__name__)

# Neo4j config (required env vars — set via K8s Secret in production,
# via .env for local dev). No fallback passwords in source — see
# agentic_workflow rollout plan W1.2.
import os
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://kg-neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
if not NEO4J_PASSWORD:
    raise RuntimeError(
        "NEO4J_PASSWORD environment variable is required. "
        "Set it via K8s Secret (production) or .env file (local dev). "
        "Never hardcode — rotation pending per agentic_workflow rollout W1.2."
    )


class GraphQuery:
    """Result from a Neo4j graph query."""

    def __init__(self, query_type: str, results: list[dict], description: str = ""):
        self.query_type = query_type
        self.results = results
        self.description = description

    def to_context_string(self) -> str:
        """Format graph results as context for the LLM."""
        if not self.results:
            return ""

        lines = [f"--- KUNSKAPSGRAF ({self.description}) ---"]
        for r in self.results:
            parts = []
            if "name" in r:
                parts.append(r["name"])
            if "h_index" in r and r["h_index"]:
                parts.append(f"h-index: {r['h_index']}")
            if "institution" in r and r["institution"]:
                parts.append(r["institution"])
            if "publications" in r:
                parts.append(f"{r['publications']} publikationer")
            if "title" in r:
                parts.append(r["title"])
            if "year" in r:
                parts.append(f"({r['year']})")
            if "themes" in r and r["themes"]:
                parts.append(f"teman: {', '.join(r['themes'])}")
            if "doi" in r and r["doi"]:
                parts.append(f"DOI: {r['doi']}")
            lines.append("  - " + " | ".join(str(p) for p in parts))

        return "\n".join(lines)


def _detect_domain(query: str) -> Optional[str]:
    """Detect research domain from query keywords."""
    q = query.lower()
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return domain
    return None


def _detect_theme(query: str) -> Optional[str]:
    """Detect specific theme from query keywords."""
    q = query.lower()
    for theme, keywords in THEME_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return theme
    return None


def _detect_institution(query: str) -> Optional[str]:
    """Detect institution name from query."""
    q = query.lower()
    institutions = {
        "kth": "KTH", "kungliga tekniska": "KTH",
        "chalmers": "Chalmers",
        "lund": "Lund", "lunds universitet": "LU",
        "uppsala": "Uppsala", "uu": "UU",
        "linköping": "LiU", "liu": "LiU",
        "stockholm": "SU", "su": "SU",
        "luleå": "LTU", "ltu": "LTU",
        "göteborg": "GU",
        "irf": "IRF", "rymdfysik": "IRF",
        "rise": "RISE",
    }
    for keyword, inst_id in institutions.items():
        if keyword in q:
            return inst_id
    return None


RESEARCH_KEYWORDS = {
    "forskning", "forskare", "studie", "artikel", "publikation",
    "abstract", "vetenskap", "akademi", "universitet", "institution",
    "research", "study", "paper", "scientist", "h-index",
}


def classify_query(query: str) -> str:
    """Classify query type: 'platform', 'research', or 'greeting'."""
    q = query.lower().strip().rstrip("!?.,")

    greetings = {"hej", "hallå", "hello", "hi", "tjena", "god dag", "hejsan"}
    if q in greetings:
        return "greeting"

    if any(kw in q for kw in RESEARCH_KEYWORDS):
        return "research"

    # Check for domain/theme/institution mentions
    if _detect_domain(q) or _detect_theme(q) or _detect_institution(q):
        return "research"

    return "platform"


def query_graph(query: str) -> list[GraphQuery]:
    """Run appropriate Cypher queries against Neo4j based on the query.

    Returns a list of GraphQuery results to inject into context.
    """
    results = []

    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    except Exception as e:
        logger.error("Failed to connect to Neo4j: %s", e)
        return results

    try:
        with driver.session() as session:
            # Detect what to query
            institution = _detect_institution(query)
            theme = _detect_theme(query)
            domain = _detect_domain(query)

            # Researchers by institution
            if institution:
                records = list(session.run(
                    RESEARCHERS_BY_INSTITUTION, keyword=institution
                ))
                if records:
                    results.append(GraphQuery(
                        "researchers",
                        [dict(r) for r in records],
                        f"Forskare vid {institution}",
                    ))

            # Researchers by theme
            if theme:
                keyword = theme.replace("_", " ")
                records = list(session.run(
                    RESEARCHERS_BY_THEME, keyword=keyword
                ))
                if records:
                    results.append(GraphQuery(
                        "researchers",
                        [dict(r) for r in records],
                        f"Forskare inom {keyword}",
                    ))

            # Top researchers by domain
            if domain and not institution and not theme:
                records = list(session.run(
                    TOP_RESEARCHERS_BY_DOMAIN, domain=domain
                ))
                if records:
                    results.append(GraphQuery(
                        "researchers",
                        [dict(r) for r in records],
                        f"Toppforskare inom {domain}",
                    ))

            # Publications
            if theme:
                keyword = theme.replace("_", " ")
                records = list(session.run(
                    PUBLICATIONS_BY_THEME, keyword=keyword
                ))
                if records:
                    results.append(GraphQuery(
                        "publications",
                        [dict(r) for r in records],
                        f"Senaste publikationer inom {keyword}",
                    ))

            # Institutions by domain
            if domain and not institution:
                records = list(session.run(
                    INSTITUTIONS_BY_DOMAIN, domain=domain
                ))
                if records:
                    results.append(GraphQuery(
                        "institutions",
                        [dict(r) for r in records],
                        f"Institutioner inom {domain}",
                    ))

    except Exception as e:
        logger.error("Neo4j query failed: %s", e)
    finally:
        driver.close()

    return results
