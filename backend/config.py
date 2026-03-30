"""
Configuration for the Digital Earth Sweden chatbot backend.
"""

import os


# vLLM / LLM settings
VLLM_URL: str = os.getenv("VLLM_URL", "http://vllm:8000/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "512"))
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# Retrieval settings
RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "3"))
RETRIEVAL_SCORE_THRESHOLD: float = float(
    os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.30")
)

# Qdrant settings
QDRANT_URL: str = os.getenv("QDRANT_URL", "http://qdrant:6333")
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "des_knowledge")

# Embedding model
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
EMBEDDING_DIMENSION: int = 384  # all-MiniLM-L6-v2 output dimension

# Digital Earth Sweden sources
DES_WORDPRESS_URL: str = os.getenv(
    "DES_WORDPRESS_URL", "https://digitalearth.se"
)
DES_STAC_URL: str = os.getenv(
    "DES_STAC_URL", "https://explorer.digitalearth.se/stac"
)

# Chunking settings
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "64"))

# Rate limiting
RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))

# CORS
ALLOWED_ORIGINS: list[str] = [
    "https://digitalearth.se",
    "https://www.digitalearth.se",
    os.getenv("EXTRA_CORS_ORIGIN", ""),
]
ALLOWED_ORIGINS = [o for o in ALLOWED_ORIGINS if o]

# Session settings
MAX_HISTORY_MESSAGES: int = 5

# Logging
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
