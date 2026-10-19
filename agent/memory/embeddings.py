"""Embedding Generation Module for Patient Memory.

Features:
- SentenceTransformers lazy singleton with 'all-MiniLM-L6-v2' (384 dimensions)
- L2-normalized embedding vectors for exact cosine similarity via dot product
- Resilient offline fallback with explicit warning logging
"""

import hashlib
import logging
from typing import Any, List, Optional

import numpy as np

logger = logging.getLogger("triage-embeddings")

_MODEL_INSTANCE = None
_MODEL_LOAD_FAILED = False
EMBEDDING_DIM = 384


def get_embedding_model() -> Any:
    """Lazy-load the sentence-transformers model instance."""
    global _MODEL_INSTANCE, _MODEL_LOAD_FAILED
    if _MODEL_INSTANCE is not None:
        return _MODEL_INSTANCE
    if _MODEL_LOAD_FAILED:
        return None

    try:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading sentence-transformers/all-MiniLM-L6-v2 (384-dim)...")
        _MODEL_INSTANCE = SentenceTransformer("all-MiniLM-L6-v2")
        return _MODEL_INSTANCE
    except Exception as exc:
        _MODEL_LOAD_FAILED = True
        logger.warning(
            "Could not load sentence-transformers (%s). "
            "Falling back to deterministic pseudo-hash embedding (warning: semantic similarity degraded).",
            exc,
        )
        return None


def _hash_fallback_embed(text: str, dim: int = EMBEDDING_DIM) -> np.ndarray:
    """Deterministic hash-based embedding fallback when neural models are unavailable."""
    vec = np.zeros(dim, dtype=np.float32)
    tokens = text.lower().split()
    if not tokens:
        tokens = ["empty"]
    for idx, token in enumerate(tokens):
        h = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
        slot = h % dim
        sign = 1.0 if ((h >> 8) & 1) else -1.0
        vec[slot] += sign * (1.0 / (idx + 1))
    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-9)


def embed(texts: List[str]) -> np.ndarray:
    """Compute L2-normalized 384-dimensional embeddings for a list of strings."""
    if not texts:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    model = get_embedding_model()
    if model is not None:
        try:
            embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            return embeddings.astype(np.float32)
        except Exception as exc:
            logger.warning("Neural encoding failed (%s); falling back to hash embedding.", exc)

    # Hash fallback
    vecs = [_hash_fallback_embed(t) for t in texts]
    return np.array(vecs, dtype=np.float32)
