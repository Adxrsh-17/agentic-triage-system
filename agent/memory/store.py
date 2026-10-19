"""Unified Patient Memory Store (Local Memory & Pinecone Memory).

Defines:
- PatientMemory Protocol
- LocalMemory: in-memory, numpy cosine similarity over sentence embeddings
- PineconeMemory: managed vector database with metadata filtering by patient_id
- get_memory_store: factory selecting Pinecone when credentials exist, otherwise Local
"""

import logging
import os
from typing import Any, Dict, List, Optional, Protocol

import numpy as np

from agent.memory.embeddings import embed

logger = logging.getLogger("triage-memory")


class PatientMemory(Protocol):
    """Protocol defining the patient encounter memory interface."""

    def add_encounter(self, patient_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Store an encounter narrative and metadata for a patient."""
        ...

    def search(self, patient_id: str, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve top-k semantically relevant past encounters for a patient."""
        ...


class LocalMemory:
    """In-process patient memory store using numpy cosine similarity."""

    def __init__(self) -> None:
        # patient_id -> list of {"text": str, "vector": np.ndarray, "metadata": dict}
        self._store: Dict[str, List[Dict[str, Any]]] = {}

    def add_encounter(self, patient_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        if not text or not text.strip():
            return
        vec = embed([text])[0]
        entry = {
            "text": text.strip(),
            "vector": vec,
            "metadata": metadata or {},
        }
        self._store.setdefault(patient_id, []).append(entry)

    def search(self, patient_id: str, query: str, k: int = 3) -> List[Dict[str, Any]]:
        records = self._store.get(patient_id, [])
        if not records or not query or not query.strip():
            return []

        q_vec = embed([query])[0]
        scored: List[Dict[str, Any]] = []

        for rec in records:
            # Vectors are L2 normalized, so cosine similarity is direct dot product
            sim = float(np.dot(q_vec, rec["vector"]))
            scored.append({
                "text": rec["text"],
                "score": round(sim, 4),
                "metadata": rec["metadata"],
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:k]

    def clear(self) -> None:
        self._store.clear()


class PineconeMemory:
    """Pinecone serverless vector memory store with metadata filtering."""

    def __init__(self, api_key: str, index_name: str, dimension: int = 384) -> None:
        from pinecone import Pinecone

        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.dimension = dimension
        self.index = self.pc.Index(index_name)

    def add_encounter(self, patient_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        if not text or not text.strip():
            return
        vec = embed([text])[0].tolist()
        doc_id = f"{patient_id}_{abs(hash(text)) % 1000000}"
        meta = metadata or {}
        meta["patient_id"] = patient_id
        meta["text"] = text.strip()

        self.index.upsert(
            vectors=[{"id": doc_id, "values": vec, "metadata": meta}],
            namespace="encounters",
        )

    def search(self, patient_id: str, query: str, k: int = 3) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            return []
        q_vec = embed([query])[0].tolist()
        res = self.index.query(
            vector=q_vec,
            top_k=k,
            include_metadata=True,
            namespace="encounters",
            filter={"patient_id": {"$eq": patient_id}},
        )
        matches = []
        for match in res.get("matches", []):
            m_meta = match.get("metadata", {})
            matches.append({
                "text": m_meta.get("text", ""),
                "score": round(match.get("score", 0.0), 4),
                "metadata": m_meta,
            })
        return matches


# Global memory singleton
_GLOBAL_MEMORY: Optional[PatientMemory] = None


def get_memory_store() -> PatientMemory:
    """Factory selecting PineconeMemory when configured, otherwise LocalMemory."""
    global _GLOBAL_MEMORY
    if _GLOBAL_MEMORY is not None:
        return _GLOBAL_MEMORY

    api_key = os.environ.get("PINECONE_API_KEY", "").strip()
    index_name = os.environ.get("PINECONE_INDEX_NAME", "").strip()

    if api_key and index_name and not api_key.startswith("your_"):
        try:
            logger.info("Initializing PineconeMemory (Index: %s, Dim: 384)...", index_name)
            _GLOBAL_MEMORY = PineconeMemory(api_key=api_key, index_name=index_name, dimension=384)
            return _GLOBAL_MEMORY
        except Exception as exc:
            logger.warning("Pinecone initialization failed: %s. Falling back to LocalMemory.", exc)

    logger.info("Active memory backend: LocalMemory (in-process numpy cosine similarity)")
    _GLOBAL_MEMORY = LocalMemory()
    return _GLOBAL_MEMORY
