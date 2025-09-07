from __future__ import annotations
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass, field
import re
from rank_bm25 import BM25Okapi

from .embeddings import embed_texts, embed_text
from .vectorstore import make_dense_store, DenseStore

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9%.$]+", text.lower())

@dataclass
class HybridRetriever:
    """
    Hybrid retrieval: dense (cosine) + sparse (BM25) with Reciprocal Rank Fusion.
    Dense search uses a vector DB backend (Chroma or Pinecone) with Qwen embeddings.
    Sparse search uses in-memory BM25 over the same chunks.
    """
    collection_name: str
    chunks: List[str]
    metadatas: List[Dict[str, Any]] = field(default_factory=list)
    dense: DenseStore = field(init=False)
    bm25: BM25Okapi = field(init=False)

    def __post_init__(self):
        if not self.metadatas:
            self.metadatas = [{} for _ in self.chunks]
        self.dense = make_dense_store(self.collection_name)
        ids = [f"{i}" for i in range(len(self.chunks))]
        embeds = embed_texts(self.chunks)
        self.dense.upsert(ids=ids, texts=self.chunks, embeddings=embeds, metadatas=self.metadatas)
        tokenized = [_tokenize(c) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized)

    def query(self, q: str, k: int = 6) -> List[Tuple[str, float]]:
        if not q.strip():
            return []
        # Dense
        q_emb = embed_text(q)
        dense_hits = self.dense.query(q_emb, k=max(k*2, 10))
        # Sparse
        token_q = _tokenize(q)
        scores = self.bm25.get_scores(token_q)
        idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:max(k*2, 10)]
        sparse_hits = [(i, float(scores[i])) for i in idxs]

        # Reciprocal Rank Fusion
        def rrf(rank: int, k_rrf: int = 60) -> float:
            return 1.0 / (k_rrf + rank)

        fused: Dict[int, float] = {}

        # dense ranks
        for r, h in enumerate(dense_hits, start=1):
            ci = None
            meta = h.meta or {}
            if "chunk_id" in meta:
                ci = int(meta["chunk_id"])
            else:
                # fallback: map by text equality
                try:
                    ci = self.chunks.index(h.text)
                except ValueError:
                    continue
            fused[ci] = fused.get(ci, 0.0) + rrf(r)

        # sparse ranks
        for r, (ci, _s) in enumerate(sparse_hits, start=1):
            fused[ci] = fused.get(ci, 0.0) + rrf(r)

        top = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]
        return [(self.chunks[i], score) for i, score in top]
