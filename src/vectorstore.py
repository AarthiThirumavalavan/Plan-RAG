from __future__ import annotations
import os
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

# Chroma
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except Exception:
    chromadb = None

# Pinecone
try:
    from pinecone import Pinecone, ServerlessSpec
except Exception:
    Pinecone = None

from .embeddings import embed_texts

@dataclass
class DenseHit:
    text: str
    score: float
    meta: Dict[str, Any]

class DenseStore:
    def upsert(self, ids: List[str], texts: List[str], embeddings: List[List[float]], metadatas: List[Dict[str, Any]]): ...
    def query(self, query_embedding: List[float], k: int) -> List[DenseHit]: ...
    def close(self): ...

class ChromaStore(DenseStore):
    def __init__(self, collection_name: str, persist_path: str = ".chroma"):
        if chromadb is None:
            raise RuntimeError("chromadb is not installed. pip install chromadb")
        self.client = chromadb.PersistentClient(path=persist_path, settings=ChromaSettings(anonymized_telemetry=False))
        # cosine distance → similarity = 1 - distance
        self.col = self.client.get_or_create_collection(collection_name, metadata={"hnsw:space": "cosine"})

    def upsert(self, ids, texts, embeddings, metadatas):
        self.col.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)

    def query(self, query_embedding, k):
        res = self.col.query(query_embeddings=[query_embedding], n_results=k, include=["distances", "documents", "metadatas"])
        docs = res.get("documents", [[]])[0]
        dists = res.get("distances", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        hits = []
        for doc, dist, meta in zip(docs, dists, metas):
            sim = 1.0 - float(dist)
            hits.append(DenseHit(text=doc, score=sim, meta=meta or {}))
        return hits

    def close(self):
        pass

class PineconeStore(DenseStore):
    def __init__(self, index_name: str, dimension: int, metric: str = "cosine", cloud: str = "aws", region: str = "us-east-1"):
        if Pinecone is None:
            raise RuntimeError("pinecone-client is not installed. pip install pinecone-client")
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise RuntimeError("PINECONE_API_KEY is not set.")
        self.pc = Pinecone(api_key=api_key)
        if index_name not in [i["name"] for i in self.pc.list_indexes().get("indexes", [])]:
            self.pc.create_index(name=index_name, dimension=dimension, metric=metric,
                                 spec=ServerlessSpec(cloud=cloud, region=region))
        self.index = self.pc.Index(index_name)

    def upsert(self, ids, texts, embeddings, metadatas):
        vectors = [{"id": vid, "values": emb, "metadata": {"text": txt, **(meta or {})}} for vid, txt, emb, meta in zip(ids, texts, embeddings, metadatas)]
        self.index.upsert(vectors=vectors)

    def query(self, query_embedding, k):
        res = self.index.query(vector=query_embedding, top_k=k, include_metadata=True)
        hits = []
        for match in res.get("matches", []):
            hits.append(DenseHit(text=(match["metadata"] or {}).get("text",""), score=float(match["score"]), meta=match["metadata"] or {}))
        return hits

    def close(self):
        pass

def make_dense_store(collection_name: str, dim_hint: Optional[int] = None) -> DenseStore:
    backend = os.getenv("VECTOR_DB", "chroma").lower()
    if backend == "pinecone":
        dim = dim_hint or len(embed_texts(["dim_probe"])[0])
        index_name = os.getenv("PINECONE_INDEX", "convfinqa")
        return PineconeStore(index_name=index_name, dimension=dim, metric="cosine")
    else:
        persist = os.getenv("CHROMA_DIR", ".chroma")
        return ChromaStore(collection_name=collection_name, persist_path=persist)
