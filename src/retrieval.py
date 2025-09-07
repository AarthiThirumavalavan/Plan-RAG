from __future__ import annotations
from typing import Dict, List, Tuple

from .data_models import ConvFinQARecord
from .hybrid_retrieval import HybridRetriever

def _flatten_table(table: Dict[str, Dict[str, object]]) -> List[str]:
    lines: List[str] = []
    for col, rowmap in table.items():
        for row, val in rowmap.items():
            lines.append(f"{row} | {col} | {val}")
    return lines

def build_doc_chunks(record: ConvFinQARecord) -> List[str]:
    chunks: List[str] = []
    for block in (record.doc.pre_text, record.doc.post_text):
        if not block:
            continue
        for para in block.split("\n\n"):
            para = para.strip()
            if len(para) > 30:
                chunks.append(para)
    chunks.extend(_flatten_table(record.doc.table))
    if not chunks:
        chunks = [record.doc.pre_text or record.doc.post_text]
    return chunks

class PerDocHybridRetriever:
    """
    Per-document hybrid retriever using vector DB (cosine) + BM25 fusion.
    """
    def __init__(self, record: ConvFinQARecord):
        chunks = build_doc_chunks(record)
        metas = [{"record_id": record.id, "chunk_id": i} for i in range(len(chunks))]
        coll = f"convfinqa_{record.id}"
        self.hybrid = HybridRetriever(collection_name=coll, chunks=chunks, metadatas=metas)

    def query(self, q: str, k: int = 6) -> List[Tuple[str, float]]:
        return self.hybrid.query(q, k=k)
