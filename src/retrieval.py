from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .data_models import ConvFinQARecord

def _flatten_table(table: Dict[str, Dict[str, object]]) -> List[str]:
    # Convert table dict[col][row] -> "row | col | value" lines for indexing
    lines: List[str] = []
    for col, rowmap in table.items():
        for row, val in rowmap.items():
            lines.append(f"{row} | {col} | {val}")
    return lines

def build_doc_chunks(record: ConvFinQARecord) -> List[str]:
    chunks: List[str] = []
    # Basic paragraph-ish chunks
    for block in (record.doc.pre_text, record.doc.post_text):
        if not block:
            continue
        for para in block.split("\n\n"):
            para = para.strip()
            if len(para) > 30:
                chunks.append(para)
    # Table
    chunks.extend(_flatten_table(record.doc.table))
    if not chunks:
        chunks = [record.doc.pre_text or record.doc.post_text]
    return chunks

class PerDocRetriever:
    """TF-IDF retriever restricted to a single record/document."""
    def __init__(self, record: ConvFinQARecord):
        self.chunks = build_doc_chunks(record)
        self.vectorizer = TfidfVectorizer(stop_words="english", max_df=0.95)
        self.matrix = self.vectorizer.fit_transform(self.chunks)

    def query(self, q: str, k: int = 6) -> List[Tuple[str, float]]:
        if not q.strip():
            return []
        q_vec = self.vectorizer.transform([q])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        idxs = np.argsort(-sims)
        out: List[Tuple[str, float]] = []
        for i in idxs[:k]:
            out.append((self.chunks[i], float(sims[i])))
        return out
