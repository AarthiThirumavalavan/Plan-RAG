from __future__ import annotations
import os
from typing import List
from openai import OpenAI

def _client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL")  # point to OpenAI-compatible Qwen embedding endpoint if needed
    )

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a batch of texts via an OpenAI-compatible API.
    Set EMBED_MODEL to your Qwen embedding model/deployment (finance-suited).
    """
    model = os.getenv("EMBED_MODEL", "qwen-embedding")
    client = _client()
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]

def embed_text(text: str) -> List[float]:
    return embed_texts([text])[0]
