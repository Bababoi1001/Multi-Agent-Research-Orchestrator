"""Cross-session memory.

A local Chroma vector store lets the pipeline reuse findings from
earlier runs instead of re-researching the same ground every time.
Uses Chroma's built-in local embedding model, so no extra API key
is required for this piece.
"""

import hashlib
from typing import List

import chromadb

COLLECTION_NAME = "research_findings"
DB_PATH = "./chroma_db"


def get_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_or_create_collection(name=COLLECTION_NAME)


def query_past_findings(topic: str, n_results: int = 5) -> List[str]:
    """Return prior findings relevant to this topic, or [] if none exist."""
    collection = get_collection()
    if collection.count() == 0:
        return []
    results = collection.query(
        query_texts=[topic],
        n_results=min(n_results, collection.count()),
    )
    return results.get("documents", [[]])[0]


def store_findings(topic: str, sub_question: str, summary: str) -> None:
    """Persist one finding so future runs can retrieve it.

    The id is a hash of (topic, sub_question, summary) so re-running
    the same research doesn't create duplicate entries.
    """
    collection = get_collection()
    raw_id = f"{topic}|{sub_question}|{summary}"
    doc_id = hashlib.sha256(raw_id.encode()).hexdigest()[:16]
    collection.add(
        documents=[summary],
        metadatas=[{"topic": topic, "sub_question": sub_question}],
        ids=[doc_id],
    )