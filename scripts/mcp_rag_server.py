
#!/usr/bin/env python3
"""CHORUS RAG MCP Server

Run this script to start the MCP server for Claude Desktop.
"""

import json
import pickle
import string
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi
from fastmcp import FastMCP


@dataclass
class ChunkWithMetadata:
    text: str
    source_file: str = ""
    source_path: str = ""
    element_type: str = ""
    chunk_index: int = 0
    total_chunks: int = 0
    page_number: Optional[int] = None
    section: str = ""
    char_start: int = 0
    char_end: int = 0
    doc_type: str = ""
    date_year: Optional[int] = None
    date_month: Optional[int] = None
    timestamp_start: Optional[float] = None
    timestamp_end: Optional[float] = None
    speakers: Optional[list] = None
    dates_mentioned: Optional[list] = None
    entities_grants: Optional[list] = None
    entities_orgs: Optional[list] = None
    entities_people: Optional[list] = None
    entities_projects: Optional[list] = None
    key_terms: Optional[list] = None
    topics: Optional[list] = None

    @classmethod
    def from_dict(cls, d: dict) -> "ChunkWithMetadata":
        return cls(**d)

    def citation(self) -> str:
        parts = [self.source_file] if self.source_file else []
        if self.page_number:
            parts.append(f"p.{self.page_number}")
        return " | ".join(parts) if parts else "Unknown"


class QueryExpander:
    ACRONYMS = {
        "ai": "artificial intelligence", "ml": "machine learning",
        "nlp": "natural language processing", "llm": "large language model",
        "nsf": "national science foundation", "nih": "national institutes of health",
    }

    def expand_query(self, query: str) -> list[str]:
        queries = [query]
        q_lower = query.lower()
        for acr, full in self.ACRONYMS.items():
            if acr in q_lower.split():
                queries.append(q_lower.replace(acr, full))
        return queries


class RAGServer:
    def __init__(self, index_path: str):
        self.index_path = Path(index_path)
        self.faiss_index = None
        self.bm25_index = None
        self.chunks = []
        self.tokenized_chunks = []
        self.model = None
        self.expander = QueryExpander()

    def load(self):
        self.faiss_index = faiss.read_index(str(self.index_path / "faiss_index.faiss"))
        with open(self.index_path / "hybrid_data.pkl", "rb") as f:
            data = pickle.load(f)
            self.chunks = [ChunkWithMetadata.from_dict(c) for c in data["chunks"]]
            self.tokenized_chunks = data["tokenized_chunks"]
        self.bm25_index = BM25Okapi(self.tokenized_chunks)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        queries = self.expander.expand_query(query)
        results = {}

        for i, q in enumerate(queries):
            weight = 1.0 if i == 0 else 0.5
            emb = self.model.encode([q], convert_to_numpy=True)
            faiss.normalize_L2(emb)
            scores, indices = self.faiss_index.search(emb, top_k * 2)

            tokens = [t for t in q.lower().translate(str.maketrans("", "", string.punctuation)).split() if len(t) > 2]
            kw_scores = self.bm25_index.get_scores(tokens)
            max_kw = max(kw_scores) if max(kw_scores) > 0 else 1

            for idx, sem in zip(indices[0], scores[0]):
                if idx < 0: continue
                combined = 0.7 * sem + 0.3 * (kw_scores[idx] / max_kw)
                if idx not in results:
                    results[idx] = {"text": self.chunks[idx].text, "citation": self.chunks[idx].citation(), "score": 0}
                results[idx]["score"] += combined * weight

        return sorted(results.values(), key=lambda x: x["score"], reverse=True)[:top_k]


# Initialize with absolute path (indexes are at repo root Data/, not scripts/Data/)
REPO_DIR = Path(__file__).parent.parent.resolve()
INDEX_PATH = REPO_DIR / "Data" / "rag_indexes" / "hybrid"

rag = RAGServer(str(INDEX_PATH))
rag.load()

mcp = FastMCP("CHORUS RAG")

@mcp.tool()
def search_documents(query: str, top_k: int = 5) -> str:
    """Search the CHORUS knowledge base."""
    return json.dumps(rag.search(query, top_k), indent=2)

@mcp.tool()
def get_context(question: str, max_sources: int = 5) -> str:
    """Get context to answer a question about research, grants, or team."""
    results = rag.search(question, max_sources)
    context = "\n\n".join(f"[{r['citation']}]\n{r['text']}" for r in results)
    return json.dumps({"context": context, "citations": [r["citation"] for r in results]})

@mcp.tool()
def list_sources() -> str:
    """List available source documents."""
    sources = {}
    for c in rag.chunks:
        if c.source_file not in sources:
            sources[c.source_file] = 0
        sources[c.source_file] += 1
    return json.dumps(sorted([{"file": k, "chunks": v} for k, v in sources.items()], key=lambda x: -x["chunks"])[:30])

if __name__ == "__main__":
    mcp.run()
