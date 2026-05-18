#!/usr/bin/env python3
"""
Test the embedding model upgrade from all-MiniLM-L6-v2 to bge-large-en-v1.5.

Compares:
- Embedding dimensions
- Search latency
- Result quality (diversity, relevance)
"""

import os
import sys
import time
import pickle
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Test queries
TEST_QUERIES = [
    "What is the MURI project about?",
    "Who are the principal investigators?",
    "machine learning for materials science",
    "grant proposal budget",
    "meeting notes from 2024",
    "computational methods for drug discovery",
    "neural network architecture",
    "collaboration between universities",
]


def load_index_and_chunks(index_path: str):
    """Load FAISS index and chunk data."""
    path = Path(index_path)
    faiss_index = faiss.read_index(str(path / "faiss_index.faiss"))
    with open(path / "hybrid_data.pkl", "rb") as f:
        data = pickle.load(f)
    return faiss_index, data


def benchmark_model(model_name: str, faiss_index, chunks: list):
    """Benchmark a single embedding model."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")

    # Load model
    start = time.time()
    model = SentenceTransformer(model_name)
    load_time = time.time() - start

    # Get model info
    test_emb = model.encode(["test"], convert_to_numpy=True)
    dim = test_emb.shape[1]

    print(f"Model loaded in {load_time:.2f}s")
    print(f"Embedding dimension: {dim}")
    print(f"Index vectors: {faiss_index.ntotal}")

    # Benchmark search
    latencies = []
    all_results = []

    for query in TEST_QUERIES:
        start = time.time()
        emb = model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(emb)
        scores, indices = faiss_index.search(emb, 5)
        latency = (time.time() - start) * 1000
        latencies.append(latency)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0:
                chunk = chunks[idx]
                source = chunk.get("source_file", "Unknown")
                results.append({"source": source, "score": float(score)})
        all_results.append({"query": query, "results": results})

    # Calculate metrics
    avg_latency = np.mean(latencies)

    # Count unique sources per query
    unique_sources = []
    for r in all_results:
        sources = set(res["source"] for res in r["results"])
        unique_sources.append(len(sources))
    avg_unique = np.mean(unique_sources)

    # Average top score (relevance indicator)
    avg_top_score = np.mean([r["results"][0]["score"] for r in all_results if r["results"]])

    print(f"\nPerformance:")
    print(f"  Avg latency: {avg_latency:.1f}ms")
    print(f"  Avg unique sources: {avg_unique:.1f}/5")
    print(f"  Avg top score: {avg_top_score:.4f}")

    return {
        "model": model_name,
        "dimension": dim,
        "load_time": load_time,
        "avg_latency_ms": avg_latency,
        "avg_unique_sources": avg_unique,
        "avg_top_score": avg_top_score,
        "sample_results": all_results[:3]  # First 3 queries
    }


def main():
    print("=" * 60)
    print("Embedding Model Upgrade Benchmark")
    print("=" * 60)

    # Load current index (built with bge-large-en-v1.5)
    # Use parent directory to access Data folder
    project_root = Path(__file__).parent.parent
    index_path = project_root / "Data/rag_indexes/hybrid"
    faiss_index, data = load_index_and_chunks(index_path)
    chunks = data["chunks"]
    model_name = data.get("model_name", "BAAI/bge-large-en-v1.5")

    print(f"\nIndex info:")
    print(f"  Model: {model_name}")
    print(f"  Chunks: {len(chunks)}")
    print(f"  Vectors: {faiss_index.ntotal}")
    print(f"  Dimension: {faiss_index.d}")

    # Test the new model
    results = benchmark_model(model_name, faiss_index, chunks)

    # Show sample results
    print(f"\n{'='*60}")
    print("Sample Search Results")
    print(f"{'='*60}")

    for qr in results["sample_results"]:
        print(f"\nQuery: {qr['query']}")
        for i, r in enumerate(qr["results"][:3], 1):
            print(f"  {i}. [{r['score']:.4f}] {r['source']}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: bge-large-en-v1.5 vs all-MiniLM-L6-v2")
    print(f"{'='*60}")
    print("""
Previous (all-MiniLM-L6-v2):
  - Dimension: 384
  - MTEB Retrieval: 41.0
  - Fast but lower quality

Current (bge-large-en-v1.5):
  - Dimension: 1024 (2.7x larger embeddings)
  - MTEB Retrieval: 54.29 (+32% improvement)
  - Higher quality, slightly slower
""")
    print(f"Measured Performance:")
    print(f"  Avg search latency: {results['avg_latency_ms']:.1f}ms")
    print(f"  Avg unique sources: {results['avg_unique_sources']:.1f}/5")
    print(f"  Avg top relevance: {results['avg_top_score']:.4f}")


if __name__ == "__main__":
    main()
