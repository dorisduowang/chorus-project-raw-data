#!/usr/bin/env python3
"""
Embedding Model Benchmark for CHORUS RAG System

Compares embedding models on:
1. Model load time
2. Query encoding speed (100 realistic lab queries)
3. Batch encoding speed (50 text chunks ~200 words each)
4. Retrieval quality (top-3 results for test queries)
"""

import pickle
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

warnings.filterwarnings("ignore")

# Models to benchmark
MODELS = {
    "current": {
        "name": "BAAI/bge-large-en-v1.5",
        "dim": 1024,
        "description": "Current model"
    },
    "candidate1": {
        "name": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        "dim": 1536,
        "description": "Candidate 1 (large)"
    },
    "candidate2": {
        "name": "BAAI/bge-m3",
        "dim": 1024,
        "description": "Candidate 2 (multilingual)"
    },
    "baseline": {
        "name": "all-MiniLM-L6-v2",
        "dim": 384,
        "description": "Baseline (fast)"
    }
}

# Realistic lab queries for speed and quality testing
TEST_QUERIES = [
    "who works on network science",
    "James Evans publications",
    "where is the Twitter data",
    "computational social science research",
    "what grants does the lab have",
    "machine learning projects",
    "data storage and computing resources",
    "who studies culture and AI",
    "network analysis methods",
    "text analysis research",
    "knowledge graphs and science",
    "collaboration patterns in science",
    "large language models research",
    "social media data analysis",
    "citation network studies",
    "science of science",
    "artificial intelligence applications",
    "data visualization tools",
    "research computing infrastructure",
    "graduate students in the lab",
    "postdoc opportunities",
    "recent papers and publications",
    "funding sources and sponsors",
    "interdisciplinary research",
    "sociology of knowledge",
    "metascience and meta-research",
    "open science initiatives",
    "reproducibility in research",
    "data management policies",
    "lab meeting schedule",
    # Expand to 100 with variations
    "deep learning for text",
    "natural language processing",
    "social network analysis",
    "bibliometric studies",
    "patent analysis",
    "innovation research",
    "technology diffusion",
    "scientific discovery",
    "knowledge production",
    "academic collaboration",
    "research funding patterns",
    "career trajectories in academia",
    "peer review process",
    "journal publication strategies",
    "conference presentations",
    "workshop and seminar information",
    "lab computing cluster",
    "GPU resources available",
    "data storage solutions",
    "Python programming resources",
    "statistical methods training",
    "machine learning tutorials",
    "network visualization tools",
    "graph databases",
    "document parsing pipelines",
    "RAG system architecture",
    "embedding models for search",
    "semantic similarity methods",
    "information retrieval techniques",
    "text preprocessing steps",
    "named entity recognition",
    "topic modeling approaches",
    "sentiment analysis tools",
    "classification algorithms",
    "clustering methods",
    "dimensionality reduction",
    "feature engineering",
    "model evaluation metrics",
    "cross-validation strategies",
    "hyperparameter tuning",
    "transfer learning applications",
    "fine-tuning pretrained models",
    "prompt engineering techniques",
    "API integrations",
    "database connections",
    "web scraping tools",
    "data cleaning procedures",
    "missing data handling",
    "outlier detection",
    "time series analysis",
    "longitudinal studies",
    "panel data methods",
    "causal inference techniques",
    "experimental design",
    "survey methodology",
    "interview protocols",
    "ethnographic methods",
    "mixed methods research",
    "qualitative analysis software",
    "quantitative research tools",
    "reproducible research practices",
    "version control systems",
    "collaborative coding",
    "code review processes",
    "documentation standards",
    "project management tools",
    "team communication channels",
    "lab onboarding materials",
    "computing access requests",
    "data access procedures",
]

# Use first 100 queries
TEST_QUERIES = TEST_QUERIES[:100]

# Quality test queries (subset with expected topics)
QUALITY_QUERIES = [
    ("who works on network science", "Should mention researchers and network analysis"),
    ("James Evans publications", "Should find papers/work by James Evans"),
    ("where is the Twitter data", "Should mention data storage/locations"),
    ("computational social science", "Should describe CSS research"),
    ("machine learning projects", "Should list ML-related work"),
    ("lab computing resources", "Should mention clusters/GPUs/storage"),
    ("science of science research", "Should describe metascience work"),
    ("grant funding sources", "Should mention NSF/NIH/other funders"),
]


def generate_sample_chunks(chunks: list, n: int = 50) -> list[str]:
    """Generate sample chunks for batch encoding test (~200 words each)."""
    # Use real chunks from the data, selecting ones close to 200 words
    target_words = 200
    selected = []

    for chunk in chunks:
        text = chunk.get('text', '') if isinstance(chunk, dict) else chunk
        word_count = len(text.split())
        if 150 <= word_count <= 250:
            selected.append(text)
            if len(selected) >= n:
                break

    # If we don't have enough, pad with what we have
    while len(selected) < n:
        for chunk in chunks:
            text = chunk.get('text', '') if isinstance(chunk, dict) else chunk
            if len(text.split()) > 50:
                selected.append(text)
                if len(selected) >= n:
                    break

    return selected[:n]


def load_model(model_name: str):
    """Load a SentenceTransformer model and return load time."""
    from sentence_transformers import SentenceTransformer

    start = time.time()
    try:
        model = SentenceTransformer(model_name, trust_remote_code=True)
        load_time = time.time() - start
        # Test that encoding works
        try:
            _ = model.encode(["test"], convert_to_numpy=True)
        except Exception as e:
            return None, load_time, f"Model loaded but encoding failed: {str(e)[:100]}"
        return model, load_time, None
    except Exception as e:
        return None, 0, str(e)[:200]


def benchmark_query_encoding(model, queries: list[str]) -> tuple[float, float]:
    """Benchmark encoding speed for individual queries.

    Returns: (total_time_seconds, ms_per_query)
    """
    start = time.time()
    for query in queries:
        _ = model.encode([query], convert_to_numpy=True)
    total_time = time.time() - start
    ms_per_query = (total_time / len(queries)) * 1000
    return total_time, ms_per_query


def benchmark_batch_encoding(model, chunks: list[str]) -> tuple[float, float]:
    """Benchmark batch encoding speed.

    Returns: (total_time_seconds, docs_per_second)
    """
    start = time.time()
    _ = model.encode(chunks, convert_to_numpy=True, batch_size=32)
    total_time = time.time() - start
    docs_per_sec = len(chunks) / total_time
    return total_time, docs_per_sec


def retrieve_with_model(model, query: str, chunk_texts: list[str],
                        chunk_embeddings: np.ndarray, top_k: int = 3) -> list[tuple[str, float]]:
    """Retrieve top-k chunks for a query using the given model."""
    import faiss

    # Encode query
    query_emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_emb)

    # Create temporary index
    index = faiss.IndexFlatIP(chunk_embeddings.shape[1])
    index.add(chunk_embeddings)

    # Search
    scores, indices = index.search(query_emb, top_k)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx >= 0:
            results.append((chunk_texts[idx], float(score)))

    return results


def run_quality_test(model, chunk_texts: list[str], queries: list[tuple[str, str]]) -> dict:
    """Run quality test: retrieve top-3 results for each query."""
    import faiss

    print(f"    Encoding {len(chunk_texts)} chunks for quality test...")
    start = time.time()
    embeddings = model.encode(chunk_texts, convert_to_numpy=True, batch_size=32, show_progress_bar=True)
    faiss.normalize_L2(embeddings)
    encode_time = time.time() - start
    print(f"    Chunk encoding took {encode_time:.1f}s")

    results = {}
    for query, expected in queries:
        top_results = retrieve_with_model(model, query, chunk_texts, embeddings, top_k=3)
        results[query] = {
            "expected": expected,
            "results": top_results
        }

    return results


def main():
    print("=" * 80)
    print("CHORUS RAG Embedding Model Benchmark")
    print("=" * 80)

    # Load chunks
    print("\n1. Loading RAG chunks...")
    chunks_path = Path("/Users/robertward/Documents/GitHub.nosync/chorus/Data/rag_indexes/hybrid/hybrid_data.pkl")
    with open(chunks_path, 'rb') as f:
        data = pickle.load(f)

    chunks = data['chunks']
    chunk_texts = [c['text'] if isinstance(c, dict) else c for c in chunks]
    print(f"   Loaded {len(chunk_texts)} chunks")

    # Generate sample chunks for batch test
    sample_chunks = generate_sample_chunks(chunks, n=50)
    print(f"   Selected {len(sample_chunks)} sample chunks for batch test")
    avg_words = np.mean([len(c.split()) for c in sample_chunks])
    print(f"   Average words per chunk: {avg_words:.1f}")

    # Results storage
    results = {}

    print("\n2. Benchmarking models...")
    print("-" * 80)

    for model_key, model_info in MODELS.items():
        model_name = model_info["name"]
        print(f"\n>>> {model_info['description']}: {model_name}")
        print(f"    Expected dimension: {model_info['dim']}")

        # Load model
        print("    Loading model...")
        model, load_time, error = load_model(model_name)

        if error:
            print(f"    ERROR: {error}")
            results[model_key] = {
                "name": model_name,
                "description": model_info['description'],
                "error": error,
                "load_time": None,
                "ms_per_query": None,
                "docs_per_sec": None,
                "quality_results": None
            }
            continue

        print(f"    Load time: {load_time:.2f}s")

        # Check actual dimension
        try:
            test_emb = model.encode(["test"], convert_to_numpy=True)
            actual_dim = test_emb.shape[1]
            print(f"    Actual dimension: {actual_dim}")
        except Exception as e:
            print(f"    ERROR during dimension check: {str(e)[:100]}")
            results[model_key] = {
                "name": model_name,
                "description": model_info['description'],
                "error": f"Encoding failed: {str(e)[:100]}",
                "load_time": load_time,
                "ms_per_query": None,
                "docs_per_sec": None,
                "quality_results": None
            }
            del model
            import gc
            gc.collect()
            continue

        # Query encoding speed test
        print(f"    Running query encoding test ({len(TEST_QUERIES)} queries)...")
        query_total, ms_per_query = benchmark_query_encoding(model, TEST_QUERIES)
        print(f"    Query encoding: {ms_per_query:.2f} ms/query ({query_total:.2f}s total)")

        # Batch encoding speed test
        print(f"    Running batch encoding test ({len(sample_chunks)} chunks)...")
        batch_total, docs_per_sec = benchmark_batch_encoding(model, sample_chunks)
        print(f"    Batch encoding: {docs_per_sec:.2f} docs/sec ({batch_total:.2f}s total)")

        # Quality test (use subset of chunks for speed)
        # Use first 2000 chunks for quality test to keep it manageable
        quality_chunks = chunk_texts[:2000]
        print(f"    Running quality test on {len(quality_chunks)} chunks...")
        quality_results = run_quality_test(model, quality_chunks, QUALITY_QUERIES)

        results[model_key] = {
            "name": model_name,
            "description": model_info['description'],
            "dimension": actual_dim,
            "error": None,
            "load_time": load_time,
            "ms_per_query": ms_per_query,
            "docs_per_sec": docs_per_sec,
            "quality_results": quality_results
        }

        # Clean up to free memory
        del model
        import gc
        gc.collect()

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)

    print(f"\n{'Model':<45} {'Load(s)':<10} {'ms/query':<12} {'docs/sec':<12} {'Dim':<8}")
    print("-" * 87)

    for model_key, res in results.items():
        name = res['name'][:42] + "..." if len(res['name']) > 45 else res['name']
        if res['error']:
            print(f"{name:<45} {'FAILED':<10} {'-':<12} {'-':<12} {'-':<8}")
        else:
            load_t = f"{res['load_time']:.2f}" if res['load_time'] else "-"
            ms_q = f"{res['ms_per_query']:.2f}" if res['ms_per_query'] else "-"
            docs_s = f"{res['docs_per_sec']:.2f}" if res['docs_per_sec'] else "-"
            dim = str(res.get('dimension', '-'))
            print(f"{name:<45} {load_t:<10} {ms_q:<12} {docs_s:<12} {dim:<8}")

    # Print quality results
    print("\n" + "=" * 80)
    print("QUALITY TEST RESULTS")
    print("=" * 80)

    for model_key, res in results.items():
        if res['error'] or not res['quality_results']:
            continue

        print(f"\n>>> {res['description']}: {res['name']}")
        print("-" * 70)

        for query, data in res['quality_results'].items():
            print(f"\nQuery: \"{query}\"")
            print(f"Expected: {data['expected']}")
            print("Top 3 results:")
            for i, (text, score) in enumerate(data['results'], 1):
                # Truncate text for display
                display_text = text[:150] + "..." if len(text) > 150 else text
                display_text = display_text.replace('\n', ' ')
                print(f"  {i}. [{score:.4f}] {display_text}")

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)

    return results


if __name__ == "__main__":
    main()
