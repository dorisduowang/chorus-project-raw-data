import time
import asyncio
from typing import Dict, List, Tuple
from dataclasses import dataclass
from contextlib import contextmanager

@dataclass
class BenchmarkResult:
    operation: str
    avg_time: float
    min_time: float
    max_time: float
    p95_time: float
    iterations: int
    
    def __str__(self):
        return f"""
{self.operation}:
  Average: {self.avg_time*1000:.2f}ms
  Min:     {self.min_time*1000:.2f}ms
  Max:     {self.max_time*1000:.2f}ms
  P95:     {self.p95_time*1000:.2f}ms
  Iterations: {self.iterations}
"""

@contextmanager
def timer():
    """Context manager for timing operations"""
    start = time.perf_counter()
    times = {"start": start}
    yield times
    times["end"] = time.perf_counter()
    times["elapsed"] = times["end"] - times["start"]

class ChorusBenchmark:
    def __init__(self, rag_system, sample_queries: List[str] = None):
        """
        Initialize benchmark suite
        
        Args:
            rag_system: Your RAG system instance (LightRAG, custom, etc.)
            sample_queries: List of representative queries to test
        """
        self.rag = rag_system
        self.sample_queries = sample_queries or [
            "What are the key findings from our Q3 research?",
            "Summarize recent discussions about interdisciplinarity",
            "What did we decide about the citation prediction model?",
        ]
        
    def _compute_stats(self, times: List[float]) -> Tuple[float, float, float, float]:
        """Compute timing statistics"""
        sorted_times = sorted(times)
        n = len(sorted_times)
        return (
            sum(times) / n,  # avg
            sorted_times[0],  # min
            sorted_times[-1],  # max
            sorted_times[int(n * 0.95)] if n > 1 else sorted_times[0]  # p95
        )
    
    async def benchmark_embedding_generation(self, iterations: int = 10) -> BenchmarkResult:
        """Test query embedding speed"""
        print(f"Benchmarking embedding generation ({iterations} iterations)...")
        times = []
        
        for i, query in enumerate(self.sample_queries * (iterations // len(self.sample_queries) + 1)):
            if i >= iterations:
                break
            with timer() as t:
                # Adjust this based on your RAG implementation
                await self.rag.embed_query(query)
            times.append(t["elapsed"])
            
        avg, min_t, max_t, p95 = self._compute_stats(times)
        return BenchmarkResult("Embedding Generation", avg, min_t, max_t, p95, iterations)
    
    async def benchmark_vector_retrieval(self, iterations: int = 10, top_k: int = 5) -> BenchmarkResult:
        """Test vector search speed"""
        print(f"Benchmarking vector retrieval ({iterations} iterations, top_k={top_k})...")
        times = []
        
        for i, query in enumerate(self.sample_queries * (iterations // len(self.sample_queries) + 1)):
            if i >= iterations:
                break
            with timer() as t:
                # Adjust based on your implementation
                await self.rag.search_documents(query, top_k=top_k)
            times.append(t["elapsed"])
            
        avg, min_t, max_t, p95 = self._compute_stats(times)
        return BenchmarkResult(f"Vector Retrieval (k={top_k})", avg, min_t, max_t, p95, iterations)
    
    async def benchmark_llm_inference(self, iterations: int = 5) -> BenchmarkResult:
        """Test LLM response time with retrieved context"""
        print(f"Benchmarking LLM inference ({iterations} iterations)...")
        times = []
        
        for i, query in enumerate(self.sample_queries * (iterations // len(self.sample_queries) + 1)):
            if i >= iterations:
                break
            with timer() as t:
                # Full query pipeline
                await self.rag.query(query)
            times.append(t["elapsed"])
            
        avg, min_t, max_t, p95 = self._compute_stats(times)
        return BenchmarkResult("LLM Inference (full pipeline)", avg, min_t, max_t, p95, iterations)
    
    async def benchmark_end_to_end(self, iterations: int = 5) -> Dict[str, float]:
        """Detailed breakdown of a full query"""
        print(f"Benchmarking end-to-end pipeline breakdown ({iterations} iterations)...")
        
        breakdown_times = {
            "embedding": [],
            "retrieval": [],
            "context_prep": [],
            "llm_call": [],
            "total": []
        }
        
        for i, query in enumerate(self.sample_queries * (iterations // len(self.sample_queries) + 1)):
            if i >= iterations:
                break
                
            with timer() as total_t:
                # Step 1: Embed query
                with timer() as embed_t:
                    query_embedding = await self.rag.embed_query(query)
                breakdown_times["embedding"].append(embed_t["elapsed"])
                
                # Step 2: Retrieve documents
                with timer() as retrieval_t:
                    docs = await self.rag.search_documents(query, top_k=5)
                breakdown_times["retrieval"].append(retrieval_t["elapsed"])
                
                # Step 3: Prepare context (formatting, reranking, etc.)
                with timer() as context_t:
                    context = await self.rag.prepare_context(docs)
                breakdown_times["context_prep"].append(context_t["elapsed"])
                
                # Step 4: LLM call
                with timer() as llm_t:
                    response = await self.rag.generate_response(query, context)
                breakdown_times["llm_call"].append(llm_t["elapsed"])
                
            breakdown_times["total"].append(total_t["elapsed"])
        
        # Compute averages
        results = {}
        for stage, times in breakdown_times.items():
            avg = sum(times) / len(times)
            results[stage] = avg
            
        return results
    
    def print_breakdown(self, breakdown: Dict[str, float]):
        """Pretty print the breakdown analysis"""
        total = breakdown["total"]
        print("\n" + "="*60)
        print("END-TO-END PIPELINE BREAKDOWN")
        print("="*60)
        for stage, avg_time in breakdown.items():
            if stage == "total":
                continue
            pct = (avg_time / total) * 100
            print(f"{stage:20s}: {avg_time*1000:7.2f}ms ({pct:5.1f}%)")
        print("-"*60)
        print(f"{'TOTAL':20s}: {total*1000:7.2f}ms (100.0%)")
        print("="*60)
        
        # Identify bottleneck
        bottleneck_stage = max(
            [(k, v) for k, v in breakdown.items() if k != "total"],
            key=lambda x: x[1]
        )
        print(f"\n🔍 BOTTLENECK: {bottleneck_stage[0]} ({bottleneck_stage[1]*1000:.2f}ms)")
        print(self._get_optimization_suggestions(bottleneck_stage[0]))
    
    def _get_optimization_suggestions(self, bottleneck: str) -> str:
        """Provide optimization suggestions based on bottleneck"""
        suggestions = {
            "embedding": """
Suggestions:
  - Use smaller/faster embedding model (e.g., BGE-small, MiniLM)
  - Cache query embeddings for repeated queries
  - Move embedding model to GPU if not already
  - Consider batching if processing multiple queries
""",
            "retrieval": """
Suggestions:
  - Verify vector index is optimized (HNSW vs flat, proper parameters)
  - Reduce dimensionality of embeddings
  - Implement relevance threshold to stop early
  - Shard your index if corpus is very large
  - Use approximate nearest neighbor if not already
""",
            "context_prep": """
Suggestions:
  - Optimize reranking (faster reranker or skip if not needed)
  - Reduce chunk processing overhead
  - Parallelize document formatting
  - Cache formatted contexts for common queries
""",
            "llm_call": """
Suggestions:
  - Use Claude Haiku for simpler queries
  - Implement prompt caching (cache system prompt + context)
  - Reduce context size (fewer/smaller chunks)
  - Stream responses for perceived speed
  - Optimize prompt length
"""
        }
        return suggestions.get(bottleneck, "No specific suggestions available")
    
    async def run_full_benchmark(self):
        """Run complete benchmark suite"""
        print("\n" + "="*60)
        print("CHORUS RAG BENCHMARK SUITE")
        print("="*60 + "\n")
        
        results = []
        
        # Individual component benchmarks
        results.append(await self.benchmark_embedding_generation(iterations=20))
        results.append(await self.benchmark_vector_retrieval(iterations=20, top_k=5))
        results.append(await self.benchmark_vector_retrieval(iterations=20, top_k=10))
        results.append(await self.benchmark_llm_inference(iterations=5))
        
        print("\n" + "="*60)
        print("COMPONENT BENCHMARKS")
        print("="*60)
        for result in results:
            print(result)
        
        # Detailed breakdown
        breakdown = await self.benchmark_end_to_end(iterations=5)
        self.print_breakdown(breakdown)
        
        return results, breakdown


# Usage example:
async def main():
    # Initialize your RAG system
    # from your_rag_system import ChorusRAG
    # rag = ChorusRAG(...)
    
    # For demonstration, let's create a mock RAG system
    class MockRAG:
        async def embed_query(self, query):
            await asyncio.sleep(0.05)  # Simulate embedding time
            return [0.1] * 768
        
        async def search_documents(self, query, top_k=5):
            await asyncio.sleep(0.1)  # Simulate search time
            return [{"text": f"doc {i}", "score": 0.9} for i in range(top_k)]
        
        async def prepare_context(self, docs):
            await asyncio.sleep(0.02)  # Simulate context prep
            return "\n".join([d["text"] for d in docs])
        
        async def generate_response(self, query, context):
            await asyncio.sleep(0.8)  # Simulate LLM call
            return "Response"
        
        async def query(self, query):
            embedding = await self.embed_query(query)
            docs = await self.search_documents(query)
            context = await self.prepare_context(docs)
            return await self.generate_response(query, context)
    
    rag = MockRAG()
    
    # Run benchmarks
    benchmark = ChorusBenchmark(rag)
    results, breakdown = await benchmark.run_full_benchmark()

if __name__ == "__main__":
    asyncio.run(main())