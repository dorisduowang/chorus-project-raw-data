#!/usr/bin/env python3
"""
Comprehensive System Test - 100 Diverse Prompts

Tests the CHORUS system with a variety of query types:
- 50 simple queries (single intent, lookups)
- 50 complex/compound queries (multi-intent, requiring reasoning)

Usage:
    python tests/test_system_comprehensive.py

Requires:
    - RAG server running on localhost:8765
"""

import os
import sys
import json
import time
import asyncio
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and (key not in os.environ or not os.environ.get(key)):
                    os.environ[key] = value


# =============================================================================
# Test Prompts - 50 Simple Queries
# =============================================================================

SIMPLE_QUERIES = [
    # People lookups (10)
    {"query": "Who is James Evans?", "category": "people", "expected_type": "lookup"},
    {"query": "Tell me about Jake Burchard", "category": "people", "expected_type": "lookup"},
    {"query": "What is Honglin Bao's research focus?", "category": "people", "expected_type": "lookup"},
    {"query": "Who are the PhD students in the lab?", "category": "people", "expected_type": "lookup"},
    {"query": "List the postdocs at Knowledge Lab", "category": "people", "expected_type": "lookup"},
    {"query": "What is Hyejin Youn's h-index?", "category": "people", "expected_type": "lookup"},
    {"query": "Who has the most publications?", "category": "people", "expected_type": "lookup"},
    {"query": "Find researchers working on AI", "category": "people", "expected_type": "lookup"},
    {"query": "Who is the lab director?", "category": "people", "expected_type": "lookup"},
    {"query": "What's Eamon Duede's email?", "category": "people", "expected_type": "lookup"},

    # Project queries (10)
    {"query": "What is the APTO project?", "category": "projects", "expected_type": "lookup"},
    {"query": "Tell me about C3S2", "category": "projects", "expected_type": "lookup"},
    {"query": "What grants does the lab have?", "category": "projects", "expected_type": "lookup"},
    {"query": "What is Socio-Cognitive AI?", "category": "projects", "expected_type": "lookup"},
    {"query": "List active research projects", "category": "projects", "expected_type": "lookup"},
    {"query": "What's the MURI project about?", "category": "projects", "expected_type": "lookup"},
    {"query": "Who leads the APTO project?", "category": "projects", "expected_type": "lookup"},
    {"query": "What funding sources does the lab have?", "category": "projects", "expected_type": "lookup"},
    {"query": "Is there an NSF grant?", "category": "projects", "expected_type": "lookup"},
    {"query": "What are the main research themes?", "category": "projects", "expected_type": "lookup"},

    # Dataset queries (10)
    {"query": "What datasets are available on Midway?", "category": "datasets", "expected_type": "lookup"},
    {"query": "Do we have access to OpenAlex data?", "category": "datasets", "expected_type": "lookup"},
    {"query": "Where is the Reddit data stored?", "category": "datasets", "expected_type": "lookup"},
    {"query": "What's in the patent similarity dataset?", "category": "datasets", "expected_type": "lookup"},
    {"query": "How do I access the Twitter data?", "category": "datasets", "expected_type": "technical"},
    {"query": "Is there Web of Science data?", "category": "datasets", "expected_type": "lookup"},
    {"query": "What embeddings are precomputed?", "category": "datasets", "expected_type": "lookup"},
    {"query": "Where is the MAG data?", "category": "datasets", "expected_type": "lookup"},
    {"query": "List social media datasets", "category": "datasets", "expected_type": "lookup"},
    {"query": "What bibliometric data do we have?", "category": "datasets", "expected_type": "lookup"},

    # Technical queries (10)
    {"query": "How do I connect to Midway?", "category": "technical", "expected_type": "technical"},
    {"query": "What's the command to check GPU status?", "category": "technical", "expected_type": "technical"},
    {"query": "How do I submit a batch job?", "category": "technical", "expected_type": "technical"},
    {"query": "What Python environment should I use?", "category": "technical", "expected_type": "technical"},
    {"query": "How do I load the sentence transformer model?", "category": "technical", "expected_type": "technical"},
    {"query": "Where are the shared scripts?", "category": "technical", "expected_type": "technical"},
    {"query": "How much storage quota do I have?", "category": "technical", "expected_type": "technical"},
    {"query": "What's the syntax for slurm jobs?", "category": "technical", "expected_type": "technical"},
    {"query": "How do I install packages on Midway?", "category": "technical", "expected_type": "technical"},
    {"query": "What GPUs are available?", "category": "technical", "expected_type": "technical"},

    # General/factual queries (10)
    {"query": "What is the Knowledge Lab?", "category": "general", "expected_type": "lookup"},
    {"query": "Where is the lab located?", "category": "general", "expected_type": "lookup"},
    {"query": "What topics does the lab study?", "category": "general", "expected_type": "lookup"},
    {"query": "How many people are in the lab?", "category": "general", "expected_type": "lookup"},
    {"query": "What department is the lab in?", "category": "general", "expected_type": "lookup"},
    {"query": "When was the lab founded?", "category": "general", "expected_type": "lookup"},
    {"query": "What is computational social science?", "category": "general", "expected_type": "lookup"},
    {"query": "What is science of science?", "category": "general", "expected_type": "lookup"},
    {"query": "What's the lab's mission?", "category": "general", "expected_type": "lookup"},
    {"query": "What universities collaborate with us?", "category": "general", "expected_type": "lookup"},
]

# =============================================================================
# Test Prompts - 50 Complex/Compound Queries
# =============================================================================

COMPLEX_QUERIES = [
    # Multi-entity queries (10)
    {"query": "Compare James Evans and Hyejin Youn's research areas and publication metrics",
     "category": "compound", "expected_type": "compound", "complexity": "high"},
    {"query": "Who are the most cited researchers and what topics do they work on?",
     "category": "compound", "expected_type": "compound", "complexity": "high"},
    {"query": "List PhD students working on network science along with their advisors",
     "category": "compound", "expected_type": "compound", "complexity": "medium"},
    {"query": "What datasets would be useful for someone studying science of science?",
     "category": "compound", "expected_type": "compound", "complexity": "medium"},
    {"query": "Find people who work on both NLP and social science",
     "category": "compound", "expected_type": "compound", "complexity": "medium"},
    {"query": "Who has published on AI ethics and what were their main findings?",
     "category": "compound", "expected_type": "compound", "complexity": "high"},
    {"query": "Compare the APTO and MURI projects in terms of goals and team members",
     "category": "compound", "expected_type": "compound", "complexity": "high"},
    {"query": "What grants fund machine learning research and who are the PIs?",
     "category": "compound", "expected_type": "compound", "complexity": "medium"},
    {"query": "List all researchers from University of Chicago with their topics",
     "category": "compound", "expected_type": "compound", "complexity": "medium"},
    {"query": "Who collaborates with James Evans and on what topics?",
     "category": "compound", "expected_type": "compound", "complexity": "high"},

    # Multi-step reasoning queries (10)
    {"query": "I want to study scientific collaboration patterns. What data and methods should I use?",
     "category": "reasoning", "expected_type": "explore", "complexity": "high"},
    {"query": "How can I replicate the analysis from the APTO project proposal?",
     "category": "reasoning", "expected_type": "compound", "complexity": "high"},
    {"query": "What's the best approach to analyze patent citation networks?",
     "category": "reasoning", "expected_type": "explore", "complexity": "medium"},
    {"query": "I need to preprocess Twitter data for sentiment analysis. What steps should I take?",
     "category": "reasoning", "expected_type": "technical", "complexity": "medium"},
    {"query": "How would I build a knowledge graph from the lab's publications?",
     "category": "reasoning", "expected_type": "explore", "complexity": "high"},
    {"query": "What machine learning models work best for predicting research impact?",
     "category": "reasoning", "expected_type": "explore", "complexity": "medium"},
    {"query": "Design a study to measure scientific novelty using our datasets",
     "category": "reasoning", "expected_type": "explore", "complexity": "high"},
    {"query": "How can I use embeddings to find similar research papers?",
     "category": "reasoning", "expected_type": "technical", "complexity": "medium"},
    {"query": "What preprocessing is needed before training on the OpenAlex data?",
     "category": "reasoning", "expected_type": "technical", "complexity": "medium"},
    {"query": "How do I evaluate the quality of my document embeddings?",
     "category": "reasoning", "expected_type": "technical", "complexity": "medium"},

    # Cross-domain queries (10)
    {"query": "Connect me with someone who knows about both NLP and sociology",
     "category": "cross-domain", "expected_type": "connect", "complexity": "medium"},
    {"query": "What datasets on Midway would help study the spread of misinformation?",
     "category": "cross-domain", "expected_type": "compound", "complexity": "medium"},
    {"query": "Who in the lab could help with a project combining network science and machine learning?",
     "category": "cross-domain", "expected_type": "connect", "complexity": "high"},
    {"query": "Find papers that combine computational linguistics with cultural analysis",
     "category": "cross-domain", "expected_type": "compound", "complexity": "high"},
    {"query": "What research bridges computer science and the social sciences?",
     "category": "cross-domain", "expected_type": "explore", "complexity": "medium"},
    {"query": "Are there any collaborations between economists and data scientists here?",
     "category": "cross-domain", "expected_type": "connect", "complexity": "medium"},
    {"query": "How does the lab combine qualitative and quantitative methods?",
     "category": "cross-domain", "expected_type": "explore", "complexity": "medium"},
    {"query": "What interdisciplinary grants does the lab have?",
     "category": "cross-domain", "expected_type": "lookup", "complexity": "low"},
    {"query": "Find expertise overlaps between Knowledge Lab and Santa Fe Institute",
     "category": "cross-domain", "expected_type": "compound", "complexity": "high"},
    {"query": "Who works at the intersection of AI and philosophy of science?",
     "category": "cross-domain", "expected_type": "connect", "complexity": "medium"},

    # Temporal/contextual queries (10)
    {"query": "What are the recent trends in science of science research beyond our lab?",
     "category": "temporal", "expected_type": "external", "complexity": "medium"},
    {"query": "How has the lab's research focus evolved over time?",
     "category": "temporal", "expected_type": "explore", "complexity": "high"},
    {"query": "What are the latest developments in LLM-based research tools?",
     "category": "temporal", "expected_type": "external", "complexity": "medium"},
    {"query": "Compare our approach to others in the field of computational social science",
     "category": "temporal", "expected_type": "external", "complexity": "high"},
    {"query": "What new datasets have been added in the past year?",
     "category": "temporal", "expected_type": "lookup", "complexity": "low"},
    {"query": "Which recent publications have had the most impact?",
     "category": "temporal", "expected_type": "compound", "complexity": "medium"},
    {"query": "What's the current state of the art in scientific knowledge discovery?",
     "category": "temporal", "expected_type": "external", "complexity": "high"},
    {"query": "How do our methods compare to the latest OpenAI research tools?",
     "category": "temporal", "expected_type": "external", "complexity": "high"},
    {"query": "What conferences should lab members attend this year?",
     "category": "temporal", "expected_type": "external", "complexity": "medium"},
    {"query": "What emerging topics should we consider for future research?",
     "category": "temporal", "expected_type": "explore", "complexity": "high"},

    # Synthesis/analysis queries (10)
    {"query": "Summarize the lab's key contributions to science of science",
     "category": "synthesis", "expected_type": "explore", "complexity": "high"},
    {"query": "What are the common themes across all our active projects?",
     "category": "synthesis", "expected_type": "compound", "complexity": "high"},
    {"query": "Create a research landscape map of Knowledge Lab's expertise",
     "category": "synthesis", "expected_type": "explore", "complexity": "high"},
    {"query": "What methodological innovations has the lab developed?",
     "category": "synthesis", "expected_type": "explore", "complexity": "medium"},
    {"query": "Identify gaps in our current research portfolio",
     "category": "synthesis", "expected_type": "explore", "complexity": "high"},
    {"query": "What makes Knowledge Lab's approach unique?",
     "category": "synthesis", "expected_type": "explore", "complexity": "medium"},
    {"query": "Summarize the data infrastructure available to lab members",
     "category": "synthesis", "expected_type": "compound", "complexity": "medium"},
    {"query": "What skills are most valuable for new lab members to learn?",
     "category": "synthesis", "expected_type": "explore", "complexity": "medium"},
    {"query": "Create a reading list for someone new to computational social science",
     "category": "synthesis", "expected_type": "explore", "complexity": "high"},
    {"query": "What are the ethical considerations in our research areas?",
     "category": "synthesis", "expected_type": "explore", "complexity": "high"},
]


# =============================================================================
# Test Result Classes
# =============================================================================

@dataclass
class QueryResult:
    """Result of a single query test."""
    query: str
    category: str
    expected_type: str
    complexity: str
    response: str
    latency_ms: float
    classification: Optional[Dict] = None
    structured_results: int = 0
    semantic_results: int = 0
    error: Optional[str] = None
    success: bool = True

@dataclass
class TestSummary:
    """Summary of all test results."""
    total_queries: int = 0
    successful: int = 0
    failed: int = 0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    by_category: Dict[str, Dict] = field(default_factory=dict)
    by_complexity: Dict[str, Dict] = field(default_factory=dict)
    errors: List[Dict] = field(default_factory=list)


# =============================================================================
# Test Runner
# =============================================================================

class SystemTester:
    """Runs comprehensive system tests."""

    def __init__(self, rag_server_url: str = "http://localhost:8765"):
        self.rag_url = rag_server_url
        self.results: List[QueryResult] = []

    async def test_query(self, query_data: Dict) -> QueryResult:
        """Test a single query against the hybrid endpoint."""
        import aiohttp

        query = query_data["query"]
        category = query_data["category"]
        expected_type = query_data["expected_type"]
        complexity = query_data.get("complexity", "low")

        start_time = time.time()

        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.rag_url}/hybrid"
                params = {"q": query, "top_k": 5}

                async with session.get(url, params=params, timeout=30) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        return QueryResult(
                            query=query,
                            category=category,
                            expected_type=expected_type,
                            complexity=complexity,
                            response="",
                            latency_ms=(time.time() - start_time) * 1000,
                            error=f"HTTP {response.status}: {error_text[:200]}",
                            success=False,
                        )

                    data = await response.json()
                    latency_ms = (time.time() - start_time) * 1000

                    # Extract key metrics
                    classification = data.get("classification", {})
                    structured_results = len(data.get("structured_results", []))
                    semantic_results = len(data.get("semantic_results", []))
                    answer = data.get("structured_answer", "")

                    # Build response summary
                    if structured_results > 0:
                        response_text = f"[{structured_results} structured, {semantic_results} semantic] {answer[:200]}"
                    else:
                        # Use first semantic result
                        sem_results = data.get("semantic_results", [])
                        if sem_results:
                            response_text = f"[{semantic_results} semantic] {sem_results[0].get('text', '')[:200]}"
                        else:
                            response_text = "[No results]"

                    return QueryResult(
                        query=query,
                        category=category,
                        expected_type=expected_type,
                        complexity=complexity,
                        response=response_text,
                        latency_ms=latency_ms,
                        classification=classification,
                        structured_results=structured_results,
                        semantic_results=semantic_results,
                    )

        except Exception as e:
            return QueryResult(
                query=query,
                category=category,
                expected_type=expected_type,
                complexity=complexity,
                response="",
                latency_ms=(time.time() - start_time) * 1000,
                error=str(e),
                success=False,
            )

    async def run_all_tests(self, show_progress: bool = True) -> TestSummary:
        """Run all 100 tests and return summary."""
        all_queries = SIMPLE_QUERIES + COMPLEX_QUERIES

        print(f"\n{'='*70}")
        print(f"CHORUS System Comprehensive Test")
        print(f"{'='*70}")
        print(f"Total queries: {len(all_queries)}")
        print(f"  - Simple: {len(SIMPLE_QUERIES)}")
        print(f"  - Complex: {len(COMPLEX_QUERIES)}")
        print(f"{'='*70}\n")

        summary = TestSummary()
        summary.total_queries = len(all_queries)

        for i, query_data in enumerate(all_queries):
            if show_progress:
                progress = (i + 1) / len(all_queries) * 100
                query_preview = query_data["query"][:50]
                print(f"[{i+1:3d}/100] ({progress:5.1f}%) {query_preview}...", end=" ", flush=True)

            result = await self.test_query(query_data)
            self.results.append(result)

            if result.success:
                summary.successful += 1
                if show_progress:
                    print(f"✓ {result.latency_ms:.0f}ms")
            else:
                summary.failed += 1
                summary.errors.append({
                    "query": result.query,
                    "error": result.error,
                })
                if show_progress:
                    print(f"✗ {result.error[:30]}")

            # Track latency
            if result.latency_ms > 0:
                summary.max_latency_ms = max(summary.max_latency_ms, result.latency_ms)
                summary.min_latency_ms = min(summary.min_latency_ms, result.latency_ms)

            # Track by category
            cat = result.category
            if cat not in summary.by_category:
                summary.by_category[cat] = {"total": 0, "success": 0, "latencies": []}
            summary.by_category[cat]["total"] += 1
            if result.success:
                summary.by_category[cat]["success"] += 1
                summary.by_category[cat]["latencies"].append(result.latency_ms)

            # Track by complexity
            comp = result.complexity
            if comp not in summary.by_complexity:
                summary.by_complexity[comp] = {"total": 0, "success": 0, "latencies": []}
            summary.by_complexity[comp]["total"] += 1
            if result.success:
                summary.by_complexity[comp]["success"] += 1
                summary.by_complexity[comp]["latencies"].append(result.latency_ms)

            # Small delay to avoid overwhelming the server
            await asyncio.sleep(0.1)

        # Calculate averages
        all_latencies = [r.latency_ms for r in self.results if r.success]
        summary.avg_latency_ms = sum(all_latencies) / len(all_latencies) if all_latencies else 0

        # Calculate category averages
        for cat, data in summary.by_category.items():
            if data["latencies"]:
                data["avg_latency_ms"] = sum(data["latencies"]) / len(data["latencies"])
            else:
                data["avg_latency_ms"] = 0

        for comp, data in summary.by_complexity.items():
            if data["latencies"]:
                data["avg_latency_ms"] = sum(data["latencies"]) / len(data["latencies"])
            else:
                data["avg_latency_ms"] = 0

        return summary

    def print_report(self, summary: TestSummary):
        """Print detailed test report."""
        print(f"\n{'='*70}")
        print("TEST RESULTS SUMMARY")
        print(f"{'='*70}\n")

        # Overall stats
        success_rate = summary.successful / summary.total_queries * 100
        print(f"Overall Results:")
        print(f"  Total queries:    {summary.total_queries}")
        print(f"  Successful:       {summary.successful} ({success_rate:.1f}%)")
        print(f"  Failed:           {summary.failed}")
        print()

        # Latency stats
        print(f"Latency Statistics:")
        print(f"  Average:          {summary.avg_latency_ms:.0f}ms")
        print(f"  Min:              {summary.min_latency_ms:.0f}ms")
        print(f"  Max:              {summary.max_latency_ms:.0f}ms")
        print()

        # By category
        print(f"Results by Category:")
        print(f"  {'Category':<15} {'Total':>6} {'Success':>8} {'Rate':>7} {'Avg Latency':>12}")
        print(f"  {'-'*15} {'-'*6} {'-'*8} {'-'*7} {'-'*12}")
        for cat, data in sorted(summary.by_category.items()):
            rate = data["success"] / data["total"] * 100 if data["total"] > 0 else 0
            avg_lat = data.get("avg_latency_ms", 0)
            print(f"  {cat:<15} {data['total']:>6} {data['success']:>8} {rate:>6.1f}% {avg_lat:>10.0f}ms")
        print()

        # By complexity
        print(f"Results by Complexity:")
        print(f"  {'Complexity':<10} {'Total':>6} {'Success':>8} {'Rate':>7} {'Avg Latency':>12}")
        print(f"  {'-'*10} {'-'*6} {'-'*8} {'-'*7} {'-'*12}")
        for comp, data in sorted(summary.by_complexity.items()):
            rate = data["success"] / data["total"] * 100 if data["total"] > 0 else 0
            avg_lat = data.get("avg_latency_ms", 0)
            print(f"  {comp:<10} {data['total']:>6} {data['success']:>8} {rate:>6.1f}% {avg_lat:>10.0f}ms")
        print()

        # Structured vs semantic results
        structured_count = sum(1 for r in self.results if r.structured_results > 0)
        semantic_only = sum(1 for r in self.results if r.structured_results == 0 and r.semantic_results > 0)
        no_results = sum(1 for r in self.results if r.structured_results == 0 and r.semantic_results == 0 and r.success)

        print(f"Result Types:")
        print(f"  With structured data: {structured_count} ({structured_count/summary.total_queries*100:.1f}%)")
        print(f"  Semantic only:        {semantic_only} ({semantic_only/summary.total_queries*100:.1f}%)")
        print(f"  No results:           {no_results} ({no_results/summary.total_queries*100:.1f}%)")
        print()

        # Errors
        if summary.errors:
            print(f"Errors ({len(summary.errors)}):")
            for err in summary.errors[:10]:  # Show first 10
                print(f"  - {err['query'][:40]}...")
                print(f"    Error: {err['error'][:60]}")
            if len(summary.errors) > 10:
                print(f"  ... and {len(summary.errors) - 10} more errors")
            print()

        # Sample successful responses
        print(f"Sample Successful Responses:")
        print(f"{'-'*70}")
        samples = [r for r in self.results if r.success][:5]
        for r in samples:
            print(f"Q: {r.query[:60]}...")
            print(f"A: {r.response[:100]}...")
            print(f"   [{r.latency_ms:.0f}ms, {r.structured_results} structured, {r.semantic_results} semantic]")
            print()

        print(f"{'='*70}")
        print(f"Test completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")

    def save_results(self, filepath: str):
        """Save detailed results to JSON file."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "total_queries": len(self.results),
            "results": [
                {
                    "query": r.query,
                    "category": r.category,
                    "expected_type": r.expected_type,
                    "complexity": r.complexity,
                    "response_preview": r.response[:200],
                    "latency_ms": r.latency_ms,
                    "classification": r.classification,
                    "structured_results": r.structured_results,
                    "semantic_results": r.semantic_results,
                    "success": r.success,
                    "error": r.error,
                }
                for r in self.results
            ]
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Results saved to: {filepath}")


# =============================================================================
# Main
# =============================================================================

async def main():
    """Run the comprehensive system test."""
    import aiohttp

    # Check if server is running
    print("Checking RAG server...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8765/health", timeout=5) as response:
                if response.status == 200:
                    health = await response.json()
                    print(f"✓ Server healthy: {health['chunks']} chunks indexed")
                else:
                    print(f"✗ Server returned {response.status}")
                    return
    except Exception as e:
        print(f"✗ Server not available: {e}")
        print("Please start the server with: python rag_server_fastapi.py")
        return

    # Run tests
    tester = SystemTester()
    summary = await tester.run_all_tests()

    # Print report
    tester.print_report(summary)

    # Save results
    results_path = Path(__file__).parent / "test_results_comprehensive.json"
    tester.save_results(str(results_path))


if __name__ == "__main__":
    asyncio.run(main())
