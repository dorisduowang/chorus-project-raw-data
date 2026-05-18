"""
Query Logger Module

Logs all queries for analysis and improvement of the system.
Supports analysis of failures, common patterns, and performance.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import Counter
import threading


class QueryLogger:
    """
    Thread-safe query logger that writes to a JSONL file.
    """

    def __init__(self, log_path: str = "Data/logs/queries.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(
        self,
        query: str,
        classification: Dict[str, Any],
        num_results: int,
        latency_ms: float,
        corrected_query: Optional[str] = None,
        corrections: Optional[List[Dict]] = None,
        error: Optional[str] = None,
    ):
        """Log a query and its results."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "classification": {
                "type": classification.get("type", "unknown"),
                "entity_types": classification.get("entity_types", []),
                "confidence": classification.get("confidence", 0),
                "is_compound": classification.get("is_compound", False),
            },
            "num_results": num_results,
            "latency_ms": round(latency_ms, 2),
        }

        if corrected_query and corrected_query != query:
            entry["corrected_query"] = corrected_query
            entry["corrections"] = corrections or []

        if error:
            entry["error"] = error

        with self._lock:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")

    def get_recent(self, n: int = 100) -> List[Dict]:
        """Get the n most recent log entries."""
        if not self.log_path.exists():
            return []

        entries = []
        with open(self.log_path) as f:
            for line in f:
                if line.strip():
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        return entries[-n:]

    def get_failures(self, n: int = 100) -> List[Dict]:
        """Get queries with zero results."""
        entries = self.get_recent(n * 10)  # Get more to filter
        failures = [e for e in entries if e.get("num_results", 0) == 0]
        return failures[-n:]

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregate statistics from the log."""
        entries = self.get_recent(10000)

        if not entries:
            return {"total_queries": 0}

        total = len(entries)
        failures = sum(1 for e in entries if e.get("num_results", 0) == 0)
        errors = sum(1 for e in entries if e.get("error"))
        corrections_made = sum(1 for e in entries if e.get("corrected_query"))

        latencies = [e.get("latency_ms", 0) for e in entries]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0

        # Count entity types
        entity_type_counts = Counter()
        for entry in entries:
            for et in entry.get("classification", {}).get("entity_types", []):
                entity_type_counts[et] += 1

        # Count query types
        query_type_counts = Counter()
        for entry in entries:
            qt = entry.get("classification", {}).get("type", "unknown")
            query_type_counts[qt] += 1

        return {
            "total_queries": total,
            "success_rate": round((total - failures) / total * 100, 1) if total else 0,
            "failure_count": failures,
            "error_count": errors,
            "corrections_made": corrections_made,
            "avg_latency_ms": round(avg_latency, 1),
            "entity_type_distribution": dict(entity_type_counts.most_common(20)),
            "query_type_distribution": dict(query_type_counts),
        }


class QueryAnalyzer:
    """
    Analyzes query logs to identify patterns and suggest improvements.
    """

    def __init__(self, logger: QueryLogger):
        self.logger = logger

    def find_common_failures(self, min_count: int = 2) -> List[Dict]:
        """Find queries that fail repeatedly."""
        failures = self.logger.get_failures(1000)

        # Group by normalized query
        query_counts = Counter()
        for f in failures:
            normalized = self._normalize_query(f.get("query", ""))
            query_counts[normalized] += 1

        common = [
            {"query": q, "count": c}
            for q, c in query_counts.most_common(20)
            if c >= min_count
        ]
        return common

    def find_unmatched_keywords(self) -> List[Dict]:
        """Find keywords that appear in failed queries but not in vocabulary."""
        failures = self.logger.get_failures(500)

        # Extract words from failed queries
        word_counts = Counter()
        for f in failures:
            query = f.get("query", "").lower()
            words = [w for w in query.split() if len(w) >= 4 and w.isalpha()]
            word_counts.update(words)

        # Return most common words from failures
        return [
            {"word": w, "count": c}
            for w, c in word_counts.most_common(30)
        ]

    def find_typo_patterns(self) -> List[Dict]:
        """Find common typo patterns from corrections."""
        entries = self.logger.get_recent(1000)

        corrections = []
        for entry in entries:
            for corr in entry.get("corrections", []):
                corrections.append({
                    "original": corr.get("original"),
                    "corrected": corr.get("corrected"),
                })

        # Count correction pairs
        pair_counts = Counter()
        for c in corrections:
            pair = (c["original"], c["corrected"])
            pair_counts[pair] += 1

        return [
            {"original": p[0], "corrected": p[1], "count": c}
            for p, c in pair_counts.most_common(20)
        ]

    def suggest_new_patterns(self) -> List[str]:
        """Suggest regex patterns based on failed queries."""
        failures = self.logger.get_failures(200)

        suggestions = []

        # Look for common phrasings
        phrasings = Counter()
        for f in failures:
            query = f.get("query", "").lower()
            # Extract 2-3 word phrases
            words = query.split()
            for i in range(len(words) - 1):
                phrase = " ".join(words[i:i+2])
                phrasings[phrase] += 1
            for i in range(len(words) - 2):
                phrase = " ".join(words[i:i+3])
                phrasings[phrase] += 1

        # Suggest patterns for common phrases
        for phrase, count in phrasings.most_common(10):
            if count >= 2:
                # Convert to regex pattern suggestion
                pattern = phrase.replace(" ", r"\s+")
                suggestions.append(f"({pattern})")

        return suggestions

    def _normalize_query(self, query: str) -> str:
        """Normalize a query for comparison."""
        return " ".join(query.lower().split())


# Singleton instance
_logger: Optional[QueryLogger] = None


def get_logger() -> QueryLogger:
    """Get or create the singleton QueryLogger instance."""
    global _logger
    if _logger is None:
        _logger = QueryLogger()
    return _logger


def log_query(
    query: str,
    classification: Dict,
    num_results: int,
    latency_ms: float,
    **kwargs
):
    """Convenience function to log a query."""
    get_logger().log(query, classification, num_results, latency_ms, **kwargs)
