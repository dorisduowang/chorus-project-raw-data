"""
Hypergraph Storage

JSON persistence with hot-reload support for the hypergraph.
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

from .models import HyperGraph


class HyperGraphStorage:
    """
    Manages hypergraph persistence to JSON with hot-reload support.

    Follows the same pattern as RegistryLookup for consistency.
    """

    def __init__(self, path: str = "Data/hypergraph.json"):
        self.path = Path(path)
        self.graph: Optional[HyperGraph] = None

        # Hot-reload tracking
        self._last_mtime: Optional[float] = None
        self._reload_check_interval = 10  # seconds
        self._last_reload_check = 0.0

    def load(self) -> HyperGraph:
        """Load or create the hypergraph."""
        if self.path.exists():
            with open(self.path, "r") as f:
                data = json.load(f)
            self.graph = HyperGraph.from_dict(data)
            self._last_mtime = self.path.stat().st_mtime
            stats = self.graph.stats()
            print(f"Loaded hypergraph: {self.path} ({stats['total_nodes']} nodes, {stats['total_edges']} edges)")
        else:
            print(f"No hypergraph found at {self.path}, creating empty graph")
            self.graph = HyperGraph()

        return self.graph

    def save(self, graph: Optional[HyperGraph] = None) -> None:
        """Save the hypergraph to JSON."""
        if graph is not None:
            self.graph = graph

        if self.graph is None:
            raise ValueError("No hypergraph to save")

        # Ensure directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)

        data = self.graph.to_dict()
        data["meta"] = {
            "saved_at": datetime.now().isoformat(),
            "version": "1.0.0",
        }

        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

        self._last_mtime = self.path.stat().st_mtime
        stats = self.graph.stats()
        print(f"Saved hypergraph: {self.path} ({stats['total_nodes']} nodes, {stats['total_edges']} edges)")

    def reload_if_changed(self) -> bool:
        """Check if file changed and reload if needed."""
        now = time.time()

        if now - self._last_reload_check < self._reload_check_interval:
            return False
        self._last_reload_check = now

        if not self.path.exists():
            return False

        current_mtime = self.path.stat().st_mtime
        if current_mtime != self._last_mtime:
            old_stats = self.graph.stats() if self.graph else {"total_nodes": 0, "total_edges": 0}
            self.load()
            new_stats = self.graph.stats()
            print(f"Hypergraph reloaded: {old_stats['total_nodes']} -> {new_stats['total_nodes']} nodes")
            return True

        return False

    def get_graph(self) -> HyperGraph:
        """Get the current hypergraph, loading if needed."""
        if self.graph is None:
            self.load()
        return self.graph


# Module-level convenience functions

_default_storage: Optional[HyperGraphStorage] = None


def get_storage(path: str = "Data/hypergraph.json") -> HyperGraphStorage:
    """Get or create the default storage instance."""
    global _default_storage
    if _default_storage is None or str(_default_storage.path) != path:
        _default_storage = HyperGraphStorage(path)
    return _default_storage


def load_hypergraph(path: str = "Data/hypergraph.json") -> HyperGraph:
    """Load a hypergraph from JSON."""
    storage = get_storage(path)
    return storage.load()


def save_hypergraph(graph: HyperGraph, path: str = "Data/hypergraph.json") -> None:
    """Save a hypergraph to JSON."""
    storage = get_storage(path)
    storage.save(graph)
