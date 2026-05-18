"""
Specialist agents for Chorus.

Each specialist handles a specific type of query:
- MemoryAgent: Fast factual retrieval from knowledge base
- MechanicAgent: Technical problems and setup help
- MuseAgent: Deep analysis and multi-perspective exploration
- MatchmakerAgent: Finding connections between people/ideas
"""

from .memory import MemoryAgent
from .mechanic import MechanicAgent
from .muse import MuseAgent
from .matchmaker import MatchmakerAgent

__all__ = [
    'MemoryAgent',
    'MechanicAgent',
    'MuseAgent',
    'MatchmakerAgent',
]
