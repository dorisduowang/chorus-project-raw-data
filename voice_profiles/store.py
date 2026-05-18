"""
Voice Profile Store

Loads, manages, and queries voice profiles for the Muse agent.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Union
import random

from .models import (
    CompoundType,
    IndividualProfile,
    DispositionVector,
    CritiqueBehavior,
)


class VoiceProfileStore:
    """
    Manages voice profiles for multi-perspective exploration.

    Loads profiles from Data/voice_profiles.json and provides methods
    to select and format voices for the Muse agent.
    """

    def __init__(self, profiles_path: Optional[str] = None):
        """
        Initialize the store.

        Args:
            profiles_path: Path to voice_profiles.json. If None, uses default.
        """
        if profiles_path is None:
            # Default to Data/voice_profiles.json relative to project root
            project_root = Path(__file__).parent.parent
            profiles_path = project_root / "Data" / "voice_profiles.json"

        self.profiles_path = Path(profiles_path)
        self._data: Dict = {}
        self._compound_types: Dict[str, CompoundType] = {}
        self._individuals: Dict[str, IndividualProfile] = {}

        self._load()

    def _load(self):
        """Load profiles from JSON file."""
        if not self.profiles_path.exists():
            raise FileNotFoundError(f"Voice profiles not found: {self.profiles_path}")

        with open(self.profiles_path, "r") as f:
            self._data = json.load(f)

        # Parse compound types
        for key, data in self._data.get("compound_types", {}).items():
            self._compound_types[key] = CompoundType.from_dict(key, data)

        # Parse individuals (skip template)
        for key, data in self._data.get("individuals", {}).items():
            if key.startswith("_"):
                continue
            self._individuals[key] = IndividualProfile.from_dict(data)

    def reload(self):
        """Reload profiles from disk."""
        self._compound_types.clear()
        self._individuals.clear()
        self._load()

    def get_compound_type(self, name: str) -> Optional[CompoundType]:
        """Get a compound type by name."""
        return self._compound_types.get(name)

    def get_individual(self, id: str) -> Optional[IndividualProfile]:
        """Get an individual profile by ID."""
        return self._individuals.get(id)

    def list_compound_types(self) -> List[str]:
        """List all compound type names."""
        return list(self._compound_types.keys())

    def list_individuals(self) -> List[str]:
        """List all individual profile IDs."""
        return list(self._individuals.keys())

    def select_voices(
        self,
        query: str,
        n: int = 3,
        include_individuals: bool = True,
        include_compound_types: bool = True,
        ensure_tension: bool = True,
    ) -> List[Union[CompoundType, IndividualProfile]]:
        """
        Select voices appropriate for a query.

        Args:
            query: The user's question or topic
            n: Number of voices to select
            include_individuals: Include individual profiles
            include_compound_types: Include compound type archetypes
            ensure_tension: Try to include voices with productive tensions

        Returns:
            List of voice profiles (CompoundType or IndividualProfile)
        """
        candidates = []

        if include_compound_types:
            candidates.extend(self._compound_types.values())

        if include_individuals:
            # Only include individuals with reasonable confidence
            candidates.extend(
                p for p in self._individuals.values()
                if p.confidence >= 0.3
            )

        if not candidates:
            # Fallback to compound types only
            candidates = list(self._compound_types.values())

        if len(candidates) <= n:
            return candidates

        # For now, simple random selection with tension consideration
        selected = []

        # First pick: random
        first = random.choice(candidates)
        selected.append(first)
        remaining = [c for c in candidates if c != first]

        # Try to pick voices with productive tension
        if ensure_tension and hasattr(first, 'critique_behavior'):
            tensions = first.critique_behavior.productive_tensions
            tension_matches = [
                c for c in remaining
                if (isinstance(c, CompoundType) and any(
                    t in c.name.lower().replace(" ", "_") for t in tensions
                ))
            ]
            if tension_matches:
                second = random.choice(tension_matches)
                selected.append(second)
                remaining = [c for c in remaining if c != second]

        # Fill remaining slots
        while len(selected) < n and remaining:
            pick = random.choice(remaining)
            selected.append(pick)
            remaining.remove(pick)

        return selected

    def select_voices_by_topic(
        self,
        topics: List[str],
        n: int = 3,
    ) -> List[Union[CompoundType, IndividualProfile]]:
        """
        Select voices based on topic relevance.

        Args:
            topics: List of topic keywords
            n: Number of voices

        Returns:
            List of voice profiles
        """
        # Topic-to-voice-type heuristics
        topic_affinities = {
            # Topics that benefit from formal/mathematical voices
            "math": ["formal_unifier"],
            "statistics": ["formal_unifier", "meticulous_naturalist"],
            "proof": ["formal_unifier"],
            "model": ["formal_unifier", "systematic_empire_builder"],

            # Topics that benefit from bridging/creative voices
            "interdisciplinary": ["playful_bridger", "seductive_scientist"],
            "analogy": ["playful_bridger"],
            "creativity": ["playful_bridger", "seductive_scientist"],
            "innovation": ["playful_bridger", "seductive_scientist"],

            # Topics that benefit from careful/patient voices
            "longitudinal": ["quiet_revolutionary", "meticulous_naturalist"],
            "historical": ["meticulous_naturalist", "quiet_revolutionary"],
            "taxonomy": ["meticulous_naturalist"],
            "classification": ["meticulous_naturalist"],

            # Topics that benefit from systems/scale voices
            "scale": ["systematic_empire_builder", "playful_bridger"],
            "organization": ["systematic_empire_builder"],
            "institution": ["systematic_empire_builder"],

            # Topics that benefit from paradigm-challenging voices
            "paradigm": ["quiet_revolutionary"],
            "assumption": ["quiet_revolutionary", "playful_bridger"],
            "foundation": ["quiet_revolutionary", "formal_unifier"],
        }

        # Score compound types by topic affinity
        scores = {name: 0.0 for name in self._compound_types}

        for topic in topics:
            topic_lower = topic.lower()
            for keyword, types in topic_affinities.items():
                if keyword in topic_lower:
                    for t in types:
                        if t in scores:
                            scores[t] += 1.0

        # Sort by score, take top n
        sorted_types = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected_names = [name for name, score in sorted_types[:n]]

        # If we don't have enough from topic matching, add random ones
        if len(selected_names) < n:
            remaining = [name for name in self._compound_types if name not in selected_names]
            random.shuffle(remaining)
            selected_names.extend(remaining[:n - len(selected_names)])

        return [self._compound_types[name] for name in selected_names if name in self._compound_types]

    def generate_voice_prompts(
        self,
        voices: List[Union[CompoundType, IndividualProfile]],
        include_tensions: bool = True,
    ) -> str:
        """
        Generate prompt text for a set of voices.

        Args:
            voices: List of voice profiles
            include_tensions: Include notes about productive tensions

        Returns:
            Formatted prompt string for Muse agent
        """
        parts = ["Channel these intellectual perspectives:\n"]

        for i, voice in enumerate(voices, 1):
            parts.append(f"\n### Voice {i}: {voice.name if hasattr(voice, 'name') else voice.id}")
            parts.append(voice.to_prompt())

        if include_tensions and len(voices) >= 2:
            parts.append("\n### Productive Tensions")
            parts.append("Let these voices engage each other directly—building on, ")
            parts.append("challenging, and reframing what others say. Key tensions:")

            for voice in voices:
                if hasattr(voice, 'critique_behavior') and voice.critique_behavior.productive_tensions:
                    tensions = voice.critique_behavior.productive_tensions
                    name = voice.name if hasattr(voice, 'name') else voice.id
                    parts.append(f"- {name} productively conflicts with: {', '.join(tensions)}")

        return "\n".join(parts)

    def add_individual(self, profile: IndividualProfile):
        """Add or update an individual profile."""
        self._individuals[profile.id] = profile

    def save(self):
        """Save current profiles to disk."""
        # Update data structure
        self._data["individuals"] = {
            id: profile.to_dict()
            for id, profile in self._individuals.items()
        }

        # Preserve template
        self._data["individuals"]["_template"] = self._data.get("individuals", {}).get("_template", {})

        with open(self.profiles_path, "w") as f:
            json.dump(self._data, f, indent=2)
