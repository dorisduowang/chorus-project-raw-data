"""
Voice Profiles Module

Builds and manages disposition-based voice profiles for the Muse agent.
Based on taxonomy from Bassett/Zurn, Berlin, Dyson, Kuhn, et al.

Usage:
    from voice_profiles import VoiceProfileStore

    store = VoiceProfileStore()

    # Get a compound type
    bridger = store.get_compound_type("playful_bridger")

    # Get an individual profile
    profile = store.get_individual("james_evans")

    # Select voices for a query
    voices = store.select_voices(
        query="What are the implications of LLMs for science?",
        n=3
    )

    # Generate critique prompts
    prompts = store.generate_voice_prompts(voices)
"""

from .store import VoiceProfileStore
from .models import (
    DispositionVector,
    CritiqueBehavior,
    CompoundType,
    IndividualProfile,
)

__all__ = [
    "VoiceProfileStore",
    "DispositionVector",
    "CritiqueBehavior",
    "CompoundType",
    "IndividualProfile",
]
