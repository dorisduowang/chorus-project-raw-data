"""
Data models for voice profiles.

Based on the scientific/creative disposition taxonomy from:
- Bassett & Zurn (Curiosity Archetypes)
- Isaiah Berlin (Hedgehog vs Fox)
- Freeman Dyson (Birds vs Frogs)
- Lévi-Strauss (Bricoleur vs Engineer)
- Guilford (Divergent vs Convergent)
- Kuhn (Revolutionary vs Normal)
- Polanyi (Tacit vs Explicit)
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Literal
from enum import Enum


class CuriosityStyle(str, Enum):
    """Bassett & Zurn curiosity archetypes."""
    BUSYBODY = "busybody"  # Broad, sparse - collects stories
    HUNTER = "hunter"      # Narrow, dense - pursues specific prey
    DANCER = "dancer"      # Loopy, unexpected - leaps between domains


@dataclass
class EpistemicAppetite:
    """How do you seek knowledge? What satisfies your curiosity?"""
    curiosity_style: CuriosityStyle = CuriosityStyle.HUNTER
    scope: float = 0.5          # 0=fox (many tools), 1=hedgehog (one big idea)
    surprise_seeking: float = 0.5  # 0=confirmation, 1=novelty


@dataclass
class CognitiveStyle:
    """How do you think through problems?"""
    thinking_mode: float = 0.5    # 0=convergent, 1=divergent
    abstraction: float = 0.5      # 0=frog (details), 1=bird (patterns)
    classification: float = 0.5   # 0=lumper, 1=splitter


@dataclass
class Methodological:
    """How do you approach the work itself?"""
    making_style: float = 0.5     # 0=engineer, 1=bricoleur
    formalism: float = 0.5        # 0=narrative, 1=mathematical
    tacit_explicit: float = 0.5   # 0=intuitive, 1=explicit


@dataclass
class Social:
    """How do you relate to other researchers?"""
    disclosure: float = 0.5       # 0=private, 1=open
    inquiry: float = 0.5          # 0=heads-down, 1=monitors others
    collaboration: float = 0.5    # 0=solitary, 0.5=dyadic, 1=collective
    competition: float = 0.5      # 0=communal, 1=agonistic


@dataclass
class Temporal:
    """How do you relate to time and knowledge history?"""
    paradigm_orientation: float = 0.5  # 0=normal science, 1=revolutionary
    patience: float = 0.5              # 0=quick iterations, 1=long-term
    archival: float = 0.5              # 0=present-focused, 1=historical


@dataclass
class Aesthetic:
    """What do you find beautiful in science?"""
    elegance_completeness: float = 0.5   # 0=completeness, 1=elegance
    mechanism_phenomenon: float = 0.5    # 0=phenomenological, 1=mechanist
    risk: float = 0.5                    # 0=cautious, 1=bold


@dataclass
class DispositionVector:
    """
    Full disposition profile for a voice.

    Each dimension captures an aspect of how a researcher/thinker
    approaches knowledge, problems, and collaboration.
    """
    epistemic_appetite: EpistemicAppetite = field(default_factory=EpistemicAppetite)
    cognitive_style: CognitiveStyle = field(default_factory=CognitiveStyle)
    methodological: Methodological = field(default_factory=Methodological)
    social: Social = field(default_factory=Social)
    temporal: Temporal = field(default_factory=Temporal)
    aesthetic: Aesthetic = field(default_factory=Aesthetic)

    @classmethod
    def from_dict(cls, d: dict) -> "DispositionVector":
        """Create from JSON-like dictionary."""
        return cls(
            epistemic_appetite=EpistemicAppetite(
                curiosity_style=CuriosityStyle(d.get("epistemic_appetite", {}).get("curiosity_style", "hunter")),
                scope=d.get("epistemic_appetite", {}).get("scope", 0.5),
                surprise_seeking=d.get("epistemic_appetite", {}).get("surprise_seeking", 0.5),
            ),
            cognitive_style=CognitiveStyle(
                thinking_mode=d.get("cognitive_style", {}).get("thinking_mode", 0.5),
                abstraction=d.get("cognitive_style", {}).get("abstraction", 0.5),
                classification=d.get("cognitive_style", {}).get("classification", 0.5),
            ),
            methodological=Methodological(
                making_style=d.get("methodological", {}).get("making_style", 0.5),
                formalism=d.get("methodological", {}).get("formalism", 0.5),
                tacit_explicit=d.get("methodological", {}).get("tacit_explicit", 0.5),
            ),
            social=Social(
                disclosure=d.get("social", {}).get("disclosure", 0.5),
                inquiry=d.get("social", {}).get("inquiry", 0.5),
                collaboration=d.get("social", {}).get("collaboration", 0.5),
                competition=d.get("social", {}).get("competition", 0.5),
            ),
            temporal=Temporal(
                paradigm_orientation=d.get("temporal", {}).get("paradigm_orientation", 0.5),
                patience=d.get("temporal", {}).get("patience", 0.5),
                archival=d.get("temporal", {}).get("archival", 0.5),
            ),
            aesthetic=Aesthetic(
                elegance_completeness=d.get("aesthetic", {}).get("elegance_completeness", 0.5),
                mechanism_phenomenon=d.get("aesthetic", {}).get("mechanism_phenomenon", 0.5),
                risk=d.get("aesthetic", {}).get("risk", 0.5),
            ),
        )

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "epistemic_appetite": {
                "curiosity_style": self.epistemic_appetite.curiosity_style.value,
                "scope": self.epistemic_appetite.scope,
                "surprise_seeking": self.epistemic_appetite.surprise_seeking,
            },
            "cognitive_style": {
                "thinking_mode": self.cognitive_style.thinking_mode,
                "abstraction": self.cognitive_style.abstraction,
                "classification": self.cognitive_style.classification,
            },
            "methodological": {
                "making_style": self.methodological.making_style,
                "formalism": self.methodological.formalism,
                "tacit_explicit": self.methodological.tacit_explicit,
            },
            "social": {
                "disclosure": self.social.disclosure,
                "inquiry": self.social.inquiry,
                "collaboration": self.social.collaboration,
                "competition": self.social.competition,
            },
            "temporal": {
                "paradigm_orientation": self.temporal.paradigm_orientation,
                "patience": self.temporal.patience,
                "archival": self.temporal.archival,
            },
            "aesthetic": {
                "elegance_completeness": self.aesthetic.elegance_completeness,
                "mechanism_phenomenon": self.aesthetic.mechanism_phenomenon,
                "risk": self.aesthetic.risk,
            },
        }

    def describe(self) -> str:
        """Generate natural language description of this disposition."""
        parts = []

        # Epistemic appetite
        style = self.epistemic_appetite.curiosity_style.value
        if style == "busybody":
            parts.append("broad collector of ideas across domains")
        elif style == "hunter":
            parts.append("focused pursuer of specific questions")
        else:
            parts.append("leaper between unexpected domains")

        if self.epistemic_appetite.scope > 0.7:
            parts.append("committed to one big unifying idea")
        elif self.epistemic_appetite.scope < 0.3:
            parts.append("draws from many frameworks as needed")

        if self.epistemic_appetite.surprise_seeking > 0.7:
            parts.append("seeks anomalies and contradictions")
        elif self.epistemic_appetite.surprise_seeking < 0.3:
            parts.append("values confirmation and precision")

        # Cognitive style
        if self.cognitive_style.abstraction > 0.7:
            parts.append("flies high to see patterns")
        elif self.cognitive_style.abstraction < 0.3:
            parts.append("stays close to concrete details")

        if self.cognitive_style.thinking_mode > 0.7:
            parts.append("generates many possibilities")
        elif self.cognitive_style.thinking_mode < 0.3:
            parts.append("converges on best answer")

        # Methodological
        if self.methodological.making_style > 0.7:
            parts.append("improvises with what's at hand")
        elif self.methodological.making_style < 0.3:
            parts.append("builds systematically from first principles")

        if self.methodological.formalism > 0.7:
            parts.append("demands mathematical precision")
        elif self.methodological.formalism < 0.3:
            parts.append("prefers narrative and qualitative richness")

        # Temporal
        if self.temporal.paradigm_orientation > 0.7:
            parts.append("challenges foundational assumptions")
        elif self.temporal.paradigm_orientation < 0.3:
            parts.append("works within established paradigms")

        if self.temporal.patience > 0.7:
            parts.append("invests for the long term")
        elif self.temporal.patience < 0.3:
            parts.append("iterates quickly")

        return ", ".join(parts)


@dataclass
class QuestionTypes:
    """Distribution of question types this voice tends to ask."""
    clarifying: float = 0.125    # Seeks definition, scope
    challenging: float = 0.125   # Questions assumptions, validity
    building: float = 0.125      # Extends, adds to the idea
    reframing: float = 0.125     # Offers alternative lens
    connecting: float = 0.125    # Links to other domains
    scaling: float = 0.125       # Asks about scale effects
    mechanistic: float = 0.125   # Asks how it works
    historical: float = 0.125    # References past work

    @classmethod
    def from_dict(cls, d: dict) -> "QuestionTypes":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return {
            "clarifying": self.clarifying,
            "challenging": self.challenging,
            "building": self.building,
            "reframing": self.reframing,
            "connecting": self.connecting,
            "scaling": self.scaling,
            "mechanistic": self.mechanistic,
            "historical": self.historical,
        }


@dataclass
class CritiqueBehavior:
    """Observable critique patterns derived from disposition."""
    question_types: QuestionTypes = field(default_factory=QuestionTypes)
    critique_patterns: List[str] = field(default_factory=list)
    signature_moves: List[str] = field(default_factory=list)
    productive_tensions: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "CritiqueBehavior":
        return cls(
            question_types=QuestionTypes.from_dict(d.get("question_types", {})),
            critique_patterns=d.get("critique_patterns", []),
            signature_moves=d.get("signature_moves", []),
            productive_tensions=d.get("productive_tensions", []),
        )

    def to_dict(self) -> dict:
        return {
            "question_types": self.question_types.to_dict(),
            "critique_patterns": self.critique_patterns,
            "signature_moves": self.signature_moves,
            "productive_tensions": self.productive_tensions,
        }


@dataclass
class DataSources:
    """Sources used to build this profile."""
    internal: Dict = field(default_factory=dict)
    external: Dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "DataSources":
        return cls(
            internal=d.get("internal", {}),
            external=d.get("external", {}),
        )

    def to_dict(self) -> dict:
        return {"internal": self.internal, "external": self.external}


@dataclass
class CompoundType:
    """
    Named compound disposition pattern from the literature.

    Examples: "Seductive Scientist", "Quiet Revolutionary", "Playful Bridger"
    """
    name: str
    description: str
    exemplars: List[str]
    disposition: DispositionVector
    critique_behavior: CritiqueBehavior

    @classmethod
    def from_dict(cls, key: str, d: dict) -> "CompoundType":
        return cls(
            name=d.get("name", key),
            description=d.get("description", ""),
            exemplars=d.get("exemplars", []),
            disposition=DispositionVector.from_dict(d.get("disposition_signature", {})),
            critique_behavior=CritiqueBehavior.from_dict(d.get("critique_behavior", {})),
        )

    def to_prompt(self) -> str:
        """Generate prompt text describing this compound type."""
        return f"""**{self.name}**
{self.description}

Disposition: {self.disposition.describe()}

Characteristic critique patterns:
{chr(10).join(f'- "{p}"' for p in self.critique_behavior.critique_patterns[:4])}

Signature intellectual moves:
{chr(10).join(f'- {m}' for m in self.critique_behavior.signature_moves[:4])}
"""


@dataclass
class IndividualProfile:
    """
    Specific person with disposition profile.

    May be linked to a compound_type as a base, with individual variations.
    """
    id: str
    name: str
    affiliation: Optional[str] = None
    compound_type: Optional[str] = None
    disposition: DispositionVector = field(default_factory=DispositionVector)
    critique_behavior: CritiqueBehavior = field(default_factory=CritiqueBehavior)
    data_sources: DataSources = field(default_factory=DataSources)
    confidence: float = 0.0

    @classmethod
    def from_dict(cls, d: dict) -> "IndividualProfile":
        return cls(
            id=d.get("id", ""),
            name=d.get("name", ""),
            affiliation=d.get("affiliation"),
            compound_type=d.get("compound_type"),
            disposition=DispositionVector.from_dict(d.get("disposition", {})),
            critique_behavior=CritiqueBehavior.from_dict(d.get("critique_behavior", {})),
            data_sources=DataSources.from_dict(d.get("data_sources", {})),
            confidence=d.get("confidence", 0.0),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "affiliation": self.affiliation,
            "compound_type": self.compound_type,
            "disposition": self.disposition.to_dict(),
            "critique_behavior": self.critique_behavior.to_dict(),
            "data_sources": self.data_sources.to_dict(),
            "confidence": self.confidence,
        }

    def to_prompt(self) -> str:
        """Generate prompt text for this individual voice."""
        base = f"**{self.name}**"
        if self.affiliation:
            base += f" ({self.affiliation})"
        base += "\n"

        base += f"Disposition: {self.disposition.describe()}\n\n"

        if self.critique_behavior.critique_patterns:
            base += "Characteristic questions:\n"
            base += "\n".join(f'- "{p}"' for p in self.critique_behavior.critique_patterns[:3])
            base += "\n\n"

        if self.critique_behavior.signature_moves:
            base += "Signature moves:\n"
            base += "\n".join(f'- {m}' for m in self.critique_behavior.signature_moves[:3])

        return base
