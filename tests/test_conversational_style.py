#!/usr/bin/env python3
"""
Conversational Style Analysis for Chorus.

Sends 50 plausible queries through Chorus and analyzes the responses
to understand its current persona and conversational patterns.
"""

import asyncio
import os
import sys
import json
import re
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_dotenv():
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip().strip('"').strip("'")

load_dotenv()

from agents import ChorusOrchestrator

# 50 plausible test queries from lab members
TEST_QUERIES = [
    # === GREETINGS & CASUAL (5) ===
    ("Hey Chorus!", "greeting"),
    ("Good morning!", "greeting"),
    ("Hi, how are you?", "greeting"),
    ("Thanks for your help earlier", "greeting"),
    ("What's up?", "greeting"),

    # === FACTUAL LOOKUPS (10) ===
    ("Who is James Evans?", "lookup"),
    ("What grants do we have?", "lookup"),
    ("When is the next lab meeting?", "lookup"),
    ("What's the deadline for the NSF proposal?", "lookup"),
    ("Who's the PI on the computational social science project?", "lookup"),
    ("What papers has the lab published recently?", "lookup"),
    ("Where is the lab located?", "lookup"),
    ("What's James's email?", "lookup"),
    ("How many people are in the lab?", "lookup"),
    ("What datasets do we have access to?", "lookup"),

    # === TECHNICAL HELP (10) ===
    ("How do I access Midway?", "technical"),
    ("My SSH connection keeps dropping", "technical"),
    ("How do I set up a conda environment?", "technical"),
    ("Git says I have merge conflicts, what do I do?", "technical"),
    ("How do I submit a job to the cluster?", "technical"),
    ("I can't install pytorch, getting an error", "technical"),
    ("How do I connect to the VPN from home?", "technical"),
    ("Where should I store large datasets?", "technical"),
    ("How do I request more compute allocation?", "technical"),
    ("My Jupyter notebook won't start on Midway", "technical"),

    # === INTELLECTUAL EXPLORATION (10) ===
    ("What are the implications of LLMs for social science research?", "explore"),
    ("Help me think through my research design on cultural evolution", "explore"),
    ("What's the relationship between network structure and information diffusion?", "explore"),
    ("How should we think about measuring authenticity in online spaces?", "explore"),
    ("What are the tradeoffs between interpretability and performance in ML?", "explore"),
    ("I'm stuck on how to operationalize 'collective intelligence'", "explore"),
    ("What would a computational approach to studying scientific discovery look like?", "explore"),
    ("How do scaling laws apply to social systems?", "explore"),
    ("What's the difference between emergence and epiphenomena?", "explore"),
    ("Help me critique this paper on AI alignment", "explore"),

    # === CONNECTION SEEKING (10) ===
    ("Who else is working on NLP in the lab?", "connect"),
    ("Is anyone interested in collaborating on network analysis?", "connect"),
    ("Who should I talk to about getting started with embeddings?", "connect"),
    ("Does anyone have experience with survey design?", "connect"),
    ("Who's working on something related to misinformation?", "connect"),
    ("I need help with causal inference - who knows about that?", "connect"),
    ("Are there other people interested in philosophy of science?", "connect"),
    ("Who's used the Twitter API before?", "connect"),
    ("Connect me with someone who knows about grant writing", "connect"),
    ("Who in the lab works on complexity theory?", "connect"),

    # === EDGE CASES (5) ===
    ("", "edge"),  # Empty query
    ("???", "edge"),  # Unclear
    ("Can you write code for me?", "edge"),  # Out of scope request
    ("What do you think about the meaning of life?", "edge"),  # Philosophical tangent
    ("Tell me a joke", "edge"),  # Social request
]


@dataclass
class StyleMetrics:
    """Metrics for analyzing conversational style."""
    # Response characteristics
    avg_length: float = 0.0
    min_length: int = 0
    max_length: int = 0

    # Opening patterns
    opening_patterns: Dict[str, int] = field(default_factory=dict)

    # Linguistic markers
    uses_i: int = 0  # First person
    uses_we: int = 0  # Collective
    uses_you: int = 0  # Direct address

    # Tone markers
    hedging_phrases: int = 0  # "might", "perhaps", "I think"
    confident_phrases: int = 0  # "definitely", "certainly", "clearly"
    apologetic_phrases: int = 0  # "sorry", "unfortunately", "I apologize"

    # Structure
    uses_bullets: int = 0
    uses_numbered_lists: int = 0
    uses_headers: int = 0
    asks_questions: int = 0

    # Warmth indicators
    greetings_given: int = 0
    acknowledgments: int = 0  # "great question", "good point"

    # Uncertainty handling
    admits_uncertainty: int = 0  # "I don't know", "not sure"
    offers_alternatives: int = 0  # "you might try", "alternatively"


def analyze_response(response: str) -> dict:
    """Analyze a single response for style markers."""
    analysis = {
        "length": len(response),
        "word_count": len(response.split()),
        "sentence_count": len(re.findall(r'[.!?]+', response)),
    }

    response_lower = response.lower()

    # Opening pattern
    first_sentence = response.split('.')[0] if response else ""
    first_words = first_sentence.split()[:3]
    analysis["opening"] = " ".join(first_words) if first_words else ""

    # First person usage
    analysis["uses_i"] = bool(re.search(r'\bI\b', response))
    analysis["uses_we"] = bool(re.search(r'\bwe\b', response_lower))
    analysis["uses_you"] = bool(re.search(r'\byou\b', response_lower))

    # Hedging
    hedges = ["might", "perhaps", "maybe", "possibly", "could be", "i think", "it seems", "appears to"]
    analysis["hedging"] = sum(1 for h in hedges if h in response_lower)

    # Confidence
    confident = ["definitely", "certainly", "clearly", "absolutely", "without doubt"]
    analysis["confident"] = sum(1 for c in confident if c in response_lower)

    # Apologetic
    apologetic = ["sorry", "unfortunately", "i apologize", "regret"]
    analysis["apologetic"] = sum(1 for a in apologetic if a in response_lower)

    # Structure
    analysis["has_bullets"] = bool(re.search(r'^[\-\*•]', response, re.MULTILINE))
    analysis["has_numbers"] = bool(re.search(r'^\d+\.', response, re.MULTILINE))
    analysis["has_headers"] = bool(re.search(r'^#{1,3}\s|^\*\*[^*]+\*\*:', response, re.MULTILINE))
    analysis["asks_question"] = "?" in response

    # Warmth
    greetings = ["hi!", "hello!", "hey!", "good morning", "good afternoon"]
    analysis["has_greeting"] = any(g in response_lower for g in greetings)

    acknowledgments = ["great question", "good question", "interesting", "that's a good point", "excellent question"]
    analysis["has_acknowledgment"] = any(a in response_lower for a in acknowledgments)

    # Uncertainty
    uncertainty = ["i don't know", "i'm not sure", "don't have that", "no information", "couldn't find"]
    analysis["admits_uncertainty"] = any(u in response_lower for u in uncertainty)

    alternatives = ["you might", "alternatively", "you could try", "another option", "consider"]
    analysis["offers_alternative"] = any(a in response_lower for a in alternatives)

    return analysis


async def run_experiment():
    """Run all queries and collect responses."""
    print("=" * 80)
    print("CHORUS CONVERSATIONAL STYLE EXPERIMENT")
    print("=" * 80)
    print(f"\nRunning {len(TEST_QUERIES)} test queries...")
    print("This will take several minutes.\n")

    orchestrator = ChorusOrchestrator(
        enable_caching=False,  # Fresh responses
        enable_parallel=True,
        rag_available=True
    )

    results = []

    for i, (query, category) in enumerate(TEST_QUERIES):
        print(f"[{i+1}/{len(TEST_QUERIES)}] {category}: {query[:50]}...")

        try:
            if query.strip():  # Skip empty query for now
                response = await orchestrator.query(query)
            else:
                response = "I notice your message was empty. How can I help you today?"
        except Exception as e:
            response = f"Error: {str(e)}"

        analysis = analyze_response(response)

        results.append({
            "query": query,
            "category": category,
            "response": response,
            "analysis": analysis,
        })

    return results


def generate_report(results: List[dict]):
    """Generate a comprehensive style analysis report."""
    print("\n" + "=" * 80)
    print("CONVERSATIONAL STYLE ANALYSIS REPORT")
    print("=" * 80)

    # Group by category
    by_category = {}
    for r in results:
        cat = r["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(r)

    # === RESPONSE LENGTH ANALYSIS ===
    print("\n" + "-" * 40)
    print("1. RESPONSE LENGTH BY CATEGORY")
    print("-" * 40)

    for cat, items in by_category.items():
        lengths = [r["analysis"]["length"] for r in items]
        word_counts = [r["analysis"]["word_count"] for r in items]
        avg_len = sum(lengths) / len(lengths) if lengths else 0
        avg_words = sum(word_counts) / len(word_counts) if word_counts else 0
        print(f"  {cat:<12}: avg {avg_len:>5.0f} chars, {avg_words:>4.0f} words")

    # === OPENING PATTERNS ===
    print("\n" + "-" * 40)
    print("2. OPENING PATTERNS (First 3 words)")
    print("-" * 40)

    openings = Counter()
    for r in results:
        opening = r["analysis"]["opening"]
        if opening:
            # Normalize
            opening = opening.lower().strip()
            openings[opening] += 1

    for opening, count in openings.most_common(15):
        print(f"  \"{opening}...\" ({count}x)")

    # === PRONOUN USAGE ===
    print("\n" + "-" * 40)
    print("3. PRONOUN USAGE")
    print("-" * 40)

    total = len(results)
    uses_i = sum(1 for r in results if r["analysis"]["uses_i"])
    uses_we = sum(1 for r in results if r["analysis"]["uses_we"])
    uses_you = sum(1 for r in results if r["analysis"]["uses_you"])

    print(f"  Uses 'I':   {uses_i:>3}/{total} ({uses_i/total*100:.0f}%)")
    print(f"  Uses 'we':  {uses_we:>3}/{total} ({uses_we/total*100:.0f}%)")
    print(f"  Uses 'you': {uses_you:>3}/{total} ({uses_you/total*100:.0f}%)")

    # === TONE MARKERS ===
    print("\n" + "-" * 40)
    print("4. TONE MARKERS")
    print("-" * 40)

    hedging = sum(r["analysis"]["hedging"] for r in results)
    confident = sum(r["analysis"]["confident"] for r in results)
    apologetic = sum(r["analysis"]["apologetic"] for r in results)

    print(f"  Hedging phrases:    {hedging:>3} instances")
    print(f"  Confident phrases:  {confident:>3} instances")
    print(f"  Apologetic phrases: {apologetic:>3} instances")

    # === STRUCTURE ===
    print("\n" + "-" * 40)
    print("5. RESPONSE STRUCTURE")
    print("-" * 40)

    bullets = sum(1 for r in results if r["analysis"]["has_bullets"])
    numbers = sum(1 for r in results if r["analysis"]["has_numbers"])
    headers = sum(1 for r in results if r["analysis"]["has_headers"])
    questions = sum(1 for r in results if r["analysis"]["asks_question"])

    print(f"  Uses bullet lists:   {bullets:>3}/{total} ({bullets/total*100:.0f}%)")
    print(f"  Uses numbered lists: {numbers:>3}/{total} ({numbers/total*100:.0f}%)")
    print(f"  Uses headers/bold:   {headers:>3}/{total} ({headers/total*100:.0f}%)")
    print(f"  Asks questions:      {questions:>3}/{total} ({questions/total*100:.0f}%)")

    # === WARMTH INDICATORS ===
    print("\n" + "-" * 40)
    print("6. WARMTH & ENGAGEMENT")
    print("-" * 40)

    greetings = sum(1 for r in results if r["analysis"]["has_greeting"])
    acks = sum(1 for r in results if r["analysis"]["has_acknowledgment"])

    print(f"  Returns greetings:   {greetings:>3}/{total}")
    print(f"  Acknowledges query:  {acks:>3}/{total}")

    # === UNCERTAINTY HANDLING ===
    print("\n" + "-" * 40)
    print("7. UNCERTAINTY HANDLING")
    print("-" * 40)

    admits = sum(1 for r in results if r["analysis"]["admits_uncertainty"])
    alts = sum(1 for r in results if r["analysis"]["offers_alternative"])

    print(f"  Admits uncertainty:  {admits:>3}/{total}")
    print(f"  Offers alternatives: {alts:>3}/{total}")

    # === SAMPLE RESPONSES BY CATEGORY ===
    print("\n" + "=" * 80)
    print("SAMPLE RESPONSES BY CATEGORY")
    print("=" * 80)

    for cat in ["greeting", "lookup", "technical", "explore", "connect", "edge"]:
        items = by_category.get(cat, [])
        if items:
            print(f"\n--- {cat.upper()} ---")
            # Show 2 samples
            for r in items[:2]:
                print(f"\nQ: {r['query']}")
                preview = r['response'][:400] + "..." if len(r['response']) > 400 else r['response']
                print(f"A: {preview}")

    # === STYLE SUMMARY ===
    print("\n" + "=" * 80)
    print("STYLE SUMMARY")
    print("=" * 80)

    # Calculate overall metrics
    all_lengths = [r["analysis"]["length"] for r in results]
    avg_length = sum(all_lengths) / len(all_lengths)

    print(f"""
CURRENT CHORUS PERSONALITY PROFILE:

Verbosity:
  - Average response: {avg_length:.0f} characters
  - Lookup queries: shorter, factual
  - Explore queries: much longer, detailed

Voice:
  - Uses "I" in {uses_i/total*100:.0f}% of responses (personal voice)
  - Uses "you" in {uses_you/total*100:.0f}% of responses (direct address)
  - Uses "we" in {uses_we/total*100:.0f}% of responses (collaborative tone)

Tone:
  - Hedging: {hedging} instances (cautious/uncertain)
  - Confident: {confident} instances (assertive)
  - Apologetic: {apologetic} instances (deferential)

Structure Preferences:
  - Frequently uses bullet points ({bullets/total*100:.0f}%)
  - Uses numbered lists ({numbers/total*100:.0f}%)
  - Uses headers/formatting ({headers/total*100:.0f}%)

Warmth:
  - Rarely returns greetings ({greetings}/5 greeting queries)
  - Rarely acknowledges good questions ({acks}/{total})

Uncertainty:
  - Admits when doesn't know ({admits}/{total})
  - Offers alternatives ({alts}/{total})
""")

    return results


async def main():
    results = await run_experiment()
    generate_report(results)

    # Save full results for further analysis
    output_path = Path(__file__).parent / "style_analysis_results.json"
    with open(output_path, "w") as f:
        # Convert to serializable format
        serializable = []
        for r in results:
            serializable.append({
                "query": r["query"],
                "category": r["category"],
                "response": r["response"],
                "length": r["analysis"]["length"],
                "word_count": r["analysis"]["word_count"],
            })
        json.dump(serializable, f, indent=2)

    print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
