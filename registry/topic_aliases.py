"""
Topic Aliases Module

Maps abbreviations and synonyms to canonical forms for better topic search.

Bidirectional matching: searching "NLP" finds "natural language processing"
and searching "natural language processing" also matches topics containing "NLP"
"""

from typing import Dict, List, Set


# =============================================================================
# Topic Aliases: Map abbreviations and synonyms to canonical forms
# =============================================================================

TOPIC_ALIASES: Dict[str, List[str]] = {
    # NLP / Computational Linguistics
    "nlp": ["natural language processing", "computational linguistics", "text analysis", "text mining"],
    "natural language processing": ["nlp", "computational linguistics"],

    # Machine Learning / AI (comprehensive cross-linking)
    "ml": ["machine learning", "statistical learning", "ai", "artificial intelligence", "deep learning", "neural network"],
    "machine learning": ["ml", "ai", "artificial intelligence", "deep learning", "neural network", "statistical learning"],
    "ai": ["artificial intelligence", "machine intelligence", "machine learning", "ml", "neural network"],
    "artificial intelligence": ["ai", "machine learning", "ml", "neural network"],
    "dl": ["deep learning", "neural networks", "machine learning", "ml"],
    "deep learning": ["dl", "neural networks", "machine learning", "ml"],
    "neural network": ["neural networks", "deep learning", "machine learning", "ml", "ai"],
    "neural networks": ["neural network", "deep learning", "machine learning", "ml", "ai"],
    "xai": ["explainable artificial intelligence", "explainable ai", "interpretable ml"],
    "explainable ai": ["xai"],

    # Computer Vision
    "cv": ["computer vision", "image processing", "visual computing"],
    "computer vision": ["cv"],

    # Human-Computer Interaction
    "hci": ["human-computer interaction", "human computer interaction", "user experience", "ux"],
    "human-computer interaction": ["hci"],
    "ux": ["user experience", "hci"],

    # Computational Social Science
    "css": ["computational social science", "social computing"],
    "computational social science": ["css"],

    # Network Science
    "network science": ["networks", "social networks", "network analysis", "complex networks", "graph theory"],
    "networks": ["network science", "network analysis"],
    "social networks": ["network science", "social network analysis"],
    "sna": ["social network analysis", "network analysis"],

    # Data Science
    "data science": ["data analytics", "data mining", "big data"],
    "big data": ["data science", "large-scale data"],

    # Information Retrieval
    "ir": ["information retrieval", "search engines"],
    "information retrieval": ["ir"],

    # Science of Science
    "science of science": ["scientometrics", "bibliometrics", "metascience"],
    "scientometrics": ["science of science", "bibliometrics"],
    "bibliometrics": ["scientometrics", "science of science"],

    # Economics
    "econ": ["economics", "economic"],
    "economics": ["econ"],

    # Statistics
    "stats": ["statistics", "statistical"],
    "statistics": ["stats"],

    # Reinforcement Learning
    "rl": ["reinforcement learning"],
    "reinforcement learning": ["rl"],

    # Natural Language Understanding/Generation
    "nlu": ["natural language understanding"],
    "nlg": ["natural language generation"],
    "llm": ["large language models", "language models"],
    "large language models": ["llm"],

    # Topic Modeling
    "lda": ["latent dirichlet allocation", "topic modeling"],
    "topic modeling": ["lda", "topic models"],

    # Other common abbreviations
    "hpc": ["high performance computing", "parallel computing", "distributed computing"],
    "high performance computing": ["hpc"],
    "iot": ["internet of things"],
    "internet of things": ["iot"],
}


def expand_topic_aliases(topic: str) -> Set[str]:
    """
    Expand a topic search term to include all its aliases.

    Args:
        topic: The original topic search term

    Returns:
        Set of all terms to search for (including the original)
    """
    topic_lower = topic.lower().strip()
    expanded = {topic_lower}

    # Check if this topic has direct aliases
    if topic_lower in TOPIC_ALIASES:
        expanded.update(alias.lower() for alias in TOPIC_ALIASES[topic_lower])

    # Also check if any alias key contains this topic as a substring
    # This handles partial matches like "network" matching "network science"
    for key, aliases in TOPIC_ALIASES.items():
        # If query is part of an alias key
        if topic_lower in key:
            expanded.add(key)
            expanded.update(alias.lower() for alias in aliases)
        # If query matches one of the aliases
        for alias in aliases:
            if topic_lower == alias.lower() or topic_lower in alias.lower():
                expanded.add(key)
                expanded.update(a.lower() for a in aliases)

    return expanded
