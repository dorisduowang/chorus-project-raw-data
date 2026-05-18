"""
Query Classifier Module

Classifies queries to determine optimal retrieval strategy.

Query Types Supported:
- Research interest queries: "What does X research?", "X's research interests"
- Contact queries: "How to contact X?", "X's email"
- Collaboration queries: "Who collaborates with X?"
- Comparison queries: "Compare X and Y's research"
- Stats queries: "How many publications does X have?"
- Topic queries: "Who works on network analysis?"
- Role queries: "List all PhD students"
- Institution queries: "Who is at University of Chicago?"
- Publication queries: "Papers by X", "Publications about Y"
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set

# Handle both package and direct imports for testing flexibility
try:
    from .models import QueryClassification
except ImportError:
    from models import QueryClassification


class QueryClassifier:
    """
    Classify queries to determine optimal retrieval strategy.

    Structured queries: direct registry lookups
    Semantic queries: embedding-based search
    Hybrid: both

    Query Types Supported:
    - Research interest queries: "What does X research?", "X's research interests"
    - Contact queries: "How to contact X?", "X's email"
    - Collaboration queries: "Who collaborates with X?"
    - Comparison queries: "Compare X and Y's research"
    - Stats queries: "How many publications does X have?"
    - Topic queries: "Who works on network analysis?"
    - Role queries: "List all PhD students"
    - Institution queries: "Who is at University of Chicago?"
    - Publication queries: "Papers by X", "Publications about Y"
    """

    # Dynamically expanded keywords (populated in __init__)
    _expanded_keywords: Set[str] = set()

    def __init__(
        self,
        registry_path: str = "Data/lab_registry.json",
        midway_path: str = "Data/midway_registry.json",
        jevans_path: str = "Data/jevans_resources.json",
    ):
        """Initialize classifier and expand keywords from registry."""
        self._expand_keywords_from_registry(registry_path, midway_path, jevans_path)

    def _expand_keywords_from_registry(self, registry_path: str, midway_path: str, jevans_path: str = ""):
        """Dynamically add keywords from registry data."""
        # Load lab registry
        lab_path = Path(registry_path)
        if lab_path.exists():
            with open(lab_path) as f:
                registry = json.load(f)

            # Add person names (first and last)
            for person in registry.get("people", {}).get("members", []):
                name = person.get("name", "")
                for part in name.lower().split():
                    if len(part) >= 3:
                        self._expanded_keywords.add(part)

            # Add project names
            for project in registry.get("projects", []):
                name = project.get("name", "").lower()
                self._expanded_keywords.add(name)

            # Add dataset names
            for dataset in registry.get("datasets", []):
                name = dataset.get("name", "").lower()
                for word in name.split():
                    if len(word) >= 3:
                        self._expanded_keywords.add(word)

            # Add topics from people
            for person in registry.get("people", {}).get("members", []):
                topics = person.get("openalex", {}).get("topics", [])
                for topic in topics:
                    if isinstance(topic, str):
                        for word in topic.lower().split():
                            if len(word) >= 4:
                                self._expanded_keywords.add(word)

        # Load midway registry
        midway_path_obj = Path(midway_path)
        if midway_path_obj.exists():
            with open(midway_path_obj) as f:
                midway = json.load(f)

            # Add from all categories
            for category in ["data_snapshots", "embeddings", "precomputed_resources",
                           "social_media_datasets", "researcher_directories"]:
                items = midway.get(category, {})
                if isinstance(items, dict) and "items" in items:
                    items = items["items"]
                elif not isinstance(items, list):
                    items = []

                for item in items:
                    # Add name words
                    name = item.get("name", "")
                    for word in name.lower().split():
                        if len(word) >= 3:
                            self._expanded_keywords.add(word)

                    # Add platform names for social media
                    platform = item.get("platform", "")
                    if isinstance(platform, str):
                        self._expanded_keywords.add(platform.lower())
                    elif isinstance(platform, list):
                        for p in platform:
                            self._expanded_keywords.add(p.lower())

        # Load jevans resources
        jevans_path_obj = Path(jevans_path) if jevans_path else None
        if jevans_path_obj and jevans_path_obj.exists():
            with open(jevans_path_obj) as f:
                jevans = json.load(f)

            for resource in jevans.get("resources", []):
                # Add resource names
                name = resource.get("name", "")
                if len(name) >= 3:
                    self._expanded_keywords.add(name.lower())

                # Add owner usernames
                owner = resource.get("owner_username", "")
                if owner and len(owner) >= 3:
                    self._expanded_keywords.add(owner.lower())

                # Add words from descriptions
                desc = resource.get("description", "") or ""
                for word in desc.lower().split():
                    # Only add meaningful words (skip common words)
                    if len(word) >= 4 and word.isalpha():
                        self._expanded_keywords.add(word)

        print(f"Classifier expanded with {len(self._expanded_keywords)} keywords from registry")

    # Common title prefixes that appear before names
    TITLE_PREFIXES = ["dr", "dr.", "prof", "prof.", "professor", "mr", "mr.", "mrs", "mrs.", "ms", "ms."]

    # Patterns that indicate structured queries
    # Each tuple contains: (regex_pattern, entity_types, query_subtype)
    # More specific patterns come BEFORE general patterns to take precedence
    STRUCTURED_PATTERNS = [
        # === Comparison queries (highest priority - involve multiple people) ===
        (r"compare\s+(.+?)\s+(?:and|with|to|vs\.?)\s+(.+?)(?:'s)?\s*(?:research|work|publications?|interests?)?", ["people", "comparison"], "comparison"),
        (r"(?:what(?:'s| is| are) the )?(?:difference|similarities?)\s+between\s+(.+?)\s+and\s+(.+?)(?:'s)?\s*(?:research|work)?", ["people", "comparison"], "comparison"),
        (r"how (?:does?|do)\s+(.+?)\s+(?:and|&)\s+(.+?)\s+(?:compare|differ|relate)", ["people", "comparison"], "comparison"),

        # === Stats/metrics queries ===
        (r"how many (?:publications?|papers?|articles?|works?)\s+(?:does?|has|have)\s+(.+?)(?:\s+(?:have|published|written))?", ["people", "stats"], "publication_count"),
        (r"(?:what(?:'s| is| are)|tell me)\s+(.+?)(?:'s)?\s+(?:citation count|citations?|h-?index|impact)", ["people", "stats"], "citations"),
        (r"(.+?)(?:'s)?\s+(?:publication|paper|article)\s+(?:count|number|statistics?|stats?)", ["people", "stats"], "publication_count"),
        (r"how (?:cited|impactful|productive)\s+is\s+(.+)", ["people", "stats"], "citations"),
        (r"(?:what(?:'s| is| are)|show|get)\s+(.+?)(?:'s)?\s+h-?index", ["people", "stats"], "h_index"),
        (r"how many (?:citations?|papers?|publications?)\s+(?:does?|has)\s+(.+?)(?:\s+have)?", ["people", "stats"], "citations"),

        # === Contact queries ===
        (r"(?:how (?:do i|can i|to))\s+(?:contact|reach|email|get in touch with)\s+(.+)", ["people", "contact"], "contact"),
        (r"(?:what(?:'s| is))\s+(.+?)(?:'s)?\s+(?:email|contact|phone|office)", ["people", "contact"], "contact"),
        (r"(.+?)(?:'s)?\s+(?:email|contact\s+info(?:rmation)?|phone|office)", ["people", "contact"], "contact"),
        (r"(?:email|contact)\s+(?:for|of)\s+(.+)", ["people", "contact"], "contact"),
        (r"(?:can you |please )?(?:give|send|share|provide)\s+(?:me\s+)?(.+?)(?:'s)?\s+(?:email|contact)", ["people", "contact"], "contact"),

        # === Research interest queries ===
        (r"what\s+(?:does?|is)\s+(.+?)\s+(?:research(?:ing)?|study(?:ing)?|work(?:ing)?\s+on|interested\s+in)", ["people", "topics"], "research_interests"),
        (r"(.+?)(?:'s)?\s+(?:research\s+)?(?:interests?|focus|expertise|specialt(?:y|ies)|area(?:s)?)", ["people", "topics"], "research_interests"),
        (r"what\s+(?:are|is)\s+(.+?)(?:'s)?\s+(?:research\s+)?(?:interests?|topics?|areas?|focus)", ["people", "topics"], "research_interests"),
        (r"(?:tell me about|describe)\s+(.+?)(?:'s)?\s+research", ["people", "topics"], "research_interests"),
        (r"what\s+(?:topics?|areas?)\s+(?:does?|is)\s+(.+?)\s+(?:work(?:ing)?\s+(?:on|in)|focus(?:ed|ing)?\s+on)", ["people", "topics"], "research_interests"),

        # === Collaboration queries ===
        (r"who\s+(?:does?|has)\s+(.+?)\s+(?:collaborate|work|publish|co-?author)\s+with", ["people", "collaboration"], "collaborators"),
        (r"who\s+(?:collaborates?|works?|publishes?)\s+with\s+(.+)", ["people", "collaboration"], "collaborators"),
        (r"(.+?)(?:'s)?\s+(?:collaborators?|co-?authors?|research\s+partners?)", ["people", "collaboration"], "collaborators"),
        (r"(?:find|list|show)\s+(?:collaborators?|co-?authors?)\s+(?:of|for)\s+(.+)", ["people", "collaboration"], "collaborators"),
        (r"(?:who|which\s+(?:people|researchers?))\s+(?:has|have)\s+(.+?)\s+(?:collaborated|worked|published)\s+with", ["people", "collaboration"], "collaborators"),

        # === Publication/paper search patterns ===
        (r"what has (.+?) published", ["publications"], "publication_search"),
        (r"(?:find|search|show|list) (?:papers?|publications?) (?:about|on|regarding)\s+(.+)", ["publications"], "publication_search"),
        (r"(?:papers?|publications?) (?:about|on|by)\s+(.+)", ["publications"], "publication_search"),
        (r"(?:papers?|publications?) by (.+)", ["publications"], "publication_search"),
        (r"(.+?)(?:'s|s') (?:papers?|publications?|recent work)", ["publications"], "publication_search"),
        (r"(?:find|search|show|list) (.+?) (?:papers?|publications?)", ["publications"], "publication_search"),
        (r"(?:research|work) published (?:by|on|about)\s+(.+)", ["publications"], "publication_search"),
        (r"recent (?:papers?|publications?|work) (?:by|from)\s+(.+)", ["publications"], "publication_search"),

        # === Topic-based search patterns ===
        (r"who\s+(?:works\s+on|studies|researches?|is\s+(?:working|researching)\s+on)\s+(?!apto|the\s+apto|c3s2)(.+?)(?:\?|$)", ["people", "topics"], "topic_search"),
        (r"(?:find|list|show)?\s*(?:researchers?|people|members?|scientists?|academics?)\s+(?:studying|working\s+on|researching|interested\s+in)\s+(.+)", ["people", "topics"], "topic_search"),
        (r"(?:researchers?|people|members?)\s+(?:who\s+work|working)\s+on\s+(.+)", ["people", "topics"], "topic_search"),
        (r"anyone\s+(?:working\s+on|studying|researching|interested\s+in)\s+(.+)", ["people", "topics"], "topic_search"),
        (r"(?:find|list|show)\s+(?:researchers?|people|members?)\s+(?:in|on|for)\s+(?:the\s+)?(.+?)\s+(?:topic|area|field)", ["people", "topics"], "topic_search"),
        (r"(?:experts?|specialists?)\s+(?:in|on)\s+(.+)", ["people", "topics"], "topic_search"),
        (r"(?:who|which\s+(?:people|researchers?))\s+(?:knows?|has\s+expertise\s+in|specializes?\s+in)\s+(.+)", ["people", "topics"], "topic_search"),

        # === Project patterns (only for known project names like APTO) ===
        (r"who\s+(?:works\s+on|is\s+on|is\s+part\s+of|leads?)\s+(apto|the\s+apto|c3s2|socio-cognitive)", ["people", "projects"], "project_members"),
        (r"(?:members?|team|people)\s+(?:on|in|of)\s+(?:the\s+)?(apto|c3s2|socio-cognitive)\s+(?:project|team)?", ["people", "projects"], "project_members"),

        # === Project-specific lookup patterns (must come BEFORE generic person lookup) ===
        # These patterns match known project names to prevent them from being classified as person lookups
        (r"(?:tell\s+me\s+about|describe|what\s+is)\s+(?:the\s+)?(apto|c3s2|socio-cognitive)(?:\s+project)?(?:\?|$)", ["projects"], "project_lookup"),
        (r"\b(apto|c3s2|socio-cognitive)\b.*(?:\?|$)", ["projects"], "project_lookup"),

        # === Data location queries (MUST come before generic person lookup) ===
        (r"(?:where|how)\s+(?:can\s+I\s+)?(?:find|get)\s+.*(?:chinese|weibo|sina)", ["midway_social"], "midway_social_lookup"),
        (r"(?:find|where|get|have)\s+.*(?:chinese|weibo|sina)\s+(?:social|data|post)", ["midway_social"], "midway_social_lookup"),

        # === Dataset/data card queries (MUST come before generic person lookup) ===
        # Known dataset names - high priority
        (r"(?:tell\s+me\s+about|describe|explain|what\s+is)\s+(?:the\s+)?(?:openalex|semantic\s+scholar|s2ag|mag|web\s+of\s+science|wos|reddit|patstat|dblp|pubmed|arxiv|biorxiv|medrxiv|dimensions|icite|papers\s+with\s+code|pwc|iris|umetrics)(?:\s+data(?:set)?)?(?:\?|$)", ["datasets"], "dataset_detail"),
        (r"(?:openalex|semantic\s+scholar|s2ag|mag|microsoft\s+academic|web\s+of\s+science|wos|reddit|patstat|dblp|pubmed|arxiv|biorxiv|medrxiv|dimensions|icite|papers\s+with\s+code|pwc|iris|umetrics)\s+(?:data(?:set)?|database|card)", ["datasets"], "dataset_detail"),

        # === Person lookup queries ===
        (r"who\s+is\s+(.+?)(?:\?|$)", ["people"], "person_lookup"),
        (r"(?:tell\s+me\s+about|describe|info(?:rmation)?\s+(?:on|about))\s+(.+?)(?:\?|$)", ["people"], "person_lookup"),
        (r"(?:find|look\s*up|search\s+for)\s+(.+?)(?:\?|$)", ["people"], "person_lookup"),

        # === Role-based queries ===
        (r"list\s+(?:all\s+)?(?:the\s+)?(phd\s+students?|postdocs?|faculty|staff|members?|people|researchers?|scientists?)", ["people"], "role_list"),
        (r"(?:show|get|find)\s+(?:all\s+)?(?:the\s+)?(phd\s+students?|postdocs?|faculty|staff|members?|people)", ["people"], "role_list"),
        (r"(?:who\s+are|how\s+many)\s+(?:the\s+)?(phd\s+students?|postdocs?|faculty|staff|members?)", ["people"], "role_list"),

        # === Project queries ===
        (r"what\s+projects?\s+(?:does?|is)\s+(.+?)\s+(?:work(?:ing)?\s+on|involved\s+(?:in|with))", ["people", "projects"], "person_projects"),
        (r"(.+?)(?:'s)?\s+projects?", ["people", "projects"], "person_projects"),
        (r"what\s+(?:datasets?|data)\s+(?:do\s+we\s+have|are\s+available|exist)", ["datasets"], "list_datasets"),
        (r"list\s+(?:all\s+)?datasets?", ["datasets"], "list_datasets"),
        (r"show\s+(?:all\s+)?datasets?", ["datasets"], "list_datasets"),
        # Data card specific queries
        (r"(?:tell\s+me\s+about|describe|explain)\s+(?:the\s+)?(.+?)\s+(?:data(?:set)?|database)", ["datasets"], "dataset_detail"),
        (r"(?:tell\s+me\s+about|describe|explain)\s+(?:the\s+)?(?:openalex|semantic\s+scholar|s2ag|mag|web\s+of\s+science|wos|reddit|patstat|dblp|pubmed|arxiv|biorxiv|medrxiv|dimensions|icite|papers\s+with\s+code|pwc|iris|umetrics)", ["datasets"], "dataset_detail"),
        (r"what\s+(?:are\s+)?(?:the\s+)?(?:variables|fields|columns|schema)\s+(?:in|for|of)\s+(?:the\s+)?(.+)", ["datasets"], "dataset_variables"),
        (r"what\s+(?:are\s+)?(?:the\s+)?limitations?\s+(?:of|for)\s+(?:the\s+)?(.+)", ["datasets"], "dataset_limitations"),
        (r"what\s+(?:are\s+)?(?:the\s+)?biases?\s+(?:in|of|for)\s+(?:the\s+)?(.+)", ["datasets"], "dataset_limitations"),
        (r"how\s+(?:do\s+i|should\s+i|to)\s+cite\s+(?:the\s+)?(.+)", ["datasets"], "dataset_citation"),
        (r"(?:what\s+is|where\s+is)\s+(?:the\s+)?(?:license|licensing)\s+(?:for|of)\s+(?:the\s+)?(.+)", ["datasets"], "dataset_license"),
        (r"what\s+(?:can\s+i|should\s+i)\s+use\s+(.+?)\s+(?:data(?:set)?|for)", ["datasets"], "dataset_use"),
        (r"(?:is|can)\s+(.+?)\s+(?:good|suitable|appropriate)\s+for\s+(.+)", ["datasets"], "dataset_suitability"),
        (r"what\s+(?:datasets?|data)\s+(?:(?:is|are)\s+)?(?:good|suitable|best)\s+for\s+(.+)", ["datasets"], "dataset_suitability"),
        (r"which\s+(?:datasets?|data)\s+(?:should\s+i\s+use|can\s+i\s+use)\s+for\s+(.+)", ["datasets"], "dataset_suitability"),
        (r"(?:data\s+card|datacard)\s+(?:for\s+)?(?:the\s+)?(.+)", ["datasets"], "dataset_detail"),
        (r"(?:documentation|docs?)\s+(?:for\s+)?(?:the\s+)?(.+?)\s+(?:data(?:set)?)", ["datasets"], "dataset_detail"),
        (r"what\s+(?:compute|resources?|clusters?)\s+(?:do\s+we\s+have|are\s+available)", ["compute"], "list_compute"),
        (r"what\s+(?:grants?|funding)\s+(?:do\s+we\s+have|are\s+active|is\s+available)", ["funding"], "list_funding"),
        (r"what\s+events?\s+(?:are|do\s+we\s+have|are\s+(?:there|scheduled|planned))", ["events"], "list_events"),
        (r"what\s+projects?\s+(?:do\s+we\s+have|are\s+there|exist)", ["projects"], "list_projects"),
        (r"(?:tell\s+me\s+about|describe|what\s+is)\s+(?:the\s+)?(.+?)\s+project", ["projects"], "project_lookup"),

        # === Count/stats queries ===
        (r"how\s+many\s+(people|members?|students?|postdocs?|faculty|researchers?|phd\s+students?)", ["people"], "count_people"),
        (r"(?:count|number\s+of)\s+(people|members?|students?|postdocs?|projects?)", ["people", "projects"], "count"),

        # === Institution queries ===
        (r"who\s+is\s+at\s+(.+?)(?:\?|$)", ["people"], "institution_lookup"),
        (r"(?:people|members?|researchers?)\s+(?:at|from)\s+(.+?)(?:\?|$)", ["people"], "institution_lookup"),

        # === Generic list queries ===
        (r"(?:list|show)\s+(?:all\s+)?(?:the\s+)?(projects?|grants?|funding|datasets?|events?|repositories?|repos?)", ["projects", "funding", "datasets", "events", "code_repos"], "list_entities"),

        # === How-to/Guide queries ===
        # Connection guides
        (r"how\s+(?:do\s+i|can\s+i|to)\s+(?:connect|ssh|log\s*in|access)\s+(?:to\s+)?(?:midway|the\s+cluster|hpc)", ["guides"], "guide_lookup"),
        (r"how\s+(?:do\s+i|can\s+i|to)\s+(?:connect|ssh|access)\s+(?:to\s+)?(?:knowledge\s+garden|k-?garden)", ["guides"], "guide_lookup"),
        # Job submission guides
        (r"how\s+(?:do\s+i|can\s+i|to)\s+(?:submit|run)\s+(?:a\s+)?(?:batch\s+)?job", ["guides"], "guide_lookup"),
        (r"how\s+(?:do\s+i|can\s+i|to)\s+(?:start|run|use)\s+(?:an?\s+)?interactive\s+(?:session|job)", ["guides"], "guide_lookup"),
        (r"how\s+(?:do\s+i|can\s+i|to)\s+use\s+(?:sbatch|sinteractive|slurm)", ["guides"], "guide_lookup"),
        # Resource requests
        (r"how\s+(?:do\s+i|can\s+i|to)\s+(?:request|use|get)\s+(?:a\s+)?gpu", ["guides"], "guide_lookup"),
        (r"how\s+(?:do\s+i|can\s+i|to)\s+(?:request|specify)\s+(?:resources?|memory|cpus?|cores?)", ["guides"], "guide_lookup"),
        # Software guides
        (r"how\s+(?:do\s+i|can\s+i|to)\s+(?:run|use|start|launch)\s+jupyter", ["guides"], "guide_lookup"),
        (r"how\s+(?:do\s+i|can\s+i|to)\s+(?:run|use|setup)\s+(?:py)?spark", ["guides"], "guide_lookup"),
        (r"how\s+(?:do\s+i|can\s+i|to)\s+install\s+(?:python\s+)?(?:packages?|libraries?|modules?)", ["guides"], "guide_lookup"),
        # Storage guides
        (r"how\s+(?:do\s+i|can\s+i|to)\s+check\s+(?:my\s+)?(?:storage|quota|disk\s+space)", ["guides"], "guide_lookup"),
        # Partition guides
        (r"how\s+(?:do\s+i|can\s+i|to)\s+(?:choose|pick|select)\s+(?:a\s+)?partition", ["guides"], "guide_lookup"),
        (r"how\s+(?:do\s+i|can\s+i|to)\s+check\s+(?:partition\s+)?availability", ["guides"], "guide_lookup"),
        (r"(?:which|what)\s+partition\s+should\s+i\s+use", ["guides"], "guide_lookup"),
        # Job array guides
        (r"how\s+(?:do\s+i|can\s+i|to)\s+(?:use|run|submit)\s+(?:a\s+)?job\s+array", ["guides"], "guide_lookup"),
        # General how-to pattern
        (r"how\s+(?:do\s+i|can\s+i|to)\s+(.+?)\s+(?:on\s+)?midway", ["guides"], "guide_lookup"),
        # Guide listing
        (r"(?:list|show|what)\s+(?:all\s+)?(?:guides?|tutorials?|how-?tos?)", ["guides"], "list_guides"),

        # === Midway storage queries ===
        # Data location queries
        (r"where\s+(?:is|are|can\s+i\s+find)\s+(?:the\s+)?(.+?)\s+(?:data|snapshot|database|dataset)(?:\?|$)", ["midway_snapshots"], "midway_location"),
        (r"where\s+(?:is|are)\s+(?:the\s+)?(.+?)(?:\s+stored)?(?:\s+on\s+midway)?(?:\?|$)", ["midway_snapshots", "midway_precomputed"], "midway_location"),
        (r"(?:path|location|directory)\s+(?:to|for|of)\s+(?:the\s+)?(.+)", ["midway_snapshots", "midway_precomputed"], "midway_location"),

        # Data snapshot queries
        (r"what\s+(?:data\s+)?snapshots?\s+(?:do\s+we\s+have|are\s+available|exist)(?:\s+on\s+midway)?", ["midway_snapshots"], "list_midway_snapshots"),
        (r"(?:list|show)\s+(?:all\s+)?(?:data\s+)?snapshots?(?:\s+on\s+midway)?", ["midway_snapshots"], "list_midway_snapshots"),
        (r"(?:latest|newest|most\s+recent)\s+(.+?)\s+(?:snapshot|data|version)", ["midway_snapshots"], "midway_latest_snapshot"),
        (r"when\s+(?:was|is)\s+(?:the\s+)?(.+?)\s+(?:snapshot|data)\s+(?:from|dated|updated)", ["midway_snapshots"], "midway_snapshot_date"),
        (r"(?:do\s+we\s+have|is\s+there)\s+(?:any\s+)?(?:openalex|semantic\s+scholar|mag|patstat|dblp|pubmed|wos|web\s+of\s+science)\s+(?:data|snapshot)?", ["midway_snapshots"], "midway_snapshot_lookup"),
        (r"(?:where|path|location)\s+(?:is|are|for|to)\s+(?:the\s+)?(?:openalex|semantic\s+scholar|mag|patstat|dblp|pubmed)", ["midway_snapshots"], "midway_snapshot_lookup"),

        # Precomputed resource queries
        (r"what\s+(?:precomputed|pre-computed|computed)\s+(?:resources?|data|assets?)\s+(?:do\s+we\s+have|exist|are\s+available)", ["midway_precomputed"], "list_midway_precomputed"),
        (r"(?:precomputed|pre-computed)\s+(.+?)(?:\s+data)?(?:\?|$)", ["midway_precomputed"], "midway_precomputed_lookup"),
        (r"(?:what|which)\s+(?:similarity\s+)?matrices?\s+(?:do\s+we\s+have|exist|are\s+available)", ["midway_precomputed"], "list_midway_precomputed"),

        # Embedding queries
        (r"what\s+embeddings?\s+(?:do\s+we\s+have|are\s+available|exist)(?:\s+on\s+midway)?", ["midway_embeddings"], "list_midway_embeddings"),
        (r"(?:where\s+(?:is|are)|find)\s+(?:the\s+)?(.+?)\s+embeddings?", ["midway_embeddings"], "midway_embedding_lookup"),
        (r"(?:patent|paper|mesh|openalex|specter|word2vec|mat2vec|hyperbolic)\s+embeddings?", ["midway_embeddings"], "midway_embedding_lookup"),
        (r"(?:do\s+we\s+have|is\s+there)\s+(?:any\s+)?(?:word2vec|specter|mat2vec|hyperbolic)\s+(?:embeddings?|vectors?|data)?", ["midway_embeddings"], "midway_embedding_lookup"),
        (r"(?:where|path|location)\s+(?:is|are|for|to)\s+(?:the\s+)?(?:word2vec|specter|mat2vec)", ["midway_embeddings"], "midway_embedding_lookup"),

        # Data availability queries (general pattern for "do we have X data")
        (r"(?:do\s+we\s+have|is\s+there|have)\s+(?:any\s+)?(?:arxiv|biorxiv|medrxiv|preprint)\s+(?:data|dataset)?", ["midway_snapshots"], "midway_snapshot_lookup"),
        (r"(?:do\s+we\s+have|is\s+there|have)\s+(?:any\s+)?(?:openalex|semantic\s+scholar|mag|dblp|pubmed|wos|web\s+of\s+science)\s+(?:data|dataset)?", ["midway_snapshots"], "midway_snapshot_lookup"),
        (r"(?:do\s+we\s+have|is\s+there)\s+(?:any\s+)?(\w+)\s+(?:data|dataset)\s+(?:on\s+midway|available)?", ["midway_snapshots", "midway_social", "midway_embeddings"], "general_data_lookup"),

        # Social media dataset queries
        (r"(?:who\s+has|where\s+is)\s+(?:the\s+)?(?:reddit|twitter|4chan|weibo|telegram|social\s+media)\s+data", ["midway_social"], "midway_social_lookup"),
        (r"what\s+(?:social\s+media|reddit|twitter)\s+(?:data|datasets?)\s+(?:do\s+we\s+have|exist|are\s+available)", ["midway_social"], "list_midway_social"),
        (r"(?:do\s+we\s+have|is\s+there|have)\s+(?:any\s+)?(?:reddit|twitter|4chan|weibo|telegram|toxicity)\s+(?:data|dataset)", ["midway_social"], "midway_social_lookup"),
        (r"(?:where|which\s+directory)\s+(?:is|has|contains)\s+(?:the\s+)?(?:toxicity|reddit|twitter|4chan|weibo)", ["midway_social"], "midway_social_lookup"),

        # Researcher directory queries
        (r"what(?:'s|\s+is)\s+in\s+(.+?)(?:'s)?\s+(?:midway\s+)?directory", ["midway_researchers"], "midway_researcher_directory"),
        (r"(.+?)(?:'s)?\s+(?:midway\s+)?(?:directory|storage|files?)(?:\s+on\s+midway)?", ["midway_researchers"], "midway_researcher_directory"),
        (r"who\s+(?:has|owns)\s+(?:the\s+)?(.+?)\s+(?:data|directory|storage)(?:\s+on\s+midway)?", ["midway_researchers"], "midway_researcher_lookup"),
        (r"(?:show|list)\s+(?:all\s+)?researcher\s+directories(?:\s+on\s+midway)?", ["midway_researchers"], "list_midway_researchers"),

        # Shared infrastructure queries
        (r"what\s+(?:llm|model)\s+(?:caches?|models?)\s+(?:do\s+we\s+have|are\s+available|exist)(?:\s+on\s+midway)?", ["midway_infrastructure"], "list_midway_infrastructure"),
        (r"(?:where\s+(?:is|are)|find)\s+(?:the\s+)?(?:shared\s+)?(?:llm|huggingface|hf|ollama)\s+(?:cache|models?)", ["midway_infrastructure"], "midway_infrastructure_lookup"),
        (r"(?:shared|lab)\s+(?:llm|model)\s+(?:cache|storage|resources?)", ["midway_infrastructure"], "midway_infrastructure_lookup"),

        # General Midway queries
        (r"what(?:'s|\s+is)\s+on\s+midway", ["midway_snapshots", "midway_precomputed", "midway_embeddings"], "list_midway_all"),
        (r"midway\s+(?:storage|data|resources?|assets?)", ["midway_snapshots", "midway_precomputed"], "list_midway_all"),

        # Catch-all patterns for specific data names (high priority)
        (r"(?:toxicity|toxic)\s+(?:data|dataset|datasets)", ["midway_social"], "midway_social_lookup"),
        (r"(?:reddit|twitter|4chan|weibo|telegram)\s+(?:data|datasets?)", ["midway_social"], "midway_social_lookup"),
        (r"(?:social\s+media)\s+(?:data|datasets?)", ["midway_social"], "list_midway_social"),
        (r"chinese\s+(?:social\s+media|posts?|data)", ["midway_social"], "midway_social_lookup"),  # Chinese → Weibo
        (r"(?:find|where|get|have)\s+.*(?:chinese|weibo|sina)", ["midway_social"], "midway_social_lookup"),
        (r"(?:where|how)\s+(?:can\s+I\s+)?(?:find|get)\s+.*(?:chinese|weibo)", ["midway_social"], "midway_social_lookup"),
        (r"(?:patent|patents)\s+(?:similarity|matrix|matrices|data)", ["midway_precomputed"], "midway_precomputed_lookup"),
        (r"(?:pmi|pointwise\s+mutual)\s+(?:statistics|data|matrix)", ["midway_precomputed"], "midway_precomputed_lookup"),
        (r"(?:concept|author)\s+(?:co-?occurrence|mapping)", ["midway_precomputed"], "midway_precomputed_lookup"),

        # Person + resource queries
        (r"what\s+(?:data|resources?|datasets?)\s+does\s+(\w+)\s+(?:have|use|work\s+with)", ["people", "midway_researchers"], "person_resources"),
        (r"(\w+)(?:'s)?\s+(?:data|resources?|midway)", ["people", "midway_researchers"], "person_resources"),
        (r"who\s+(?:uses?|works?\s+with)\s+(?:the\s+)?(.+?)(?:\s+data)?(?:\?|$)", ["people", "midway_researchers"], "resource_users"),
    ]

    # Keywords that suggest entity types (expanded for better coverage)
    ENTITY_KEYWORDS = {
        "people": [
            # Basic identifiers
            "who", "person", "people", "someone", "anybody", "anyone",
            # Academic roles
            "researcher", "researchers", "scientist", "scientists", "academic", "academics",
            "student", "students", "phd", "doctoral", "graduate",
            "postdoc", "postdocs", "postdoctoral",
            "faculty", "professor", "professors", "prof", "dr",
            "member", "members", "team", "staff", "fellow", "fellows",
            # Collaborative terms
            "collaborator", "collaborators", "co-author", "coauthor", "author",
            "colleague", "colleagues", "advisor", "advisee", "mentee", "mentor",
        ],
        "projects": [
            "project", "projects", "initiative", "initiatives",
            "program", "programs", "effort", "efforts",
            # Known project names
            "apto", "c3s2", "socio-cognitive",
        ],
        "funding": [
            "grant", "grants", "funding", "fund", "funds",
            "award", "awards", "fellowship", "fellowships",
            # Known funding sources
            "nsf", "nih", "muri", "darpa", "doe", "aro", "afosr", "onr",
            "foundation", "endowment",
        ],
        "datasets": [
            "dataset", "datasets", "data", "corpus", "corpora",
            "database", "databases", "collection", "collections",
            # Data card queries
            "data card", "datacard", "documentation",
            "variables", "schema", "fields", "columns",
            "limitations", "biases", "coverage",
            "provenance", "source", "license", "citation",
            "intended use", "use case", "suitable for",
            # Known datasets
            "openalex", "semantic scholar", "s2ag", "mag", "microsoft academic",
            "web of science", "wos", "patstat", "dblp", "pubmed",
            "reddit", "twitter", "4chan", "arxiv",
        ],
        "compute": [
            "compute", "computing", "computational",
            "cluster", "clusters", "gpu", "gpus", "hpc",
            "midway", "server", "servers", "machine", "machines",
            "resource", "resources", "infrastructure",
        ],
        "events": [
            "event", "events", "meeting", "meetings",
            "seminar", "seminars", "talk", "talks",
            "conference", "conferences", "workshop", "workshops",
            "retreat", "retreats", "symposium", "symposia",
        ],
        "code_repos": [
            "code", "codes", "repo", "repos", "repository", "repositories",
            "github", "gitlab", "software", "package", "packages", "library", "libraries",
        ],
        "topics": [
            "topic", "topics", "research area", "research areas",
            "field", "fields", "discipline", "disciplines",
            "expertise", "specialty", "specialties", "specialization",
            "interest", "interests", "focus", "focuses",
            # Action terms indicating topic search
            "works on", "working on", "studies", "studying",
            "researches", "researching", "specializes", "specializing",
        ],
        "publications": [
            "paper", "papers", "publication", "publications", "published",
            "article", "articles", "journal", "journals",
            "work", "works", "authored", "wrote", "written",
        ],
        "stats": [
            "publications", "papers", "articles", "works",
            "citations", "cited", "h-index", "hindex", "impact",
            "count", "number", "how many", "statistics", "stats", "metrics",
        ],
        "contact": [
            "contact", "email", "phone", "reach", "office",
            "get in touch", "how to contact",
        ],
        "comparison": [
            "compare", "comparison", "versus", "vs", "difference", "differences",
            "similar", "similarities", "between",
        ],
        "collaboration": [
            "collaborate", "collaborates", "collaboration", "collaborations",
            "co-author", "coauthor", "co-authors", "coauthors",
            "work with", "works with", "working with",
            "publish with", "publishes with",
            "partner", "partners", "partnership",
        ],
        "tools": [
            "tool", "tools", "application", "applications",
            "platform", "platforms", "service", "services",
            "slurm", "pyspark", "spark", "mysql", "jupyter",
            "openalex", "slack", "what tools",
        ],
        # Midway storage entity types
        "midway_snapshots": [
            "snapshot", "snapshots", "data snapshot", "data snapshots",
            "openalex", "semantic scholar", "s2ag", "mag", "dblp", "dimensions",
            "patstat", "icite", "pubmed", "pkg", "papers with code", "pwc",
            "bibliographic", "bibliometric",
            # Preprint servers (frequently asked about)
            "arxiv", "biorxiv", "medrxiv", "preprint", "preprints",
            # Additional academic databases
            "wos", "web of science", "scopus", "crossref",
        ],
        "midway_precomputed": [
            "precomputed", "pre-computed", "computed",
            "similarity matrix", "similarity matrices",
            "pmi", "pointwise mutual information",
            "patent similarity", "concept co-occurrence",
            "irreplaceable", "expensive computation",
        ],
        "midway_embeddings": [
            "embedding", "embeddings", "vectors", "representations",
            "specter", "specter2", "word2vec", "mat2vec", "hyperbolic",
            "patent embedding", "paper embedding", "mesh embedding",
            "openalex embedding", "author embedding",
            "google news", "citation embedding", "materials science embedding",
        ],
        "midway_social": [
            "reddit", "twitter", "4chan", "weibo", "telegram",
            "social media", "social media data", "social dataset",
            "toxicity", "toxic", "political corpus", "political corpora",
            "covid weibo", "conservative", "liberal",
        ],
        "midway_researchers": [
            "midway directory", "researcher directory", "storage",
            "what's in", "directory contents",
            "/project/jevans",
        ],
        "midway_infrastructure": [
            "llm cache", "model cache", "huggingface cache", "hf cache",
            "ollama", "shared models", "shared infrastructure",
            "gemma", "llama", "cached models",
        ],
        "guides": [
            "how to", "how do i", "how can i", "guide", "guides",
            "tutorial", "tutorials", "instructions", "steps",
            "connect", "submit", "run", "use", "start", "install",
            "setup", "configure", "check", "monitor",
            # Common guide topics
            "ssh", "slurm", "sbatch", "sinteractive", "jupyter",
            "gpu", "partition", "batch job", "interactive job",
            "storage", "quota", "pyspark", "job array",
        ],
    }

    # Known institutions for compound query detection
    KNOWN_INSTITUTIONS = {
        "stanford": "Stanford",
        "stanford university": "Stanford",
        "uchicago": "University of Chicago",
        "university of chicago": "University of Chicago",
        "chicago": "University of Chicago",
        "harvard": "Harvard",
        "harvard university": "Harvard",
        "mit": "MIT",
        "berkeley": "Berkeley",
        "uc berkeley": "Berkeley",
        "princeton": "Princeton",
        "yale": "Yale",
        "columbia": "Columbia",
        "nyu": "NYU",
        "cornell": "Cornell",
        "indiana": "Indiana University",
        "indiana university": "Indiana University",
        "michigan": "University of Michigan",
        "university of michigan": "University of Michigan",
        "northwestern": "Northwestern",
        "northwestern university": "Northwestern",
        "penn": "University of Pennsylvania",
        "upenn": "University of Pennsylvania",
        "university of pennsylvania": "University of Pennsylvania",
        "caltech": "Caltech",
        "duke": "Duke",
        "duke university": "Duke",
        "santa fe": "Santa Fe Institute",
        "santa fe institute": "Santa Fe Institute",
    }

    # Role keywords for compound query detection
    ROLE_KEYWORDS_MAP = {
        "researcher": ["researcher", "researchers"],
        "student": ["student", "students", "phd student", "phd students", "graduate student", "graduate students", "doctoral student", "doctoral students"],
        "phd": ["phd", "phd student", "phd students", "doctoral"],
        "postdoc": ["postdoc", "postdocs", "postdoctoral", "postdoctoral scholar", "postdoctoral scholars"],
        "faculty": ["faculty", "professor", "professors", "prof"],
        "staff": ["staff", "research staff"],
        "affiliate": ["affiliate", "affiliates"],
        "people": ["people", "members", "member"],
    }

    # Patterns for compound queries: {institution} {role} working on {topic}
    COMPOUND_PATTERNS = [
        # "{institution} {role} working on {topic}"
        (r"(\w+(?:\s+\w+)?)\s+(researchers?|students?|phd\s+students?|postdocs?|faculty|people|members?)\s+(?:working\s+on|studying|researching|in)\s+(.+?)(?:\?|$)",
         ["institution", "role", "topic"]),

        # "{role} at {institution} {studying/working on} {topic}"
        (r"(researchers?|students?|phd\s+students?|postdocs?|faculty|people|members?)\s+(?:at|from)\s+(\w+(?:\s+\w+)?)\s+(?:working\s+on|studying|researching|in)\s+(.+?)(?:\?|$)",
         ["role", "institution", "topic"]),

        # "people who work on {topic} at {institution}"
        (r"(?:people|researchers?|members?)\s+who\s+(?:work\s+on|study|research)\s+(.+?)\s+(?:at|from)\s+(\w+(?:\s+\w+)?)(?:\?|$)",
         ["topic", "institution"]),

        # "{topic} researchers at {institution}"
        (r"(.+?)\s+(researchers?|experts?|specialists?)\s+(?:at|from)\s+(\w+(?:\s+\w+)?)(?:\?|$)",
         ["topic", "role", "institution"]),

        # "{role} studying {topic}"
        (r"(researchers?|students?|phd\s+students?|postdocs?|faculty|people|members?)\s+(?:studying|researching|working\s+on|in)\s+(.+?)(?:\?|$)",
         ["role", "topic"]),

        # "who at {institution} works on {topic}"
        (r"who\s+(?:at|from)\s+(\w+(?:\s+\w+)?)\s+(?:works?\s+on|studies|researches?)\s+(.+?)(?:\?|$)",
         ["institution", "topic"]),

        # "{institution} {topic} {role}"
        (r"(\w+(?:\s+\w+)?)\s+(.+?)\s+(researchers?|experts?|specialists?|people)(?:\?|$)",
         ["institution", "topic", "role"]),

        # "find {role} in {topic} at {institution}"
        (r"(?:find|show|list)\s+(researchers?|people|members?|students?)\s+in\s+(.+?)\s+(?:at|from)\s+(\w+(?:\s+\w+)?)(?:\?|$)",
         ["role", "topic", "institution"]),
    ]

    def _detect_compound_query(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Detect if a query is a compound/intersection query with multiple filters.

        Returns a dict with extracted filters if compound, None otherwise.
        Filters can include: institution, role, topic
        """
        query_lower = query.lower().strip()

        # First, try to match against compound patterns
        for pattern, field_names in self.COMPOUND_PATTERNS:
            match = re.search(pattern, query_lower)
            if match:
                filters = {}
                groups = match.groups()

                for i, field_name in enumerate(field_names):
                    if i < len(groups) and groups[i]:
                        value = groups[i].strip()

                        # Normalize institutions
                        if field_name == "institution":
                            normalized = self._normalize_institution(value)
                            if normalized:
                                filters["institution"] = normalized
                        # Normalize roles
                        elif field_name == "role":
                            normalized = self._normalize_role(value)
                            if normalized:
                                filters["role"] = normalized
                        # Topics are kept as-is but cleaned up
                        elif field_name == "topic":
                            # Clean up common trailing words
                            topic = re.sub(r'\s+(research|field|area|topic)$', '', value)
                            if topic and len(topic) > 2:
                                filters["topic"] = topic

                # Only return as compound if we have at least 2 different filter types
                if len(filters) >= 2:
                    return filters

        # Fallback: Try to detect individual components without pattern matching
        filters = {}

        # Check for institution mentions
        for inst_key, inst_name in self.KNOWN_INSTITUTIONS.items():
            if inst_key in query_lower:
                filters["institution"] = inst_name
                break

        # Check for role mentions
        for role_type, role_variants in self.ROLE_KEYWORDS_MAP.items():
            for variant in role_variants:
                if variant in query_lower:
                    filters["role"] = role_type
                    break
            if "role" in filters:
                break

        # Check for topic indicators with content
        topic_indicators = [
            r"(?:working\s+on|studying|researching|in|on)\s+(.+?)(?:\s+at|\s+from|\?|$)",
            r"(.+?)\s+research(?:ers?)?",
        ]

        for indicator in topic_indicators:
            match = re.search(indicator, query_lower)
            if match:
                potential_topic = match.group(1).strip()
                # Filter out institution/role words from topic
                skip_words = set(self.KNOWN_INSTITUTIONS.keys())
                for variants in self.ROLE_KEYWORDS_MAP.values():
                    skip_words.update(variants)
                skip_words.update(["who", "what", "find", "list", "show", "the", "are", "is"])

                topic_words = [w for w in potential_topic.split() if w not in skip_words]
                if topic_words:
                    filters["topic"] = " ".join(topic_words)
                    break

        # Return filters only if we have at least 2 different types
        if len(filters) >= 2:
            return filters

        return None

    def _normalize_institution(self, value: str) -> Optional[str]:
        """Normalize an institution name to its canonical form."""
        value_lower = value.lower().strip()

        # Direct lookup
        if value_lower in self.KNOWN_INSTITUTIONS:
            return self.KNOWN_INSTITUTIONS[value_lower]

        # Partial match
        for inst_key, inst_name in self.KNOWN_INSTITUTIONS.items():
            if inst_key in value_lower or value_lower in inst_key:
                return inst_name

        # If not in known list but looks like an institution, return as-is
        if any(word in value_lower for word in ["university", "institute", "college"]):
            return value.title()

        return None

    def _normalize_role(self, value: str) -> Optional[str]:
        """Normalize a role keyword to its canonical form."""
        value_lower = value.lower().strip()

        for role_type, variants in self.ROLE_KEYWORDS_MAP.items():
            if value_lower in variants:
                return role_type

        # Partial match
        for role_type, variants in self.ROLE_KEYWORDS_MAP.items():
            for variant in variants:
                if variant in value_lower or value_lower in variant:
                    return role_type

        return None

    def classify(self, query: str) -> QueryClassification:
        """
        Classify a query for routing.

        Returns a QueryClassification with:
        - query_type: 'structured', 'semantic', or 'hybrid'
        - entity_types: list of relevant entity types
        - search_terms: extracted search terms
        - confidence: 0.0 to 1.0
        - is_compound: whether this is a compound/intersection query
        - compound_filters: extracted filters for compound queries
        """
        query_lower = query.lower().strip()

        # Check for compound queries FIRST (these take precedence)
        compound_filters = self._detect_compound_query(query)
        if compound_filters:
            # Build entity types based on filters
            entity_types = ["people"]  # Compound queries always search people
            if "topic" in compound_filters:
                entity_types.append("topics")

            return QueryClassification(
                query_type="structured",
                entity_types=entity_types,
                search_terms=list(compound_filters.values()),
                confidence=0.95,  # High confidence for compound queries
                is_compound=True,
                compound_filters=compound_filters
            )

        # Check structured patterns (ordered by specificity)
        for pattern_tuple in self.STRUCTURED_PATTERNS:
            pattern = pattern_tuple[0]
            entity_types = pattern_tuple[1]
            query_subtype = pattern_tuple[2] if len(pattern_tuple) > 2 else None

            match = re.search(pattern, query_lower)
            if match:
                # Calculate confidence based on pattern specificity
                confidence = self._calculate_pattern_confidence(pattern, query_subtype, match)
                return QueryClassification(
                    query_type="structured",
                    entity_types=entity_types,
                    search_terms=self._extract_terms(query_lower, match),
                    confidence=confidence
                )

        # Check for entity keywords
        detected_types = []
        keyword_matches = 0

        for entity_type, keywords in self.ENTITY_KEYWORDS.items():
            type_matched = False
            for kw in keywords:
                if kw in query_lower:
                    if not type_matched:
                        detected_types.append(entity_type)
                        type_matched = True
                    keyword_matches += 1

        if detected_types:
            # Calculate confidence based on keyword coverage
            confidence = min(0.5 + (keyword_matches * 0.1), 0.85)

            # Has entity keywords but not a clear structured pattern -> hybrid
            return QueryClassification(
                query_type="hybrid",
                entity_types=detected_types,
                search_terms=self._extract_terms(query_lower),
                confidence=confidence
            )

        # Check if query contains what looks like a person name
        extracted_names = self.extract_person_names(query)
        if extracted_names:
            return QueryClassification(
                query_type="hybrid",
                entity_types=["people"],
                search_terms=extracted_names + self._extract_terms(query_lower),
                confidence=0.6
            )

        # Default to semantic search
        return QueryClassification(
            query_type="semantic",
            entity_types=[],
            search_terms=self._extract_terms(query_lower),
            confidence=0.5
        )

    def _calculate_pattern_confidence(self, pattern: str, query_subtype: Optional[str], match: re.Match) -> float:
        """
        Calculate confidence score based on pattern characteristics.

        More specific patterns get higher confidence.
        """
        base_confidence = 0.85

        # Boost confidence for specific query subtypes
        high_confidence_subtypes = {
            "comparison", "publication_count", "citations", "h_index",
            "contact", "collaborators", "research_interests"
        }
        if query_subtype in high_confidence_subtypes:
            base_confidence = 0.95

        medium_confidence_subtypes = {
            "person_lookup", "topic_search", "project_members", "publication_search"
        }
        if query_subtype in medium_confidence_subtypes:
            base_confidence = 0.90

        # Boost if the match covers most of the query
        match_coverage = len(match.group(0)) / max(len(match.string), 1)
        if match_coverage > 0.7:
            base_confidence = min(base_confidence + 0.05, 1.0)

        return base_confidence

    def extract_person_names(self, query: str) -> List[str]:
        """
        Extract potential person names from a query.

        Uses multiple strategies:
        1. Look for capitalized word sequences (First Last pattern)
        2. Remove common title prefixes
        3. Handle possessive forms ('s)
        4. Handle quoted names

        Returns:
            List of potential person names found in the query
        """
        names = []

        # Strategy 1: Look for quoted names
        quoted_pattern = r'["\']([^"\']+)["\']'
        quoted_matches = re.findall(quoted_pattern, query)
        for match in quoted_matches:
            # Check if it looks like a name (1-4 capitalized words)
            words = match.split()
            if 1 <= len(words) <= 4 and all(w[0].isupper() for w in words if w):
                names.append(match)

        # Strategy 2: Look for capitalized word sequences
        # Pattern: 1-4 capitalized words in sequence
        cap_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b'
        cap_matches = re.findall(cap_pattern, query)

        # Filter out common non-name capitalized words
        non_names = {
            "Who", "What", "Where", "When", "How", "Which", "Tell", "Find",
            "List", "Show", "Get", "Give", "Send", "University", "College",
            "Institute", "Department", "School", "Lab", "Laboratory",
            "Research", "Project", "Grant", "PhD", "Professor", "Dr",
            "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
            "Chicago", "Stanford", "Harvard", "MIT", "Berkeley", "Yale", "Princeton",
            "Compare", "Difference", "Between", "Publications", "Papers", "Articles",
        }

        for match in cap_matches:
            # Skip if it's a common non-name word
            if match in non_names:
                continue

            # Skip single words that are likely not names
            words = match.split()
            if len(words) == 1 and match in non_names:
                continue

            # Remove title prefixes if present
            clean_name = match
            for prefix in self.TITLE_PREFIXES:
                if clean_name.lower().startswith(prefix + " "):
                    clean_name = clean_name[len(prefix)+1:].strip()
                    break

            if clean_name and len(clean_name) > 2:
                names.append(clean_name)

        # Strategy 3: Handle possessive forms
        possessive_pattern = r"(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'s\b"
        poss_matches = re.findall(possessive_pattern, query)
        for match in poss_matches:
            if match not in non_names and match not in names:
                names.append(match)

        # Strategy 4: Look for patterns like "contact X" or "about X" where X might be a name
        about_pattern = r"(?:contact|about|regarding|for|of|with)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"
        about_matches = re.findall(about_pattern, query)
        for match in about_matches:
            if match not in non_names and match not in names:
                names.append(match)

        # Deduplicate while preserving order
        seen = set()
        unique_names = []
        for name in names:
            if name.lower() not in seen:
                seen.add(name.lower())
                unique_names.append(name)

        return unique_names

    def _extract_terms(self, query: str, match: Optional[re.Match] = None) -> List[str]:
        """
        Extract search terms from query.

        If a regex match is provided, prioritize captured groups.
        """
        terms = []

        # If we have a regex match, extract captured groups first
        if match:
            for group in match.groups():
                if group:
                    # Clean up the captured group
                    cleaned = group.strip().strip("?.,!").strip()
                    if cleaned and len(cleaned) > 2:
                        terms.append(cleaned)

        # Extended stopwords list
        stopwords = {
            # Question words
            "who", "what", "where", "when", "how", "why", "which",
            # Articles and prepositions
            "is", "are", "was", "were", "be", "been", "being",
            "the", "a", "an", "and", "or", "but", "if", "then",
            "do", "does", "did", "done", "doing",
            "we", "i", "you", "they", "he", "she", "it",
            "have", "has", "had", "having",
            "all", "any", "some", "no", "not",
            # Common verbs in queries
            "list", "tell", "me", "about", "describe", "show", "find", "get", "give",
            "can", "could", "would", "should", "may", "might",
            # Prepositions
            "on", "in", "at", "to", "for", "of", "with", "by", "from",
            "up", "down", "into", "onto", "upon",
            # Other common words
            "this", "that", "these", "those",
            "there", "here", "please", "thanks", "thank",
        }

        # Extract individual words
        words = re.findall(r'\b\w+\b', query.lower())
        for word in words:
            if word not in stopwords and len(word) > 2 and word not in [t.lower() for t in terms]:
                terms.append(word)

        return terms

    def get_query_intent(self, query: str) -> Dict[str, Any]:
        """
        Analyze a query and return detailed intent information.

        Returns a dictionary with:
        - classification: the QueryClassification result
        - extracted_names: list of person names found
        - query_subtype: specific type of query (if determinable)
        - suggested_actions: list of actions to take
        """
        classification = self.classify(query)
        extracted_names = self.extract_person_names(query)
        query_lower = query.lower()

        # Determine query subtype
        query_subtype = None
        for pattern_tuple in self.STRUCTURED_PATTERNS:
            if len(pattern_tuple) > 2:
                pattern, _, subtype = pattern_tuple
                if re.search(pattern, query_lower):
                    query_subtype = subtype
                    break

        # Determine suggested actions based on classification
        suggested_actions = []

        if "people" in classification.entity_types:
            if query_subtype == "contact":
                suggested_actions.append("lookup_person_contact")
            elif query_subtype == "research_interests":
                suggested_actions.append("lookup_person_topics")
            elif query_subtype == "collaborators":
                suggested_actions.append("lookup_collaborations")
            elif query_subtype == "comparison":
                suggested_actions.append("compare_people")
            elif query_subtype in ("publication_count", "citations", "h_index"):
                suggested_actions.append("lookup_person_stats")
            else:
                suggested_actions.append("lookup_person")

        if "topics" in classification.entity_types:
            suggested_actions.append("search_by_topic")

        if "projects" in classification.entity_types:
            suggested_actions.append("lookup_project")

        if "publications" in classification.entity_types:
            suggested_actions.append("search_publications")

        return {
            "classification": classification,
            "extracted_names": extracted_names,
            "query_subtype": query_subtype,
            "suggested_actions": suggested_actions,
        }
