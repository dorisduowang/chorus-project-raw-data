"""
Registry RAG Module

Main class combining structured registry lookup with semantic RAG.

Query flow:
1. Correct typos in query
2. Classify query (structured, semantic, hybrid)
3. For structured: direct registry lookup
4. For semantic/hybrid: also search embedded summaries
5. Merge and rank results
6. Log query for analysis
"""

import json
import re
import time
from typing import Dict, List, Any, Optional

from .models import QueryClassification
from .lookup import RegistryLookup, MidwayLookup
from .classifier import QueryClassifier
from .text_generator import RegistryTextGenerator
from .cross_links import CrossRegistryLinker, get_linker
from .query_suggest import QuerySuggester, get_suggester
from .typo_correction import TypoCorrector, get_corrector
from .query_logger import QueryLogger, get_logger, log_query


class RegistryRAG:
    """
    Main class combining structured registry lookup with semantic RAG.

    Query flow:
    1. Classify query (structured, semantic, hybrid)
    2. For structured: direct registry lookup
    3. For semantic/hybrid: also search embedded summaries
    4. Merge and rank results
    """

    def __init__(self, registry_path: str = "Data/lab_registry.json", midway_path: str = "Data/midway_registry.json"):
        self.lookup = RegistryLookup(registry_path)
        self.midway = MidwayLookup(midway_path)
        self.classifier = QueryClassifier()
        self.text_gen = RegistryTextGenerator()
        self.cross_linker = CrossRegistryLinker(registry_path, midway_path)
        self.suggester = QuerySuggester(registry_path, midway_path)
        self.typo_corrector = TypoCorrector(registry_path, midway_path)
        self.query_logger = get_logger()

    def reload_if_changed(self) -> bool:
        """Check if registry or midway data changed and reload if needed."""
        registry_reloaded = self.lookup.reload_if_changed()
        midway_reloaded = self.midway.reload_if_changed()
        return registry_reloaded or midway_reloaded

    def force_reload(self) -> Dict[str, Any]:
        """Force reload the registry and midway data."""
        result = self.lookup.force_reload()
        self.midway._load_registry()  # Force reload midway
        return result

    def query(self, query: str, include_semantic: bool = True) -> Dict[str, Any]:
        """
        Execute a query against the registry.

        Returns:
            Dict with:
            - query_type: how the query was classified
            - structured_results: direct registry matches
            - semantic_results: embedding-based matches (if enabled)
            - combined_answer: merged response
        """
        start_time = time.time()
        original_query = query

        # Step 1: Typo correction
        corrected_query, corrections = self.typo_corrector.correct_query(query)
        if corrections:
            query = corrected_query  # Use corrected query for search

        # Step 2: Classify the query
        classification = self.classifier.classify(query)

        result = {
            "query": original_query,
            "classification": {
                "type": classification.query_type,
                "entity_types": classification.entity_types,
                "confidence": classification.confidence,
                "is_compound": classification.is_compound,
                "compound_filters": classification.compound_filters
            },
            "structured_results": [],
            "answer": None
        }

        # Add correction info if we made corrections
        if corrections:
            result["corrected_query"] = corrected_query
            result["corrections"] = corrections

        # Handle compound queries with intersection logic
        if classification.is_compound and classification.compound_filters:
            structured = self._compound_search(classification.compound_filters)
            result["structured_results"] = structured
            result["answer"] = self._format_compound_answer(classification.compound_filters, structured)
        # Execute structured lookup
        elif classification.query_type in ("structured", "hybrid"):
            structured = self._structured_search(query, classification)

            # Check for person-resource queries (cross-registry)
            if not structured and "person_resources" in classification.query_type:
                cross_result = self.cross_linker.query_person_resources(query)
                if cross_result.get("resources"):
                    structured = self._format_cross_link_results(cross_result)
                    result["classification"]["cross_registry"] = True

            # Fallback: if no results and query might be Midway-related, search all categories
            if not structured:
                midway_keywords = ["midway", "snapshot", "embedding", "precomputed", "social", "reddit", "twitter",
                                   "weibo", "toxicity", "word2vec", "specter", "openalex", "semantic scholar",
                                   "patstat", "mag", "dblp", "pubmed", "/project/jevans"]
                query_lower = query.lower()
                if any(kw in query_lower for kw in midway_keywords):
                    structured = self._midway_fallback_search(query)
                    if structured:
                        result["classification"]["fallback_used"] = True

            result["structured_results"] = structured

            # Validate result relevance: check if results actually match key query terms
            if structured:
                structured = self._validate_result_relevance(query, structured, classification)
                result["structured_results"] = structured
                if not structured:
                    result["classification"]["relevance_filtered"] = True

            # Add suggestions if no results
            if not structured:
                suggestions = self.suggester.get_suggestions(
                    query,
                    result["classification"],
                    num_results=0
                )
                if suggestions["has_suggestions"]:
                    result["suggestions"] = suggestions
                    result["answer"] = self.suggester.format_no_results_message(
                        query, result["classification"]
                    )
                else:
                    result["answer"] = f"No results found in the registry for: {query}"
            else:
                result["answer"] = self._format_structured_answer(query, structured, classification)

        # Log the query for analysis
        latency_ms = (time.time() - start_time) * 1000
        try:
            self.query_logger.log(
                query=original_query,
                classification=result["classification"],
                num_results=len(result.get("structured_results", [])),
                latency_ms=latency_ms,
                corrected_query=result.get("corrected_query"),
                corrections=result.get("corrections"),
            )
        except Exception:
            pass  # Don't fail query if logging fails

        return result

    def _validate_result_relevance(
        self,
        query: str,
        results: List[Dict],
        classification: QueryClassification
    ) -> List[Dict]:
        """
        Validate that results are actually relevant to the query.

        If the user asks for "arxiv data" but we return "researcher directories",
        this catches the mismatch and returns empty results to trigger fallback.

        Args:
            query: Original query string
            results: Structured results from search
            classification: Query classification

        Returns:
            Filtered results, or empty list if results don't match query intent
        """
        # Don't filter people queries - they're usually correct
        if classification.query_type == "structured" and "people" in classification.entity_types:
            if len(classification.entity_types) == 1:
                return results  # Pure people query, trust it

        # Extract key data terms from query
        query_lower = query.lower()
        data_keywords = {
            # Preprint servers
            "arxiv", "biorxiv", "medrxiv", "preprint",
            # Academic databases
            "openalex", "semantic scholar", "mag", "dblp", "pubmed",
            "scopus", "wos", "web of science", "crossref", "dimensions",
            # Social media
            "reddit", "twitter", "4chan", "weibo", "toxicity",
            # Embeddings
            "embedding", "embeddings", "word2vec", "specter", "mat2vec",
            # Other data types
            "patent", "pmi", "similarity", "snapshot",
        }

        # Find data keywords in query
        query_data_terms = [kw for kw in data_keywords if kw in query_lower]

        # If query mentions specific data, verify results contain it
        if query_data_terms:
            # Serialize results to check content
            results_text = json.dumps(results).lower()

            # Check if any query terms appear in results
            found_terms = [term for term in query_data_terms if term in results_text]

            if not found_terms:
                # Results don't contain what user asked for
                return []

        return results

    def _format_cross_link_results(self, cross_result: Dict) -> List[Dict]:
        """Format cross-registry link results."""
        results = []
        person = cross_result.get("person", {})

        # Add person as first result
        if person:
            results.append({
                "type": "person",
                "data": person,
                "match_reason": "Query subject"
            })

        # Add their resources
        for resource in cross_result.get("resources", []):
            results.append({
                "type": f"midway_{resource.get('category', 'resource')}",
                "data": resource,
                "match_reason": f"Used by {person.get('name', 'person')}: {', '.join(resource.get('reasons', []))}"
            })

        return results

    def _structured_search(self, query: str, classification: QueryClassification) -> List[Dict[str, Any]]:
        """Execute structured search against registry."""
        results = []
        seen_ids = set()  # Track seen IDs to avoid duplicates
        query_lower = query.lower()

        def add_result(result_type: str, data: Dict, reason: str):
            """Add result if not already seen."""
            entity_id = data.get("id", "")
            if entity_id and entity_id not in seen_ids:
                seen_ids.add(entity_id)
                results.append({
                    "type": result_type,
                    "data": data,
                    "match_reason": reason
                })

        # Person lookup patterns
        if "people" in classification.entity_types:
            # "Who is X?" or "Tell me about X"
            for term in classification.search_terms:
                person = self.lookup.find_person(term)
                if person:
                    add_result("person", person, f"Name match: {term}")

            # "Who is at X?" - institution lookup (skip if this is a topic search)
            if "topics" not in classification.entity_types:
                inst_match = re.search(r"who is at (.+?)[\?\.]?$", query_lower)
                if inst_match:
                    institution = inst_match.group(1).strip()
                    people = self.lookup.find_people_by_institution(institution)
                    for p in people:
                        add_result("person", p, f"Institution: {institution}")

                # Also check if any search term matches an institution
                for term in classification.search_terms:
                    if len(term) > 3:  # Avoid short matches
                        people = self.lookup.find_people_by_institution(term)
                        for p in people:
                            add_result("person", p, f"Institution: {term}")

            # Role-based queries
            role_patterns = [
                ("phd student", "PhD Student"),
                ("postdoc", "Postdoctoral Scholar"),
                ("faculty", "Faculty"),
                ("affiliate", "Affiliate"),
            ]
            for pattern, role in role_patterns:
                if pattern in query_lower:
                    people = self.lookup.find_people_by_role(role)
                    for p in people:
                        add_result("person", p, f"Role: {role}")

        # Topic-based search
        if "topics" in classification.entity_types:
            # Extract the topic from various query patterns
            topic = None

            # Pattern: "who works on X" / "who studies X" / "who researches X"
            topic_patterns = [
                r"who (?:works on|studies|researches?|is (?:working|researching) on)\s+(.+?)(?:\?|$)",
                r"(?:find |list |show )?(?:researchers?|people|members?|who) (?:studying|working on|researching)\s+(.+?)(?:\?|$)",
                r"(?:researchers?|people|members?) (?:who work|working) on\s+(.+?)(?:\?|$)",
                r"anyone (?:working on|studying|researching)\s+(.+?)(?:\?|$)",
                r"(?:find|list|show) (?:researchers?|people|members?) (?:in|on|for) (?:the )?(.+?)(?: topic| area| field)?(?:\?|$)",
            ]

            for pattern in topic_patterns:
                match = re.search(pattern, query_lower)
                if match:
                    topic = match.group(1).strip()
                    # Clean up common trailing words
                    topic = re.sub(r'\s+(topic|area|field|research)$', '', topic)
                    break

            # Fallback: use search terms if no pattern matched
            if not topic and classification.search_terms:
                # Filter out common non-topic words
                skip_words = {"who", "find", "list", "show", "people", "researchers", "members", "working", "studying"}
                topic_terms = [t for t in classification.search_terms if t not in skip_words]
                if topic_terms:
                    topic = " ".join(topic_terms)

            if topic:
                topic_matches = self.lookup.find_people_by_topic(topic)
                for match in topic_matches:
                    person = match["person"]
                    matched_topics = match["matched_topics"]
                    # Create enhanced data with matched topics info
                    enhanced_person = dict(person)
                    enhanced_person["_matched_topics"] = matched_topics
                    add_result("person", enhanced_person, f"Topic: {', '.join(matched_topics[:2])}")

        # Project lookup
        if "projects" in classification.entity_types:
            # Check for list queries first
            if any(kw in query_lower for kw in ["list", "all", "what projects", "show"]):
                for project in self.lookup.list_by_type("projects"):
                    add_result("project", project, "List: projects")
            else:
                for term in classification.search_terms:
                    project = self.lookup.find_project(term)
                    if project:
                        add_result("project", project, f"Project match: {term}")

        # Funding lookup
        if "funding" in classification.entity_types:
            # Check for list queries first
            if any(kw in query_lower for kw in ["list", "all", "what funding", "what grants", "show"]):
                for funding in self.lookup.list_by_type("funding"):
                    add_result("funding", funding, "List: funding")
            else:
                for term in classification.search_terms:
                    funding = self.lookup.find_funding(term)
                    if funding:
                        add_result("funding", funding, f"Funding match: {term}")

        # Datasets lookup (data cards)
        if "datasets" in classification.entity_types:
            # Check for list queries first
            if any(kw in query_lower for kw in ["list", "all", "what datasets", "what data do we have"]):
                for dataset in self.lookup.list_datasets():
                    add_result("dataset", dataset, "List: datasets")
            else:
                # Try to find specific dataset by name
                dataset_patterns = [
                    r"(?:tell\s+me\s+about|describe|explain)\s+(?:the\s+)?(.+?)(?:\s+data(?:set)?|\?|$)",
                    r"(?:data\s+card|datacard)\s+(?:for\s+)?(?:the\s+)?(.+?)(?:\?|$)",
                    r"(?:what\s+(?:are|is)\s+(?:the\s+)?(?:variables?|fields?|limitations?|biases?|license))\s+(?:in|for|of)\s+(?:the\s+)?(.+?)(?:\?|$)",
                    r"how\s+(?:do\s+i|to)\s+cite\s+(?:the\s+)?(.+?)(?:\?|$)",
                ]

                search_term = None
                for pattern in dataset_patterns:
                    match = re.search(pattern, query_lower)
                    if match:
                        search_term = match.group(1).strip()
                        break

                if not search_term:
                    # Use search terms from classification
                    search_term = " ".join(classification.search_terms) if classification.search_terms else query_lower

                # Try to find matching dataset
                dataset = self.lookup.find_dataset(search_term)
                if dataset:
                    add_result("dataset", dataset, f"Dataset match: {search_term}")
                else:
                    # Try tag-based search
                    for term in search_term.split():
                        if len(term) > 2:
                            matches = self.lookup.find_datasets_by_tag(term)
                            for m in matches:
                                add_result("dataset", m, f"Tag match: {term}")

                    # Try use case matching
                    if "use" in query_lower or "suitable" in query_lower or "good for" in query_lower:
                        matches = self.lookup.find_datasets_for_use(search_term)
                        for m in matches:
                            add_result("dataset", m, f"Use case match: {search_term}")

        # Events lookup
        if "events" in classification.entity_types:
            for event in self.lookup.list_by_type("events"):
                add_result("event", event, "List: events")

        # Compute resources lookup
        if "compute" in classification.entity_types:
            for item in self.lookup.list_by_type("compute"):
                add_result("compute", item, "List: compute")

        # Tools lookup
        if "tools" in classification.entity_types:
            for item in self.lookup.list_by_type("tools"):
                add_result("tool", item, "List: tools")

        # Midway data snapshots lookup
        if "midway_snapshots" in classification.entity_types:
            # Check for list queries
            if any(kw in query_lower for kw in ["list", "all", "what snapshots", "what data"]):
                for snapshot in self.midway.list_snapshots():
                    add_result("midway_snapshot", snapshot, "List: snapshots")
            else:
                # Search for specific snapshot
                for term in classification.search_terms:
                    snapshot = self.midway.find_snapshot(term)
                    if snapshot:
                        add_result("midway_snapshot", snapshot, f"Snapshot match: {term}")
                    else:
                        # Try keyword search
                        matches = self.midway.find_snapshots_by_keyword(term)
                        for m in matches:
                            add_result("midway_snapshot", m, f"Keyword match: {term}")

        # Midway precomputed resources lookup
        if "midway_precomputed" in classification.entity_types:
            if any(kw in query_lower for kw in ["list", "all", "what precomputed", "what matrices"]):
                for resource in self.midway.list_precomputed():
                    add_result("midway_precomputed", resource, "List: precomputed")
            else:
                for term in classification.search_terms:
                    resource = self.midway.find_precomputed(term)
                    if resource:
                        add_result("midway_precomputed", resource, f"Precomputed match: {term}")

        # Midway embeddings lookup
        if "midway_embeddings" in classification.entity_types:
            if any(kw in query_lower for kw in ["list", "all", "what embeddings"]):
                for embedding in self.midway.list_embeddings():
                    add_result("midway_embedding", embedding, "List: embeddings")
            else:
                for term in classification.search_terms:
                    embedding = self.midway.find_embedding(term)
                    if embedding:
                        add_result("midway_embedding", embedding, f"Embedding match: {term}")

        # Midway social media datasets lookup
        if "midway_social" in classification.entity_types:
            if any(kw in query_lower for kw in ["list", "all", "what social", "what reddit"]):
                for dataset in self.midway.list_social_datasets():
                    add_result("midway_social", dataset, "List: social media")
            else:
                # Check for platform-specific queries
                platforms = ["reddit", "twitter", "4chan", "weibo", "telegram"]
                for platform in platforms:
                    if platform in query_lower:
                        matches = self.midway.find_social_by_platform(platform)
                        for m in matches:
                            add_result("midway_social", m, f"Platform: {platform}")
                        break
                else:
                    for term in classification.search_terms:
                        dataset = self.midway.find_social_dataset(term)
                        if dataset:
                            add_result("midway_social", dataset, f"Social match: {term}")

        # Midway researcher directories lookup
        if "midway_researchers" in classification.entity_types:
            if any(kw in query_lower for kw in ["list", "all", "show researcher"]):
                for researcher in self.midway.list_researcher_directories():
                    add_result("midway_researcher", researcher, "List: researchers")
            else:
                for term in classification.search_terms:
                    researcher = self.midway.find_researcher_directory(term)
                    if researcher:
                        add_result("midway_researcher", researcher, f"Researcher match: {term}")
                    else:
                        # Find researchers with specific data
                        matches = self.midway.find_researchers_with_data(term)
                        for m in matches:
                            add_result("midway_researcher", m, f"Has data: {term}")

        # Midway infrastructure lookup
        if "midway_infrastructure" in classification.entity_types:
            if any(kw in query_lower for kw in ["list", "all", "what llm", "what model", "shared"]):
                for infra in self.midway.list_infrastructure():
                    add_result("midway_infrastructure", infra, "List: infrastructure")
            else:
                for term in classification.search_terms:
                    infra = self.midway.find_infrastructure(term)
                    if infra:
                        add_result("midway_infrastructure", infra, f"Infrastructure match: {term}")

        # Guides lookup
        if "guides" in classification.entity_types:
            if any(kw in query_lower for kw in ["list", "all guides", "what guides", "show guides"]):
                for guide in self.lookup.list_guides():
                    add_result("guide", guide, "List: guides")
            else:
                # Extract guide topic from how-to queries
                guide_patterns = [
                    r"how\s+(?:do\s+i|can\s+i|to)\s+(?:connect|ssh|log\s*in|access)\s+(?:to\s+)?(.+?)(?:\?|$)",
                    r"how\s+(?:do\s+i|can\s+i|to)\s+(.+?)(?:\?|$)",
                    r"(?:which|what)\s+partition",
                ]

                search_term = None
                for pattern in guide_patterns:
                    match = re.search(pattern, query_lower)
                    if match:
                        search_term = match.group(1).strip() if match.lastindex else query_lower
                        break

                if not search_term:
                    search_term = query_lower

                # Try to find matching guide
                guide = self.lookup.find_guide(search_term)
                if guide:
                    add_result("guide", guide, f"Guide match: {search_term}")
                else:
                    # Try keyword search
                    keywords = search_term.split()
                    for kw in keywords:
                        if len(kw) > 2:  # Skip very short words
                            matches = self.lookup.find_guides_by_keyword(kw)
                            for m in matches:
                                add_result("guide", m, f"Keyword match: {kw}")

        # Publications lookup
        if "publications" in classification.entity_types:
            # Extract search term from query patterns
            search_term = None

            pub_patterns = [
                r"what has (.+?) published",
                r"(?:find|search|show|list) (?:papers?|publications?) (?:about|on|regarding)\s+(.+?)(?:\?|$)",
                r"(?:papers?|publications?) (?:about|on|by)\s+(.+?)(?:\?|$)",
                r"(?:papers?|publications?) by (.+?)(?:\?|$)",
                r"(.+?)(?:'s|s') (?:papers?|publications?)",
                r"(?:find|search|show|list) (.+?) (?:papers?|publications?)",
                r"(?:research|work) published (?:by|on|about)\s+(.+?)(?:\?|$)",
            ]

            for pattern in pub_patterns:
                match = re.search(pattern, query_lower)
                if match:
                    search_term = match.group(1).strip()
                    # Clean up common words
                    search_term = re.sub(r'^(all |any |some )', '', search_term)
                    break

            # Fallback to search terms
            if not search_term and classification.search_terms:
                skip_words = {"find", "search", "show", "list", "papers", "publications", "published", "about", "what", "has"}
                pub_terms = [t for t in classification.search_terms if t not in skip_words]
                if pub_terms:
                    search_term = " ".join(pub_terms)

            if search_term:
                publications = self.lookup.find_publications(search_term)
                for pub in publications[:20]:  # Limit to top 20
                    # Use DOI as ID for deduplication, or title hash
                    pub_id = pub.get("doi", "") or f"pub_{hash(pub.get('title', ''))}"
                    pub_with_id = dict(pub)
                    pub_with_id["id"] = pub_id
                    results.append({
                        "type": "publication",
                        "data": pub_with_id,
                        "match_reason": pub.get("match_reason", "Search match")
                    })

        return results

    def _midway_fallback_search(self, query: str) -> List[Dict[str, Any]]:
        """
        Fallback search across all Midway categories when primary search returns no results.
        Searches for keyword matches in names, descriptions, and paths.
        """
        results = []
        seen_ids = set()
        query_lower = query.lower()

        # Extract search terms (remove common words)
        stop_words = {"do", "we", "have", "any", "is", "are", "there", "the", "where", "what", "which", "find", "data", "dataset", "on", "midway", "path", "location", "directory"}
        terms = [w for w in query_lower.split() if w not in stop_words and len(w) > 2]

        if not terms:
            return results

        def matches_query(item: Dict) -> bool:
            """Check if item matches any search term."""
            searchable = " ".join([
                str(item.get("name", "")),
                str(item.get("description", "")),
                str(item.get("path", "")),
                str(item.get("username", "")),
                str(item.get("primary_focus", "")),
                " ".join(item.get("entity_types", [])) if isinstance(item.get("entity_types"), list) else "",
            ]).lower()
            return any(term in searchable for term in terms)

        def add_result(result_type: str, data: Dict, reason: str):
            entity_id = data.get("id", data.get("username", ""))
            if entity_id and entity_id not in seen_ids:
                seen_ids.add(entity_id)
                results.append({"type": result_type, "data": data, "match_reason": reason})

        # Search all Midway categories
        for snapshot in self.midway.list_snapshots():
            if matches_query(snapshot):
                add_result("midway_snapshot", snapshot, f"Fallback match: {', '.join(terms)}")

        for precomputed in self.midway.list_precomputed():
            if matches_query(precomputed):
                add_result("midway_precomputed", precomputed, f"Fallback match: {', '.join(terms)}")

        for embedding in self.midway.list_embeddings():
            if matches_query(embedding):
                add_result("midway_embedding", embedding, f"Fallback match: {', '.join(terms)}")

        for social in self.midway.list_social_datasets():
            if matches_query(social):
                add_result("midway_social", social, f"Fallback match: {', '.join(terms)}")

        for researcher in self.midway.list_researcher_directories():
            if matches_query(researcher):
                add_result("midway_researcher", researcher, f"Fallback match: {', '.join(terms)}")

        for infra in self.midway.list_infrastructure():
            if matches_query(infra):
                add_result("midway_infrastructure", infra, f"Fallback match: {', '.join(terms)}")

        return results

    def _format_structured_answer(self, query: str, results: List[Dict], classification: QueryClassification) -> str:
        """Format structured results into a readable answer."""
        if not results:
            return f"No results found in the registry for: {query}"

        # Single person query
        if len(results) == 1 and results[0]["type"] == "person":
            p = results[0]["data"]
            lines = [f"**{p.get('name')}**"]
            lines.append(f"- Role: {p.get('role')}")
            if p.get("institution"):
                lines.append(f"- Institution: {p['institution']}")
            if p.get("email"):
                lines.append(f"- Email: {p['email']}")
            if p.get("openalex"):
                oa = p["openalex"]
                lines.append(f"- Publications: {oa.get('works_count', 0)} works, {oa.get('cited_by_count', 0)} citations")
                lines.append(f"- h-index: {oa.get('h_index', 0)}")
            elif p.get("google_scholar", {}).get("citations"):
                lines.append(f"- Google Scholar citations: {p['google_scholar']['citations']}")
            return "\n".join(lines)

        # List of people
        if all(r["type"] == "person" for r in results):
            # Check if this is a topic-based search (has matched topics)
            is_topic_search = any(r["data"].get("_matched_topics") for r in results)

            if is_topic_search:
                lines = [f"Found {len(results)} researchers matching the topic:\n"]
                for r in results[:15]:
                    p = r["data"]
                    info = f"- **{p.get('name')}** ({p.get('role')})"
                    if p.get("institution"):
                        info += f" - {p['institution']}"
                    # Show matched topics
                    matched = p.get("_matched_topics", [])
                    if matched:
                        info += f"\n  Topics: {', '.join(matched[:3])}"
                        if len(matched) > 3:
                            info += f" (+{len(matched) - 3} more)"
                    lines.append(info)
            else:
                lines = [f"Found {len(results)} people:\n"]
                for r in results[:15]:
                    p = r["data"]
                    info = f"- **{p.get('name')}** ({p.get('role')})"
                    if p.get("institution"):
                        info += f" - {p['institution']}"
                    lines.append(info)

            if len(results) > 15:
                lines.append(f"\n... and {len(results) - 15} more")
            return "\n".join(lines)

        # Single project
        if len(results) == 1 and results[0]["type"] == "project":
            proj = results[0]["data"]
            lines = [f"**{proj.get('name')}** ({proj.get('status')})"]
            if proj.get("description"):
                lines.append(f"\n{proj['description']}")
            if proj.get("leads"):
                lines.append(f"\n- Leads: {', '.join(proj['leads'])}")
            if proj.get("tags"):
                lines.append(f"- Topics: {', '.join(proj['tags'])}")
            return "\n".join(lines)

        # Single dataset (data card)
        if len(results) == 1 and results[0]["type"] == "dataset":
            ds = results[0]["data"]
            lines = [f"**{ds.get('name')}**"]
            if ds.get("version"):
                lines[0] += f" (v{ds['version']})"

            if ds.get("description"):
                lines.append(f"\n{ds['description']}")

            # Provenance section
            prov = ds.get("provenance", {})
            if prov:
                lines.append("\n**Provenance:**")
                if prov.get("source"):
                    lines.append(f"- Source: {prov['source']}")
                if prov.get("collection_date"):
                    lines.append(f"- Date: {prov['collection_date']}")
                if prov.get("license"):
                    lines.append(f"- License: {prov['license']}")
                if prov.get("citation"):
                    lines.append(f"- Citation: {prov['citation']}")

            # Technical details
            tech = ds.get("technical", {})
            if tech:
                lines.append("\n**Technical:**")
                if tech.get("format"):
                    lines.append(f"- Format: {tech['format']}")
                if tech.get("size_gb"):
                    lines.append(f"- Size: {tech['size_gb']} GB")
                if tech.get("num_records"):
                    lines.append(f"- Records: {tech['num_records']:,}")
                if tech.get("location"):
                    lines.append(f"- Path: `{tech['location']}`")

            # Key variables (show first 5)
            variables = ds.get("variables", [])
            if variables:
                lines.append("\n**Key Variables:**")
                for var in variables[:5]:
                    lines.append(f"- `{var.get('name')}` ({var.get('type')}): {var.get('description', '')}")
                if len(variables) > 5:
                    lines.append(f"- ... and {len(variables) - 5} more")

            # Intended use
            intended = ds.get("intended_use", {})
            if intended:
                lines.append("\n**Intended Use:**")
                if intended.get("primary_use"):
                    lines.append(f"- Primary: {intended['primary_use']}")
                use_cases = intended.get("use_cases", [])
                if use_cases:
                    lines.append(f"- Use cases: {', '.join(use_cases[:3])}")

            # Limitations
            limits = ds.get("limitations", {})
            if limits:
                lines.append("\n**Limitations:**")
                for issue in limits.get("known_issues", [])[:2]:
                    lines.append(f"- {issue}")
                for bias in limits.get("biases", [])[:2]:
                    lines.append(f"- Bias: {bias}")

            # Access
            access = ds.get("access", {})
            if access:
                lines.append("\n**Access:**")
                if access.get("level"):
                    lines.append(f"- Level: {access['level']}")
                reqs = access.get("requirements", [])
                if reqs:
                    lines.append(f"- Requirements: {', '.join(reqs)}")

            # Sensitivity (Data Cards)
            sensitivity = ds.get("sensitivity", {}) or {}
            if sensitivity.get("contains_pii") is not None or sensitivity.get("risk_level"):
                lines.append("\n**Sensitivity:**")
                if sensitivity.get("contains_pii") is not None:
                    lines.append(f"- Contains PII: {'Yes' if sensitivity['contains_pii'] else 'No'}")
                if sensitivity.get("pii_categories"):
                    lines.append(f"- PII categories: {', '.join(sensitivity['pii_categories'])}")
                if sensitivity.get("consent_status"):
                    lines.append(f"- Consent: {sensitivity['consent_status']}")
                if sensitivity.get("risk_level"):
                    lines.append(f"- Risk level: {sensitivity['risk_level']}")
                if sensitivity.get("data_subjects"):
                    lines.append(f"- Data subjects: {sensitivity['data_subjects']}")

            # Preprocessing
            preprocessing = ds.get("preprocessing", {}) or {}
            steps = preprocessing.get("steps", [])
            if steps:
                lines.append("\n**Preprocessing:**")
                for step in steps[:5]:
                    lines.append(f"- {step}")
                if preprocessing.get("tools_used"):
                    lines.append(f"- Tools: {', '.join(preprocessing['tools_used'])}")

            # Annotation
            annotation = ds.get("annotation", {}) or {}
            if annotation.get("is_annotated"):
                lines.append("\n**Annotation:**")
                if annotation.get("annotation_type"):
                    lines.append(f"- Type: {annotation['annotation_type']}")
                if annotation.get("annotator_count"):
                    lines.append(f"- Annotators: {annotation['annotator_count']}")
                if annotation.get("inter_annotator_agreement"):
                    lines.append(f"- IAA: {annotation['inter_annotator_agreement']}")
                if annotation.get("label_schema"):
                    lines.append(f"- Labels: {', '.join(annotation['label_schema'])}")

            # Known models and benchmarks
            models_info = ds.get("known_models_and_benchmarks", {}) or {}
            models = models_info.get("models_trained", [])
            benchmarks = models_info.get("benchmark_results", [])
            if models or benchmarks:
                lines.append("\n**Models & Benchmarks:**")
                for m in models[:5]:
                    lines.append(f"- Model: {m}")
                for b in benchmarks[:3]:
                    lines.append(f"- Benchmark: {b}")
                if models_info.get("leaderboard_url"):
                    lines.append(f"- Leaderboard: {models_info['leaderboard_url']}")

            # Lifecycle
            lifecycle = ds.get("lifecycle", {}) or {}
            if lifecycle.get("retention_policy") or lifecycle.get("deprecation_notes"):
                lines.append("\n**Lifecycle:**")
                if lifecycle.get("retention_policy"):
                    lines.append(f"- Retention: {lifecycle['retention_policy']}")
                if lifecycle.get("deprecation_date"):
                    lines.append(f"- Deprecated: {lifecycle['deprecation_date']}")
                if lifecycle.get("deprecation_notes"):
                    lines.append(f"- Notes: {lifecycle['deprecation_notes']}")
                if lifecycle.get("update_plan"):
                    lines.append(f"- Update plan: {lifecycle['update_plan']}")

            # Data card completeness
            completeness = ds.get("data_card_completeness", {}) or {}
            if completeness.get("last_reviewed"):
                lines.append("\n**Data Card:**")
                lines.append(f"- Last reviewed: {completeness['last_reviewed']}")
                if completeness.get("reviewed_by"):
                    lines.append(f"- Reviewed by: {completeness['reviewed_by']}")

            return "\n".join(lines)

        # List of datasets
        if all(r["type"] == "dataset" for r in results):
            lines = [f"Found {len(results)} datasets:\n"]
            for r in results[:15]:
                ds = r["data"]
                name = ds.get("name", "Unknown")
                version = ds.get("version", "")
                tech = ds.get("technical", {})
                size = tech.get("size_gb", "")
                num_records = tech.get("num_records", "")

                info = f"- **{name}**"
                if version:
                    info += f" (v{version})"
                lines.append(info)

                details = []
                if size:
                    details.append(f"{size} GB")
                if num_records:
                    details.append(f"{num_records:,} records")
                access_level = ds.get("access", {}).get("level", "")
                if access_level:
                    details.append(access_level)
                if details:
                    lines.append(f"  {' | '.join(details)}")

                desc = ds.get("description", "")
                if desc:
                    lines.append(f"  {desc[:80]}..." if len(desc) > 80 else f"  {desc}")

            if len(results) > 15:
                lines.append(f"\n... and {len(results) - 15} more")
            return "\n".join(lines)

        # List of publications
        if all(r["type"] == "publication" for r in results):
            lines = [f"Found {len(results)} publications:\n"]
            for r in results[:15]:
                pub = r["data"]
                title = pub.get("title", "Untitled")
                year = pub.get("year", "")
                venue = pub.get("venue", "")
                authors = pub.get("authors", [])
                citations = pub.get("citation_count", 0)
                registry_author = pub.get("registry_author", "")

                # Format: Title (Year) - Venue
                info = f"- **{title}**"
                if year:
                    info += f" ({year})"
                if venue:
                    info += f" - {venue}"
                lines.append(info)

                # Add author and citation info on next line
                details = []
                if authors:
                    author_str = ", ".join(authors[:3])
                    if len(authors) > 3:
                        author_str += f" et al."
                    details.append(f"Authors: {author_str}")
                if citations:
                    details.append(f"Citations: {citations}")
                if details:
                    lines.append(f"  {' | '.join(details)}")

            if len(results) > 15:
                lines.append(f"\n... and {len(results) - 15} more")
            return "\n".join(lines)

        # Midway data snapshots
        if all(r["type"] == "midway_snapshot" for r in results):
            lines = [f"Found {len(results)} data snapshots on Midway:\n"]
            for r in results[:15]:
                snap = r["data"]
                name = snap.get("name", "Unknown")
                path = snap.get("path", "")
                size = snap.get("size_gb", "Unknown")
                vintage = snap.get("vintage", "")
                lines.append(f"- **{name}**")
                if vintage:
                    lines.append(f"  Date: {vintage}")
                if size:
                    lines.append(f"  Size: {size} GB")
                if path:
                    lines.append(f"  Path: `{path}`")
            if len(results) > 15:
                lines.append(f"\n... and {len(results) - 15} more")
            return "\n".join(lines)

        # Single Midway snapshot
        if len(results) == 1 and results[0]["type"] == "midway_snapshot":
            snap = results[0]["data"]
            lines = [f"**{snap.get('name')}**"]
            if snap.get("description"):
                lines.append(f"\n{snap['description']}")
            if snap.get("path"):
                lines.append(f"\n- Path: `{snap['path']}`")
            if snap.get("size_gb"):
                lines.append(f"- Size: {snap['size_gb']} GB")
            if snap.get("vintage"):
                lines.append(f"- Date: {snap['vintage']}")
            if snap.get("format"):
                lines.append(f"- Format: {snap['format']}")
            if snap.get("entity_types"):
                lines.append(f"- Contains: {', '.join(snap['entity_types'])}")
            if snap.get("owner"):
                lines.append(f"- Owner: {snap['owner']}")
            return "\n".join(lines)

        # Midway precomputed resources
        if all(r["type"] == "midway_precomputed" for r in results):
            lines = [f"Found {len(results)} precomputed resources on Midway:\n"]
            for r in results[:15]:
                res = r["data"]
                name = res.get("name", "Unknown")
                size = res.get("size_gb", "Unknown")
                irreplaceable = res.get("irreplaceable", False)
                lines.append(f"- **{name}**" + (" ⚠️ IRREPLACEABLE" if irreplaceable else ""))
                if size:
                    lines.append(f"  Size: {size} GB")
                if res.get("path"):
                    lines.append(f"  Path: `{res['path']}`")
            return "\n".join(lines)

        # Single precomputed resource
        if len(results) == 1 and results[0]["type"] == "midway_precomputed":
            res = results[0]["data"]
            lines = [f"**{res.get('name')}**"]
            if res.get("irreplaceable"):
                lines.append("\n⚠️ **IRREPLACEABLE** - This resource is expensive to reproduce")
            if res.get("description"):
                lines.append(f"\n{res['description']}")
            if res.get("path"):
                lines.append(f"\n- Path: `{res['path']}`")
            if res.get("size_gb"):
                lines.append(f"- Size: {res['size_gb']} GB")
            if res.get("owner"):
                lines.append(f"- Owner: {res['owner']}")
            if res.get("computation_time_estimate"):
                lines.append(f"- Computation time: {res['computation_time_estimate']}")
            return "\n".join(lines)

        # Midway embeddings
        if all(r["type"] == "midway_embedding" for r in results):
            lines = [f"Found {len(results)} embedding resources on Midway:\n"]
            for r in results[:15]:
                emb = r["data"]
                name = emb.get("name", "Unknown")
                domain = emb.get("domain", "")
                lines.append(f"- **{name}**" + (f" ({domain})" if domain else ""))
                if emb.get("path"):
                    lines.append(f"  Path: `{emb['path']}`")
            return "\n".join(lines)

        # Midway social media datasets
        if all(r["type"] == "midway_social" for r in results):
            lines = [f"Found {len(results)} social media datasets on Midway:\n"]
            for r in results[:15]:
                ds = r["data"]
                name = ds.get("name", "Unknown")
                platform = ds.get("platform", "")
                size = ds.get("size_gb", "")
                lines.append(f"- **{name}**" + (f" [{platform}]" if platform else ""))
                if size:
                    lines.append(f"  Size: {size} GB")
                if ds.get("path"):
                    lines.append(f"  Path: `{ds['path']}`")
            return "\n".join(lines)

        # Midway researcher directories
        if all(r["type"] == "midway_researcher" for r in results):
            lines = [f"Found {len(results)} researcher directories on Midway:\n"]
            for r in results[:15]:
                res = r["data"]
                username = res.get("username", "Unknown")
                size = res.get("size_estimate", "")
                focus = res.get("primary_focus", "")
                lines.append(f"- **{username}**" + (f" - {size}" if size else ""))
                if focus:
                    lines.append(f"  Focus: {focus}")
                if res.get("directory_path"):
                    lines.append(f"  Path: `{res['directory_path']}`")
            return "\n".join(lines)

        # Single researcher directory
        if len(results) == 1 and results[0]["type"] == "midway_researcher":
            res = results[0]["data"]
            lines = [f"**{res.get('username')}** Midway Directory"]
            if res.get("directory_path"):
                lines.append(f"\n- Path: `{res['directory_path']}`")
            if res.get("size_estimate"):
                lines.append(f"- Size: {res['size_estimate']}")
            if res.get("primary_focus"):
                lines.append(f"- Focus: {res['primary_focus']}")
            if res.get("active_projects"):
                lines.append(f"- Projects: {', '.join(res['active_projects'][:5])}")
            if res.get("assessment"):
                lines.append(f"\n*Assessment: {res['assessment']}*")
            return "\n".join(lines)

        # Midway infrastructure
        if all(r["type"] == "midway_infrastructure" for r in results):
            lines = [f"Found {len(results)} shared infrastructure on Midway:\n"]
            for r in results[:15]:
                infra = r["data"]
                name = infra.get("name", "Unknown")
                infra_type = infra.get("type", "")
                lines.append(f"- **{name}**" + (f" ({infra_type})" if infra_type else ""))
                if infra.get("path"):
                    lines.append(f"  Path: `{infra['path']}`")
                if infra.get("contents"):
                    lines.append(f"  Contains: {', '.join(infra['contents'][:5])}")
            return "\n".join(lines)

        # Single guide
        if len(results) == 1 and results[0]["type"] == "guide":
            guide = results[0]["data"]
            lines = [f"**{guide.get('title')}**"]
            if guide.get("content"):
                lines.append(f"\n{guide['content']}")
            if guide.get("steps"):
                lines.append("\n**Steps:**")
                for i, step in enumerate(guide["steps"], 1):
                    lines.append(f"{i}. {step}")
            if guide.get("prerequisites"):
                prereqs = guide["prerequisites"]
                if prereqs:
                    lines.append(f"\n**Prerequisites:** {', '.join(prereqs)}")
            if guide.get("related_guides"):
                related = guide["related_guides"]
                if related:
                    lines.append(f"\n**Related guides:** {', '.join(related)}")
            return "\n".join(lines)

        # List of guides
        if all(r["type"] == "guide" for r in results):
            lines = [f"Found {len(results)} guides:\n"]
            for r in results[:15]:
                guide = r["data"]
                title = guide.get("title", "Untitled")
                category = guide.get("category", "")
                lines.append(f"- **{title}**" + (f" [{category}]" if category else ""))
                if guide.get("content"):
                    # Show first 100 chars of content
                    preview = guide["content"][:100]
                    if len(guide["content"]) > 100:
                        preview += "..."
                    lines.append(f"  {preview}")
            if len(results) > 15:
                lines.append(f"\n... and {len(results) - 15} more")
            return "\n".join(lines)

        # Mixed results
        lines = [f"Found {len(results)} results:\n"]
        for r in results[:10]:
            if r["type"] == "person":
                lines.append(f"- [Person] {r['data'].get('name')} ({r['data'].get('role')})")
            elif r["type"] == "project":
                lines.append(f"- [Project] {r['data'].get('name')}")
            elif r["type"] == "funding":
                lines.append(f"- [Funding] {r['data'].get('name')}")
            elif r["type"] == "publication":
                pub = r["data"]
                title = pub.get("title", "Untitled")[:60]
                if len(pub.get("title", "")) > 60:
                    title += "..."
                year = pub.get("year", "")
                lines.append(f"- [Publication] {title} ({year})")
            elif r["type"] == "midway_snapshot":
                lines.append(f"- [Snapshot] {r['data'].get('name')} ({r['data'].get('vintage', '')})")
            elif r["type"] == "midway_precomputed":
                lines.append(f"- [Precomputed] {r['data'].get('name')}")
            elif r["type"] == "midway_embedding":
                lines.append(f"- [Embedding] {r['data'].get('name')}")
            elif r["type"] == "midway_social":
                lines.append(f"- [Social Data] {r['data'].get('name')} [{r['data'].get('platform', '')}]")
            elif r["type"] == "midway_researcher":
                lines.append(f"- [Researcher Dir] {r['data'].get('username')} - {r['data'].get('size_estimate', '')}")
            elif r["type"] == "midway_infrastructure":
                lines.append(f"- [Infrastructure] {r['data'].get('name')}")
            elif r["type"] == "guide":
                lines.append(f"- [Guide] {r['data'].get('title')}")
            else:
                lines.append(f"- [{r['type']}] {r['data'].get('name', r['data'].get('id', 'Unknown'))}")
        return "\n".join(lines)

    def _compound_search(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute a compound search with multiple filters (intersection logic).

        Supported filters:
        - topic: topic/research area to match
        - role: role type (student, faculty, researcher, etc.)
        - institution: institution name
        """
        from .topic_aliases import expand_topic_aliases

        # Start with all people
        candidates = list(self.lookup.list_by_type("people"))

        # Apply institution filter
        if filters.get("institution"):
            inst = filters["institution"].lower()
            candidates = [
                p for p in candidates
                if isinstance(p.get("institution"), str) and inst in p["institution"].lower()
            ]

        # Apply role filter
        if filters.get("role"):
            role = filters["role"].lower()
            role_matches = {
                "student": ["student", "phd"],
                "faculty": ["faculty", "professor", "director"],
                "researcher": ["research", "scientist", "scholar", "fellow"],
                "postdoc": ["postdoc", "postdoctoral"],
            }
            role_terms = role_matches.get(role, [role])
            candidates = [
                p for p in candidates
                if any(term in p.get("role", "").lower() for term in role_terms)
            ]

        # Apply topic filter with alias expansion
        if filters.get("topic"):
            topic = filters["topic"]
            expanded_topics = expand_topic_aliases(topic)

            matching = []
            for p in candidates:
                # Get person's topics from OpenAlex data
                person_topics = []
                if p.get("openalex", {}).get("topics"):
                    person_topics = [t.lower() if isinstance(t, str) else t.get("display_name", "").lower()
                                    for t in p["openalex"]["topics"]]

                # Check for any match
                matched_topics = []
                for pt in person_topics:
                    for et in expanded_topics:
                        if et in pt or pt in et:
                            matched_topics.append(pt)
                            break

                if matched_topics:
                    p_copy = dict(p)
                    p_copy["_matched_topics"] = list(set(matched_topics))[:5]
                    matching.append(p_copy)

            candidates = matching

        # Convert to result format
        results = []
        for p in candidates:
            results.append({
                "type": "person",
                "data": p,
                "match_reason": f"Compound: {', '.join(f'{k}={v}' for k,v in filters.items())}"
            })

        return results

    def _format_compound_answer(self, filters: Dict[str, Any], results: List[Dict]) -> str:
        """Format compound search results."""
        # Build filter description
        parts = []
        if filters.get("institution"):
            parts.append(f"at {filters['institution']}")
        if filters.get("role"):
            parts.append(f"({filters['role']}s)")
        if filters.get("topic"):
            parts.append(f"working on {filters['topic']}")

        filter_desc = " ".join(parts)

        if not results:
            return f"No researchers found {filter_desc}"

        lines = [f"Found {len(results)} researchers {filter_desc}:\n"]
        for r in results[:15]:
            p = r["data"]
            info = f"- **{p.get('name')}** ({p.get('role')})"
            if p.get("institution"):
                info += f" - {p['institution']}"
            if p.get("_matched_topics"):
                info += f"\n  Topics: {', '.join(p['_matched_topics'][:3])}"
            lines.append(info)

        if len(results) > 15:
            lines.append(f"\n... and {len(results) - 15} more")

        return "\n".join(lines)

    def get_registry_stats(self) -> Dict[str, int]:
        """Get registry statistics."""
        return self.lookup.get_stats()

    def generate_embeddings_data(self) -> List[Dict[str, str]]:
        """Generate text summaries for embedding."""
        return self.text_gen.generate_all_summaries(self.lookup.registry)
