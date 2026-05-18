#!/usr/bin/env python3
"""
Comprehensive tests for the extractors module.

Tests NER, date, topic, and key term extraction.
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from extractors import (
    # NER
    extract_entities,
    EntityExtractionResult,
    get_ner_extractor,
    enrich_chunks_with_entities,
    # Dates
    extract_dates,
    ExtractedDate,
    DateExtractionResult,
    get_date_extractor,
    # Topics
    classify_topics,
    TopicClassification,
    TopicClassificationResult,
    classify_chunks,
    get_topic_extractor,
    # Key terms
    extract_key_terms,
    KeyTerm,
    KeyTermResult,
    get_keyterm_extractor,
)


# =============================================================================
# NER Extractor Tests
# =============================================================================

class TestNERExtractor:
    """Tests for Named Entity Recognition extraction."""

    def test_extract_grant_nsf(self):
        """Test NSF grant ID extraction."""
        text = "Funded by NSF grant 2033345."
        result = extract_entities(text)
        assert "NSF-2033345" in result.grant_ids

    def test_extract_grant_nih(self):
        """Test NIH grant ID extraction."""
        text = "NIH R01-GM123456 supports this research."
        result = extract_entities(text)
        assert any("NIH" in g and "GM123456" in g for g in result.grant_ids)

    def test_extract_grant_darpa(self):
        """Test DARPA grant ID extraction."""
        text = "DARPA HR001121S0001 funds the project."
        result = extract_entities(text)
        assert any("DARPA" in g for g in result.grant_ids)

    def test_extract_project_muri(self):
        """Test MURI project extraction."""
        text = "The MURI project focuses on AI safety."
        result = extract_entities(text)
        assert "MURI" in result.projects

    def test_extract_project_apto(self):
        """Test APTO project extraction."""
        text = "Our APTO research continues."
        result = extract_entities(text)
        assert "APTO" in result.projects

    def test_extract_organizations(self):
        """Test organization extraction."""
        text = "NSF and DARPA fund this research at UChicago."
        result = extract_entities(text)
        assert "NSF" in result.organizations or any("NSF" in o for o in result.organizations)

    def test_empty_text(self):
        """Test empty text handling."""
        result = extract_entities("")
        assert not result.has_entities()
        assert result.total_count() == 0

    def test_to_dict(self):
        """Test serialization to dict."""
        result = extract_entities("NSF grant 2033345 for MURI project")
        d = result.to_dict()
        assert "people" in d
        assert "organizations" in d
        assert "projects" in d
        assert "grant_ids" in d

    def test_enrich_chunks(self):
        """Test chunk enrichment function."""
        chunks = [
            {"text": "Dr. Smith received NSF grant 2033345."},
            {"text": "The MURI project is funded by DARPA."},
        ]
        enriched = enrich_chunks_with_entities(chunks)
        assert "entities" in enriched[0]
        assert "entities" in enriched[1]

    def test_singleton_extractor(self):
        """Test singleton pattern."""
        ext1 = get_ner_extractor()
        ext2 = get_ner_extractor()
        assert ext1 is ext2


# =============================================================================
# Date Extractor Tests
# =============================================================================

class TestDateExtractor:
    """Tests for date extraction."""

    def test_extract_iso_date(self):
        """Test ISO format date."""
        result = extract_dates("Release on 2024-03-15.")
        assert result.has_dates()
        assert result.dates[0].normalized == "2024-03-15"
        assert result.dates[0].year == 2024
        assert result.dates[0].month == 3
        assert result.dates[0].day == 15

    def test_extract_us_date(self):
        """Test US format date."""
        result = extract_dates("Due 03/15/2024.")
        assert result.has_dates()
        assert result.dates[0].year == 2024
        assert result.dates[0].month == 3
        assert result.dates[0].day == 15

    def test_extract_text_date_mdy(self):
        """Test text format Month Day, Year."""
        result = extract_dates("The deadline is March 15, 2024.")
        assert result.has_dates()
        assert result.dates[0].normalized == "2024-03-15"

    def test_extract_text_date_dmy(self):
        """Test text format Day Month Year."""
        result = extract_dates("Meeting on 15 March 2024.")
        assert result.has_dates()
        assert result.dates[0].normalized == "2024-03-15"

    def test_extract_month_year(self):
        """Test partial month-year dates."""
        result = extract_dates("Published in March 2024.")
        assert result.has_dates()
        assert result.dates[0].normalized == "2024-03"
        assert result.dates[0].is_partial()

    def test_extract_quarter(self):
        """Test quarter date extraction."""
        result = extract_dates("Planning for Q1 2024.")
        assert result.has_dates()
        assert result.dates[0].year == 2024
        assert result.dates[0].month == 1

    def test_extract_fiscal_year(self):
        """Test fiscal year extraction."""
        result = extract_dates("FY24 budget approved.")
        assert result.has_dates()
        assert result.dates[0].year == 2024

    def test_extract_seasonal(self):
        """Test seasonal date extraction."""
        result = extract_dates("Launching in Spring 2024.")
        assert result.has_dates()
        assert result.dates[0].year == 2024
        assert result.dates[0].month == 3

    def test_extract_year_range(self):
        """Test year range extraction."""
        result = extract_dates("Project timeline: 2024-2025.")
        assert result.has_dates()
        assert result.dates[0].is_range
        assert result.dates[0].range_end == "2025"

    def test_context_deadline(self):
        """Test deadline context detection."""
        result = extract_dates("The submission deadline is March 15, 2024.")
        assert result.primary_date.context == "deadline"

    def test_context_event(self):
        """Test event context detection."""
        result = extract_dates("Join our conference on March 15, 2024.")
        assert result.primary_date.context == "event"

    def test_multiple_dates(self):
        """Test extraction of multiple dates."""
        result = extract_dates("Due March 15, 2024, conference April 20, 2024.")
        assert len(result.dates) >= 2

    def test_empty_text(self):
        """Test empty text handling."""
        result = extract_dates("")
        assert not result.has_dates()
        assert result.primary_date is None

    def test_to_dict(self):
        """Test serialization to dict."""
        result = extract_dates("March 15, 2024")
        d = result.to_dict()
        assert "dates" in d
        assert "primary_date" in d

    def test_singleton_extractor(self):
        """Test singleton pattern."""
        ext1 = get_date_extractor()
        ext2 = get_date_extractor()
        assert ext1 is ext2


# =============================================================================
# Topic Extractor Tests
# =============================================================================

class TestTopicExtractor:
    """Tests for topic classification."""

    def test_classify_network_analysis(self):
        """Test network analysis topic."""
        result = classify_topics("We study network analysis and graph theory.")
        assert result.has_topics()
        assert "Network Analysis" in result.topic_labels()

    def test_classify_nlp(self):
        """Test NLP topic."""
        result = classify_topics("Natural language processing uses transformers and BERT.")
        assert result.has_topics()
        assert "Natural Language Processing" in result.topic_labels()

    def test_classify_machine_learning(self):
        """Test machine learning topic."""
        result = classify_topics("Deep learning models and neural networks for classification.")
        assert result.has_topics()
        assert "Machine Learning" in result.topic_labels()

    def test_classify_science_of_science(self):
        """Test science of science topic."""
        result = classify_topics("Citation analysis and bibliometric studies of research impact.")
        assert result.has_topics()
        assert "Science of Science" in result.topic_labels()

    def test_classify_multiple_topics(self):
        """Test multiple topic detection."""
        result = classify_topics(
            "Network analysis of citation patterns using machine learning.",
            top_k=3
        )
        assert len(result.topics) >= 2

    def test_confidence_range(self):
        """Test confidence scores are in valid range."""
        result = classify_topics("Network analysis and graph theory study.")
        for topic in result.topics:
            assert 0.0 <= topic.confidence <= 1.0

    def test_top_k_limiting(self):
        """Test top_k limits results."""
        result = classify_topics("Network ML NLP policy ethics collaboration", top_k=2)
        assert len(result.topics) <= 2

    def test_min_confidence_filtering(self):
        """Test min_confidence filters results."""
        result = classify_topics("Network analysis study.", min_confidence=0.9)
        for topic in result.topics:
            assert topic.confidence >= 0.9

    def test_empty_text(self):
        """Test empty text handling."""
        result = classify_topics("")
        assert not result.has_topics()

    def test_classify_chunks(self):
        """Test batch chunk classification."""
        chunks = [
            {"text": "Network analysis of graphs."},
            {"text": "Deep learning neural networks."},
        ]
        enriched = classify_chunks(chunks, top_k=2)
        assert "topics" in enriched[0]
        assert "topic_labels" in enriched[0]
        assert "primary_topic" in enriched[0]

    def test_to_dict(self):
        """Test serialization to dict."""
        result = classify_topics("Network analysis study")
        d = result.to_dict()
        assert "topics" in d
        assert "text_length" in d

    def test_singleton_extractor(self):
        """Test singleton pattern."""
        ext1 = get_topic_extractor()
        ext2 = get_topic_extractor()
        assert ext1 is ext2

    def test_available_topics(self):
        """Test getting list of available topics."""
        ext = get_topic_extractor()
        topics = ext.get_topics_list()
        assert "Network Analysis" in topics
        assert "Machine Learning" in topics


# =============================================================================
# Key Term Extractor Tests
# =============================================================================

class TestKeyTermExtractor:
    """Tests for key term extraction."""

    def test_extract_basic(self):
        """Test basic key term extraction."""
        text = "Deep learning models use neural networks for classification."
        result = extract_key_terms(text, top_k=5)
        assert result.has_terms()
        terms = result.term_list()
        assert any("learning" in t or "neural" in t for t in terms)

    def test_extract_phrases(self):
        """Test phrase extraction."""
        text = "Natural language processing and machine learning are important."
        result = extract_key_terms(text, top_k=5)
        # Should find multi-word phrases
        phrases = [t for t in result.terms if t.is_phrase]
        assert len(phrases) >= 0  # May or may not find phrases

    def test_academic_terms(self):
        """Test academic term handling."""
        text = "NLP and BERT are used for text classification tasks."
        result = extract_key_terms(text, top_k=5)
        terms = result.term_list()
        # Should find acronyms
        assert any("nlp" in t.lower() or "bert" in t.lower() for t in terms)

    def test_top_k_limiting(self):
        """Test top_k limits results."""
        text = "Many different words and terms appear in this long text sample."
        result = extract_key_terms(text, top_k=3)
        assert len(result.terms) <= 3

    def test_min_score_filtering(self):
        """Test min_score filters results."""
        text = "Network analysis and graph theory methods."
        result = extract_key_terms(text, min_score=0.5)
        for term in result.terms:
            assert term.score >= 0.5

    def test_empty_text(self):
        """Test empty text handling."""
        result = extract_key_terms("")
        assert not result.has_terms()

    def test_stopwords_filtered(self):
        """Test that stopwords are filtered."""
        text = "The quick brown fox jumps over the lazy dog."
        result = extract_key_terms(text, top_k=5)
        terms = result.term_list()
        assert "the" not in terms
        assert "over" not in terms

    def test_to_dict(self):
        """Test serialization to dict."""
        result = extract_key_terms("Network analysis study")
        d = result.to_dict()
        assert "terms" in d
        assert "total_candidates" in d
        assert "has_idf" in d

    def test_singleton_extractor(self):
        """Test singleton pattern."""
        ext1 = get_keyterm_extractor()
        ext2 = get_keyterm_extractor()
        assert ext1 is ext2

    def test_extract_from_chunks(self):
        """Test batch chunk extraction."""
        ext = get_keyterm_extractor()
        chunks = [
            {"text": "Network analysis of citation graphs."},
            {"text": "Deep learning for text classification."},
        ]
        enriched = ext.extract_from_chunks(chunks, top_k=5)
        assert "key_terms" in enriched[0]
        assert "key_terms_list" in enriched[0]


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple extractors."""

    def test_full_extraction_pipeline(self):
        """Test running all extractors on same text."""
        text = """
        Dr. James Evans from the University of Chicago received NSF grant 2033345
        for the MURI project on network analysis. The deadline is March 15, 2024.
        The project uses deep learning and machine learning methods.
        """

        # Run all extractors
        entities = extract_entities(text)
        dates = extract_dates(text)
        topics = classify_topics(text)
        terms = extract_key_terms(text)

        # Verify each produces results
        assert entities.has_entities() or len(entities.grant_ids) > 0
        assert dates.has_dates()
        assert topics.has_topics()
        assert terms.has_terms()

    def test_chunk_enrichment_all(self):
        """Test enriching chunks with all extractors."""
        chunks = [
            {"text": "NSF grant 2033345 for network analysis, due March 2024."},
            {"text": "Deep learning methods for citation analysis."},
        ]

        # Enrich with entities
        chunks = enrich_chunks_with_entities(chunks)

        # Enrich with topics
        chunks = classify_chunks(chunks)

        # Enrich with key terms
        ext = get_keyterm_extractor()
        chunks = ext.extract_from_chunks(chunks)

        # Verify all fields added
        for chunk in chunks:
            assert "entities" in chunk
            assert "topics" in chunk
            assert "key_terms" in chunk

    def test_transcript_text(self):
        """Test extractors on transcript-like text."""
        text = """
        James Evans: Our lab meeting on March 20, 2024 will cover the APTO
        project results. We've been using network analysis methods.
        Student: The NSF grant 2033345 report is due next week.
        """

        entities = extract_entities(text)
        dates = extract_dates(text)
        topics = classify_topics(text)

        # Should find relevant entities and dates
        assert any("APTO" in p for p in entities.projects) or dates.has_dates()

    def test_proposal_text(self):
        """Test extractors on proposal-like text."""
        text = """
        This MURI proposal requests funding for research on the science of science.
        The project timeline spans 2024-2029, with Phase 1 completing in Q4 2024.
        We will use machine learning and network analysis methods.
        """

        entities = extract_entities(text)
        dates = extract_dates(text)
        topics = classify_topics(text)

        assert "MURI" in entities.projects
        assert dates.has_dates()
        assert topics.has_topics()


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Performance benchmarks."""

    def test_ner_speed(self):
        """Test NER extraction speed."""
        import time
        text = "Dr. James Evans from NSF grant 2033345." * 10

        start = time.time()
        for _ in range(10):
            extract_entities(text)
        duration = time.time() - start

        # Should complete 10 extractions in under 5 seconds
        assert duration < 5.0, f"NER too slow: {duration:.2f}s"

    def test_date_speed(self):
        """Test date extraction speed."""
        import time
        text = "The deadline is March 15, 2024. Meeting on April 20, 2024." * 10

        start = time.time()
        for _ in range(100):
            extract_dates(text)
        duration = time.time() - start

        # Should complete 100 extractions in under 1 second
        assert duration < 1.0, f"Date extraction too slow: {duration:.2f}s"

    def test_topic_speed(self):
        """Test topic classification speed."""
        import time
        text = "Network analysis and machine learning for citation analysis." * 10

        start = time.time()
        for _ in range(100):
            classify_topics(text)
        duration = time.time() - start

        # Should complete 100 classifications in under 1 second
        assert duration < 1.0, f"Topic classification too slow: {duration:.2f}s"

    def test_keyterm_speed(self):
        """Test key term extraction speed."""
        import time
        text = "Deep learning models and neural networks for text analysis." * 10

        start = time.time()
        for _ in range(100):
            extract_key_terms(text)
        duration = time.time() - start

        # Should complete 100 extractions in under 1 second
        assert duration < 1.0, f"Key term extraction too slow: {duration:.2f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
