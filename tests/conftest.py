"""
CHORUS Test Configuration and Shared Fixtures

This module provides:
- Centralized path configuration (eliminates sys.path hacks)
- Shared fixtures for common test objects
- Mock factories for external dependencies
- Test utilities and helpers
"""

import os
import sys
import time
import json
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


# =============================================================================
# Path Configuration
# =============================================================================

# Add project root to Python path (replaces sys.path.insert in each test file)
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment before any other imports
from config import load_dotenv
load_dotenv()


# =============================================================================
# Environment Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def data_dir():
    """Return the data directory path."""
    return PROJECT_ROOT / "Data"


@pytest.fixture(scope="session")
def test_data_dir():
    """Return the test data directory path."""
    test_data = PROJECT_ROOT / "tests" / "test_data"
    test_data.mkdir(exist_ok=True)
    return test_data


# =============================================================================
# Mock Fixtures - External Services
# =============================================================================

@pytest.fixture
def mock_anthropic_client():
    """
    Create a mock Anthropic client for testing without API calls.

    Usage:
        def test_something(mock_anthropic_client):
            mock_anthropic_client.messages.create.return_value = ...
    """
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Mock response")]
    mock_response.usage = MagicMock(input_tokens=10, output_tokens=20)
    mock_client.messages.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_anthropic_async():
    """Create an async mock Anthropic client."""
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Mock async response")]
    mock_response.usage = MagicMock(input_tokens=10, output_tokens=20)
    mock_client.messages.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_embedding_model():
    """
    Create a mock SentenceTransformer for testing without loading models.

    Returns embeddings as normalized random vectors.
    """
    import numpy as np

    mock_model = MagicMock()

    def mock_encode(texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        # Return normalized random vectors (768-dim like typical models)
        embeddings = np.random.randn(len(texts), 768)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings

    mock_model.encode = mock_encode
    mock_model.get_sentence_embedding_dimension.return_value = 768
    return mock_model


@pytest.fixture
def mock_reranker():
    """Create a mock CrossEncoder for testing."""
    mock_reranker = MagicMock()

    def mock_predict(pairs):
        # Return scores based on simple similarity
        return [0.8 - (0.1 * i) for i in range(len(pairs))]

    mock_reranker.predict = mock_predict
    return mock_reranker


# =============================================================================
# Sample Data Fixtures
# =============================================================================

@pytest.fixture
def sample_chunk():
    """Create a sample document chunk for testing."""
    return {
        "text": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "file_path": "/path/to/doc.pdf",
        "doc_type": "report",
        "chunk_index": 0,
        "total_chunks": 5,
        "citation": "Sample Report, 2024",
        "date_year": 2024,
    }


@pytest.fixture
def sample_chunks():
    """Create a list of sample chunks for testing."""
    return [
        {
            "text": f"This is test chunk {i} about topic {['ML', 'NLP', 'AI', 'Data', 'Research'][i % 5]}.",
            "file_path": f"/path/to/doc_{i}.pdf",
            "doc_type": ["report", "proposal", "cv", "transcript", "meeting"][i % 5],
            "chunk_index": i,
            "total_chunks": 10,
            "score": 0.9 - (i * 0.05),
        }
        for i in range(10)
    ]


@pytest.fixture
def sample_search_result():
    """Create a sample search result dictionary."""
    return {
        "query": "machine learning research",
        "results": [
            {"text": "ML result 1", "score": 0.95, "doc_type": "report"},
            {"text": "ML result 2", "score": 0.88, "doc_type": "proposal"},
            {"text": "ML result 3", "score": 0.75, "doc_type": "cv"},
        ],
        "llm_reformulation": True,
        "reranking": True,
        "mmr_diversity": True,
        "total_results": 3,
    }


@pytest.fixture
def sample_person():
    """Create a sample person registry entry."""
    return {
        "name": "Test Researcher",
        "role": "PhD Student",
        "institution": "University of Test",
        "email": "test@example.com",
        "expertise": ["machine learning", "NLP"],
    }


@pytest.fixture
def sample_project():
    """Create a sample project registry entry."""
    return {
        "name": "Test Project",
        "description": "A test project for unit testing",
        "status": "active",
        "team": ["Test Researcher"],
        "funding": "NSF-12345",
    }


# =============================================================================
# Cache Fixtures
# =============================================================================

@pytest.fixture
def clean_cache():
    """
    Ensure cache is reset before and after test.

    Yields control to the test, then cleans up.
    """
    # Import here to avoid import errors if cache not available
    try:
        from cache.query_cache import reset_cache
        reset_cache()
        yield
        reset_cache()
    except ImportError:
        yield  # No cache module, just continue


@pytest.fixture
def clean_reformulation_cache():
    """Ensure reformulation cache is reset."""
    try:
        from cache.reformulation_cache import reset_reformulation_cache
        reset_reformulation_cache()
        yield
        reset_reformulation_cache()
    except ImportError:
        yield


# =============================================================================
# RAG Server Fixtures
# =============================================================================

@pytest.fixture
def mock_rag_server():
    """Create a mock RAG server for endpoint testing."""
    mock_rag = MagicMock()
    mock_rag.search.return_value = [
        {"text": "Result 1", "score": 0.9},
        {"text": "Result 2", "score": 0.8},
    ]
    mock_rag.reload_if_changed.return_value = False
    return mock_rag


@pytest.fixture
def rag_server_url():
    """Return the RAG server URL for integration tests."""
    return os.environ.get("RAG_SERVER_URL", "http://localhost:8765")


# =============================================================================
# Test Markers Configuration
# =============================================================================

def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line("markers", "unit: Fast isolated unit tests")
    config.addinivalue_line("markers", "integration: Multi-component tests")
    config.addinivalue_line("markers", "performance: Speed benchmarks")
    config.addinivalue_line("markers", "slow: Tests > 5 seconds")
    config.addinivalue_line("markers", "ner: Tests requiring spaCy")
    config.addinivalue_line("markers", "llm: Tests requiring LLM API")
    config.addinivalue_line("markers", "rag: Tests requiring RAG server")


# =============================================================================
# Skip Conditions
# =============================================================================

# Skip if spaCy not installed
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

skip_no_spacy = pytest.mark.skipif(
    not SPACY_AVAILABLE,
    reason="spaCy not installed"
)

# Skip if RAG server not running
def is_rag_server_running():
    """Check if RAG server is available."""
    import socket
    try:
        with socket.create_connection(("localhost", 8765), timeout=1):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False

skip_no_rag_server = pytest.mark.skipif(
    not is_rag_server_running(),
    reason="RAG server not running on localhost:8765"
)


# =============================================================================
# Utility Functions
# =============================================================================

def assert_valid_search_result(result: Dict[str, Any]) -> None:
    """Assert that a search result has the expected structure."""
    assert "query" in result or "results" in result
    if "results" in result:
        assert isinstance(result["results"], list)
        for r in result["results"]:
            assert "text" in r or "content" in r
            assert "score" in r


def create_temp_json_file(data: Any, tmp_path: Path, filename: str = "test.json") -> Path:
    """Create a temporary JSON file for testing."""
    file_path = tmp_path / filename
    with open(file_path, "w") as f:
        json.dump(data, f)
    return file_path
