"""
Unit tests for FastAPI RAG server.

Uses FastAPI's TestClient for synchronous testing.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_rag_server():
    """Create a mock RAGServer."""
    mock = Mock()
    mock.chunks = [Mock() for _ in range(100)]
    mock.llm_reformulator = Mock()
    mock.llm_reformulator.enabled = True
    mock.reranker = Mock()
    mock.reload_if_changed = Mock()
    mock.force_reload = Mock(return_value={
        "status": "reloaded",
        "previous_chunks": 100,
        "current_chunks": 105,
        "timestamp": "2024-01-01T00:00:00",
    })
    mock.search = Mock(return_value=[
        {
            "text": "Test result text",
            "citation": "test.pdf | p.1",
            "source_file": "test.pdf",
            "doc_type": "proposal",
            "score": 0.95,
        }
    ])
    mock.list_sources = Mock(return_value=[
        {"file": "test.pdf", "chunks": 10, "doc_type": "proposal"},
    ])
    mock.list_doc_types = Mock(return_value={"proposal": 50, "transcript": 30})
    return mock


@pytest.fixture
def mock_registry_rag():
    """Create a mock RegistryRAG."""
    mock = Mock()
    mock.reload_if_changed = Mock()
    mock.query = Mock(return_value={
        "classification": "lookup",
        "structured_results": [],
        "answer": "Test answer",
    })
    mock.get_registry_stats = Mock(return_value={"people": 50, "projects": 10})
    mock.lookup = Mock()
    mock.lookup.get_all_topics = Mock(return_value={"AI": 10, "ML": 5})
    mock.lookup.find_person = Mock(return_value={"name": "Test Person"})
    mock.lookup.find_people_by_topic = Mock(return_value=[])
    mock.lookup.find_publications = Mock(return_value=[])
    mock.lookup.get_all_publications = Mock(return_value=[])
    mock.lookup.registry = {"people": {"members": []}}
    return mock


@pytest.fixture
def client(mock_rag_server, mock_registry_rag):
    """Create test client with mocked dependencies."""
    # Import here to avoid loading models during test collection
    with patch.dict('sys.modules', {
        'rag_http_server': MagicMock(
            RAGServer=Mock(return_value=mock_rag_server),
            ENABLE_MMR=True,
            MMR_LAMBDA=0.7,
            RERANK_MODEL="test-model",
            REGISTRY_AVAILABLE=True,
            SEMANTIC_REGISTRY_AVAILABLE=False,
            HYPERGRAPH_AVAILABLE=False,
        ),
    }):
        # Patch the global state before importing
        import rag_server_fastapi
        rag_server_fastapi.rag = mock_rag_server
        rag_server_fastapi.registry_rag = mock_registry_rag
        rag_server_fastapi.semantic_registry = None
        rag_server_fastapi.hypergraph = None

        from fastapi.testclient import TestClient
        return TestClient(rag_server_fastapi.app)


# =============================================================================
# Health Endpoint Tests
# =============================================================================

class TestHealthEndpoints:
    """Test health and info endpoints."""

    def test_root_endpoint(self, client):
        """Test / returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert data["name"] == "CHORUS RAG Server"

    def test_health_check(self, client):
        """Test /health returns status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "chunks" in data
        assert "llm_reformulation_enabled" in data


# =============================================================================
# Search Endpoint Tests
# =============================================================================

class TestSearchEndpoints:
    """Test search functionality."""

    def test_search_basic(self, client, mock_rag_server):
        """Test /search with minimal params."""
        response = client.get("/search?q=machine+learning")
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "machine learning"
        assert "results" in data
        mock_rag_server.search.assert_called()

    def test_search_missing_query(self, client):
        """Test /search without query returns 422."""
        response = client.get("/search")
        assert response.status_code == 422  # FastAPI validation error

    def test_search_with_top_k(self, client, mock_rag_server):
        """Test /search with custom top_k."""
        response = client.get("/search?q=test&top_k=10")
        assert response.status_code == 200
        call_kwargs = mock_rag_server.search.call_args.kwargs
        assert call_kwargs["top_k"] == 10

    def test_search_with_filters(self, client, mock_rag_server):
        """Test /search with doc_type filter."""
        response = client.get("/search?q=test&doc_type=proposal")
        assert response.status_code == 200
        call_kwargs = mock_rag_server.search.call_args.kwargs
        assert call_kwargs["filters"]["doc_type"] == "proposal"

    def test_search_with_all_options(self, client, mock_rag_server):
        """Test /search with all parameters."""
        response = client.get(
            "/search?q=test&top_k=3&llm=false&rerank=false&mmr=false"
        )
        assert response.status_code == 200
        call_kwargs = mock_rag_server.search.call_args.kwargs
        assert call_kwargs["use_llm_reformulation"] is False
        assert call_kwargs["use_reranking"] is False
        assert call_kwargs["use_mmr"] is False

    def test_search_top_k_validation(self, client):
        """Test /search rejects invalid top_k."""
        response = client.get("/search?q=test&top_k=100")
        assert response.status_code == 422  # Exceeds max (50)


# =============================================================================
# Context Endpoint Tests
# =============================================================================

class TestContextEndpoint:
    """Test context endpoint."""

    def test_context_basic(self, client):
        """Test /context returns formatted context."""
        response = client.get("/context?q=test+question")
        assert response.status_code == 200
        data = response.json()
        assert "context" in data
        assert "citations" in data


# =============================================================================
# Sources Endpoint Tests
# =============================================================================

class TestSourcesEndpoint:
    """Test sources listing."""

    def test_sources_list(self, client):
        """Test /sources lists indexed files."""
        response = client.get("/sources")
        assert response.status_code == 200
        data = response.json()
        assert "total_sources" in data
        assert "sources" in data

    def test_sources_with_filter(self, client, mock_rag_server):
        """Test /sources with doc_type filter."""
        mock_rag_server.list_sources.return_value = [
            {"file": "test.pdf", "chunks": 10, "doc_type": "proposal"},
            {"file": "meeting.txt", "chunks": 5, "doc_type": "transcript"},
        ]
        response = client.get("/sources?doc_type=proposal")
        assert response.status_code == 200
        data = response.json()
        # Should filter to only proposals
        for source in data["sources"]:
            assert source["doc_type"] == "proposal"


# =============================================================================
# Doc Types Endpoint Tests
# =============================================================================

class TestDocTypesEndpoint:
    """Test document types listing."""

    def test_doc_types(self, client):
        """Test /doc_types returns type counts."""
        response = client.get("/doc_types")
        assert response.status_code == 200
        data = response.json()
        assert "doc_types" in data


# =============================================================================
# Reload Endpoint Tests
# =============================================================================

class TestReloadEndpoint:
    """Test reload functionality."""

    def test_reload_index(self, client, mock_rag_server):
        """Test /reload forces index reload."""
        response = client.post("/reload")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "reloaded"
        mock_rag_server.force_reload.assert_called_once()


# =============================================================================
# Registry Endpoint Tests
# =============================================================================

class TestRegistryEndpoints:
    """Test registry integration."""

    def test_registry_query(self, client, mock_registry_rag):
        """Test /registry query."""
        response = client.get("/registry?q=who+is+Jake")
        assert response.status_code == 200
        mock_registry_rag.query.assert_called()

    def test_registry_stats(self, client):
        """Test /registry/stats."""
        response = client.get("/registry/stats")
        assert response.status_code == 200
        data = response.json()
        assert "stats" in data

    def test_registry_topics(self, client):
        """Test /registry/topics."""
        response = client.get("/registry/topics")
        assert response.status_code == 200
        data = response.json()
        assert "total_topics" in data

    def test_registry_person(self, client, mock_registry_rag):
        """Test /registry/person lookup."""
        response = client.get("/registry/person?name=Test")
        assert response.status_code == 200
        data = response.json()
        assert data["found"] is True


# =============================================================================
# Hybrid Endpoint Tests
# =============================================================================

class TestHybridEndpoint:
    """Test hybrid search."""

    def test_hybrid_search(self, client):
        """Test /hybrid combines registry and RAG."""
        response = client.get("/hybrid?q=test+query")
        assert response.status_code == 200
        data = response.json()
        assert data["hybrid"] is True
        assert "classification" in data
        assert "semantic_results" in data


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Test error responses."""

    def test_invalid_endpoint(self, client):
        """Test non-existent endpoint returns 404."""
        response = client.get("/nonexistent")
        assert response.status_code == 404


# =============================================================================
# API Documentation Tests
# =============================================================================

class TestAPIDocumentation:
    """Test auto-generated docs."""

    def test_openapi_schema(self, client):
        """Test /openapi.json is available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "paths" in schema
        assert "/search" in schema["paths"]

    def test_docs_ui(self, client):
        """Test /docs UI is available."""
        response = client.get("/docs")
        assert response.status_code == 200


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
