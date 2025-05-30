"""
Test configuration and fixtures for the search agent.
"""

import os
import pytest
import asyncio
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock
from pathlib import Path

# Test configuration
TEST_CONFIG = {
    "GOOGLE_API_KEY": "test_google_api_key",
    "GOOGLE_CSE_ID": "test_cse_id",
    "BRAVE_API_KEY": "test_brave_api_key", 
    "GEMINI_API_KEY": "test_gemini_api_key",
    "CORE_API_KEY": "test_core_api_key",
    "LOG_LEVEL": "DEBUG",
    "CACHE_TYPE": "memory",
    "RATE_LIMIT_ENABLED": "false",
}

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    config = Mock()
    for key, value in TEST_CONFIG.items():
        setattr(config, key.lower(), value)
    return config

@pytest.fixture
def sample_search_results():
    """Sample search results for testing."""
    return {
        "query": "artificial intelligence machine learning",
        "optimized_query": "artificial intelligence machine learning applications",
        "context": {
            "background": "Artificial intelligence and machine learning are...",
            "key_concepts": ["AI", "ML", "neural networks", "deep learning"]
        },
        "results": {
            "web_search": [
                {
                    "title": "AI and ML Overview",
                    "url": "https://example.com/ai-ml",
                    "snippet": "Comprehensive guide to AI and ML...",
                    "source": "google_search"
                }
            ],
            "academic_papers": [
                {
                    "title": "Deep Learning for AI",
                    "authors": ["John Doe", "Jane Smith"],
                    "abstract": "This paper discusses...",
                    "url": "https://arxiv.org/abs/1234.5678",
                    "source": "arxiv"
                }
            ],
            "wikipedia": [
                {
                    "title": "Artificial Intelligence",
                    "summary": "AI is the simulation of human intelligence...",
                    "url": "https://en.wikipedia.org/wiki/Artificial_intelligence",
                    "source": "wikipedia"
                }
            ]
        },
        "summary": "The search found comprehensive information about AI and ML...",
        "metadata": {
            "total_results": 15,
            "search_time": 2.5,
            "sources_used": ["google_search", "arxiv", "wikipedia"],
            "timestamp": "2024-01-01T12:00:00Z"
        }
    }

@pytest.fixture
def mock_wikipedia_response():
    """Mock Wikipedia API response."""
    return {
        "query": {
            "search": [
                {
                    "title": "Artificial intelligence",
                    "snippet": "AI is the simulation of human intelligence processes by machines",
                    "size": 150000,
                    "wordcount": 12000,
                    "timestamp": "2024-01-01T00:00:00Z"
                }
            ]
        },
        "parse": {
            "title": "Artificial intelligence",
            "text": {
                "*": "Artificial intelligence (AI) is the simulation of human intelligence..."
            }
        }
    }

@pytest.fixture
def mock_google_response():
    """Mock Google Custom Search API response."""
    return {
        "items": [
            {
                "title": "AI and Machine Learning Guide",
                "link": "https://example.com/ai-guide",
                "snippet": "Comprehensive guide to artificial intelligence and machine learning",
                "displayLink": "example.com"
            },
            {
                "title": "Deep Learning Tutorial", 
                "link": "https://example.com/deep-learning",
                "snippet": "Learn deep learning fundamentals and applications",
                "displayLink": "example.com"
            }
        ],
        "searchInformation": {
            "totalResults": "1000000",
            "searchTime": 0.45
        }
    }

@pytest.fixture
def mock_arxiv_response():
    """Mock ArXiv API response."""
    return """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
        <entry>
            <id>http://arxiv.org/abs/1234.5678v1</id>
            <title>Deep Learning for Artificial Intelligence</title>
            <summary>This paper presents novel approaches to deep learning...</summary>
            <author>
                <name>John Doe</name>
            </author>
            <author>
                <name>Jane Smith</name>
            </author>
            <published>2024-01-01T00:00:00Z</published>
            <updated>2024-01-01T00:00:00Z</updated>
            <link href="http://arxiv.org/abs/1234.5678v1" rel="alternate" type="text/html"/>
            <link title="pdf" href="http://arxiv.org/pdf/1234.5678v1.pdf" rel="related" type="application/pdf"/>
        </entry>
    </feed>"""

@pytest.fixture
def mock_semantic_scholar_response():
    """Mock Semantic Scholar API response."""
    return {
        "data": [
            {
                "paperId": "123456789",
                "title": "Advances in Machine Learning",
                "abstract": "This paper reviews recent advances in machine learning...",
                "authors": [
                    {"name": "Alice Johnson"},
                    {"name": "Bob Wilson"}
                ],
                "year": 2024,
                "url": "https://www.semanticscholar.org/paper/123456789",
                "citationCount": 42,
                "influentialCitationCount": 15
            }
        ],
        "total": 1000
    }

@pytest.fixture
def mock_gemini_response():
    """Mock Google Gemini API response."""
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": '{"optimized_query": "artificial intelligence machine learning applications", "key_concepts": ["AI", "ML", "neural networks"], "search_strategy": "academic_focus"}'
                        }
                    ]
                },
                "finishReason": "STOP"
            }
        ]
    }

@pytest.fixture
async def mock_search_tool():
    """Mock search tool for testing."""
    tool = AsyncMock()
    tool.name = "test_tool"
    tool.search.return_value = [
        {
            "title": "Test Result",
            "url": "https://example.com/test",
            "snippet": "Test snippet",
            "source": "test_tool"
        }
    ]
    return tool

@pytest.fixture
def test_data_dir():
    """Get the test data directory."""
    return Path(__file__).parent / "data"

@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir

@pytest.fixture
def mock_http_client():
    """Mock HTTP client for testing API calls."""
    client = AsyncMock()
    client.get.return_value.status_code = 200
    client.get.return_value.json.return_value = {"test": "data"}
    client.post.return_value.status_code = 200
    client.post.return_value.json.return_value = {"test": "data"}
    return client

# Test utilities
def assert_search_result_structure(result: Dict[str, Any]):
    """Assert that a search result has the expected structure."""
    required_fields = ["query", "results", "metadata"]
    for field in required_fields:
        assert field in result, f"Missing required field: {field}"
    
    assert "timestamp" in result["metadata"]
    assert "search_time" in result["metadata"]
    assert "sources_used" in result["metadata"]

def assert_valid_url(url: str):
    """Assert that a URL is valid."""
    assert url.startswith(("http://", "https://")), f"Invalid URL: {url}"

def create_mock_response(status_code: int = 200, json_data: Dict = None, text: str = None):
    """Create a mock HTTP response."""
    response = Mock()
    response.status_code = status_code
    if json_data:
        response.json.return_value = json_data
    if text:
        response.text = text
    return response
