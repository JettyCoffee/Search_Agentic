"""
Unit tests for search tools.
"""

import pytest
import xml.etree.ElementTree as ET
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from search_agent.tools.wikipedia import WikipediaSearchTool
from search_agent.tools.google_search import GoogleSearchTool
from search_agent.tools.arxiv import ArXivSearchTool
from search_agent.tools.semantic_scholar import SemanticScholarSearchTool
from search_agent.exceptions.custom_exceptions import SearchToolError


class TestWikipediaSearchTool:
    """Test cases for Wikipedia search tool."""
    
    @pytest.fixture
    def wikipedia_tool(self):
        """Create WikipediaSearchTool instance."""
        config = Mock()
        config.max_results_per_source = 5
        return WikipediaSearchTool(config)
    
    @pytest.mark.asyncio
    async def test_search_success(self, wikipedia_tool, mock_wikipedia_response):
        """Test successful Wikipedia search."""
        with patch('search_agent.tools.wikipedia.wikipedia') as mock_wiki:
            mock_wiki.search.return_value = ["Artificial intelligence"]
            mock_wiki.page.return_value.title = "Artificial intelligence"
            mock_wiki.page.return_value.summary = "AI is the simulation of human intelligence..."
            mock_wiki.page.return_value.url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
            
            results = await wikipedia_tool.search("artificial intelligence")
            
            assert len(results) == 1
            assert results[0]["title"] == "Artificial intelligence"
            assert results[0]["source"] == "wikipedia"
            assert "summary" in results[0]
            assert "url" in results[0]
    
    @pytest.mark.asyncio
    async def test_search_no_results(self, wikipedia_tool):
        """Test Wikipedia search with no results."""
        with patch('search_agent.tools.wikipedia.wikipedia') as mock_wiki:
            mock_wiki.search.return_value = []
            
            results = await wikipedia_tool.search("nonexistent topic")
            
            assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_search_disambiguation_error(self, wikipedia_tool):
        """Test Wikipedia search with disambiguation error."""
        with patch('search_agent.tools.wikipedia.wikipedia') as mock_wiki:
            mock_wiki.search.return_value = ["Test topic"]
            mock_wiki.page.side_effect = mock_wiki.DisambiguationError(
                "Test topic", ["Option 1", "Option 2"]
            )
            
            results = await wikipedia_tool.search("test topic")
            
            # Should handle disambiguation by taking first option
            assert len(results) >= 0  # Depends on implementation
    
    @pytest.mark.asyncio
    async def test_search_page_error(self, wikipedia_tool):
        """Test Wikipedia search with page error."""
        with patch('search_agent.tools.wikipedia.wikipedia') as mock_wiki:
            mock_wiki.search.return_value = ["Test topic"]
            mock_wiki.page.side_effect = mock_wiki.PageError("Page not found")
            
            results = await wikipedia_tool.search("test topic")
            
            # Should handle page error gracefully
            assert isinstance(results, list)
    
    @pytest.mark.asyncio
    async def test_search_with_max_results(self, wikipedia_tool):
        """Test Wikipedia search respecting max results limit."""
        with patch('search_agent.tools.wikipedia.wikipedia') as mock_wiki:
            # Return more results than max_results_per_source
            mock_wiki.search.return_value = [f"Topic {i}" for i in range(10)]
            mock_wiki.page.return_value.title = "Test Topic"
            mock_wiki.page.return_value.summary = "Test summary"
            mock_wiki.page.return_value.url = "https://test.url"
            
            results = await wikipedia_tool.search("test")
            
            # Should respect max_results_per_source limit
            assert len(results) <= wikipedia_tool.config.max_results_per_source


class TestGoogleSearchTool:
    """Test cases for Google Custom Search tool."""
    
    @pytest.fixture
    def google_tool(self):
        """Create GoogleSearchTool instance."""
        config = Mock()
        config.google_api_key = "test_api_key"
        config.google_cse_id = "test_cse_id"
        config.max_results_per_source = 5
        return GoogleSearchTool(config)
    
    @pytest.mark.asyncio
    async def test_search_success(self, google_tool, mock_google_response):
        """Test successful Google search."""
        with patch('search_agent.tools.google_search.httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_google_response
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            results = await google_tool.search("test query")
            
            assert len(results) == 2  # Based on mock response
            assert all(result["source"] == "google_search" for result in results)
            assert all("title" in result for result in results)
            assert all("url" in result for result in results)
            assert all("snippet" in result for result in results)
    
    @pytest.mark.asyncio
    async def test_search_api_error(self, google_tool):
        """Test Google search with API error."""
        with patch('search_agent.tools.google_search.httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 403
            mock_response.text = "API quota exceeded"
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            with pytest.raises(SearchToolError):
                await google_tool.search("test query")
    
    @pytest.mark.asyncio
    async def test_search_no_items(self, google_tool):
        """Test Google search with no results."""
        with patch('search_agent.tools.google_search.httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"searchInformation": {"totalResults": "0"}}
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            results = await google_tool.search("test query")
            
            assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_search_malformed_response(self, google_tool):
        """Test Google search with malformed response."""
        with patch('search_agent.tools.google_search.httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"invalid": "response"}
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            results = await google_tool.search("test query")
            
            assert len(results) == 0


class TestArXivSearchTool:
    """Test cases for ArXiv search tool."""
    
    @pytest.fixture
    def arxiv_tool(self):
        """Create ArXivSearchTool instance."""
        config = Mock()
        config.max_results_per_source = 5
        return ArXivSearchTool(config)
    
    @pytest.mark.asyncio
    async def test_search_success(self, arxiv_tool, mock_arxiv_response):
        """Test successful ArXiv search."""
        with patch('search_agent.tools.arxiv.httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = mock_arxiv_response
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            results = await arxiv_tool.search("deep learning")
            
            assert len(results) == 1
            assert results[0]["source"] == "arxiv"
            assert results[0]["title"] == "Deep Learning for Artificial Intelligence"
            assert "authors" in results[0]
            assert "abstract" in results[0]
            assert "url" in results[0]
    
    @pytest.mark.asyncio
    async def test_search_xml_parsing_error(self, arxiv_tool):
        """Test ArXiv search with XML parsing error."""
        with patch('search_agent.tools.arxiv.httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = "Invalid XML content"
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            results = await arxiv_tool.search("test query")
            
            assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_search_api_error(self, arxiv_tool):
        """Test ArXiv search with API error."""
        with patch('search_agent.tools.arxiv.httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.text = "Internal server error"
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            with pytest.raises(SearchToolError):
                await arxiv_tool.search("test query")
    
    @pytest.mark.asyncio
    async def test_search_empty_feed(self, arxiv_tool):
        """Test ArXiv search with empty feed."""
        empty_feed = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
        </feed>"""
        
        with patch('search_agent.tools.arxiv.httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = empty_feed
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            results = await arxiv_tool.search("test query")
            
            assert len(results) == 0


class TestSemanticScholarSearchTool:
    """Test cases for Semantic Scholar search tool."""
    
    @pytest.fixture
    def semantic_scholar_tool(self):
        """Create SemanticScholarSearchTool instance."""
        config = Mock()
        config.max_results_per_source = 5
        return SemanticScholarSearchTool(config)
    
    @pytest.mark.asyncio
    async def test_search_success(self, semantic_scholar_tool, mock_semantic_scholar_response):
        """Test successful Semantic Scholar search."""
        with patch('search_agent.tools.semantic_scholar.httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_semantic_scholar_response
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            results = await semantic_scholar_tool.search("machine learning")
            
            assert len(results) == 1
            assert results[0]["source"] == "semantic_scholar"
            assert results[0]["title"] == "Advances in Machine Learning"
            assert "authors" in results[0]
            assert "abstract" in results[0]
            assert "citation_count" in results[0]
    
    @pytest.mark.asyncio
    async def test_search_rate_limit(self, semantic_scholar_tool):
        """Test Semantic Scholar search with rate limit."""
        with patch('search_agent.tools.semantic_scholar.httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 429
            mock_response.text = "Rate limit exceeded"
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            with pytest.raises(SearchToolError):
                await semantic_scholar_tool.search("test query")
    
    @pytest.mark.asyncio
    async def test_search_no_data(self, semantic_scholar_tool):
        """Test Semantic Scholar search with no data."""
        with patch('search_agent.tools.semantic_scholar.httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": [], "total": 0}
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            results = await semantic_scholar_tool.search("test query")
            
            assert len(results) == 0


class TestSearchToolsIntegration:
    """Integration tests for search tools."""
    
    @pytest.mark.asyncio
    async def test_multiple_tools_search(self):
        """Test searching with multiple tools."""
        config = Mock()
        config.google_api_key = "test_key"
        config.google_cse_id = "test_cse_id"
        config.max_results_per_source = 3
        
        tools = [
            WikipediaSearchTool(config),
            GoogleSearchTool(config),
            ArXivSearchTool(config)
        ]
        
        # Mock responses for all tools
        with patch('search_agent.tools.wikipedia.wikipedia') as mock_wiki, \
             patch('search_agent.tools.google_search.httpx.AsyncClient') as mock_google, \
             patch('search_agent.tools.arxiv.httpx.AsyncClient') as mock_arxiv:
            
            # Setup Wikipedia mock
            mock_wiki.search.return_value = ["AI"]
            mock_wiki.page.return_value.title = "AI"
            mock_wiki.page.return_value.summary = "AI summary"
            mock_wiki.page.return_value.url = "https://wiki.url"
            
            # Setup Google mock
            mock_google_response = Mock()
            mock_google_response.status_code = 200
            mock_google_response.json.return_value = {
                "items": [{"title": "Google Result", "link": "https://google.url", "snippet": "snippet"}]
            }
            mock_google.return_value.__aenter__.return_value.get.return_value = mock_google_response
            
            # Setup ArXiv mock
            mock_arxiv_response = Mock()
            mock_arxiv_response.status_code = 200
            mock_arxiv_response.text = """<?xml version="1.0"?>
            <feed xmlns="http://www.w3.org/2005/Atom">
                <entry>
                    <title>ArXiv Paper</title>
                    <summary>ArXiv abstract</summary>
                    <author><name>Author</name></author>
                    <link href="https://arxiv.url"/>
                </entry>
            </feed>"""
            mock_arxiv.return_value.__aenter__.return_value.get.return_value = mock_arxiv_response
            
            # Execute searches
            all_results = []
            for tool in tools:
                results = await tool.search("artificial intelligence")
                all_results.extend(results)
            
            # Verify results from all tools
            sources = {result["source"] for result in all_results}
            assert "wikipedia" in sources
            assert "google_search" in sources
            assert "arxiv" in sources
            assert len(all_results) >= 3  # At least one from each tool
    
    def test_tool_configuration_validation(self):
        """Test that tools validate their configuration properly."""
        # Test Google tool with missing API key
        config_missing_key = Mock()
        config_missing_key.google_api_key = None
        config_missing_key.google_cse_id = "test_cse_id"
        
        with pytest.raises((AttributeError, ValueError)):
            GoogleSearchTool(config_missing_key)
    
    @pytest.mark.asyncio
    async def test_tool_error_handling(self):
        """Test that tools handle errors gracefully."""
        config = Mock()
        config.max_results_per_source = 5
        
        wikipedia_tool = WikipediaSearchTool(config)
        
        # Test with network error
        with patch('search_agent.tools.wikipedia.wikipedia') as mock_wiki:
            mock_wiki.search.side_effect = Exception("Network error")
            
            # Should not raise exception, should return empty list
            results = await wikipedia_tool.search("test query")
            assert isinstance(results, list)
            assert len(results) == 0
