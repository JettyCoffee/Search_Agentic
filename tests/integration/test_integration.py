"""
Integration tests for the search agent system.
"""

import pytest
import asyncio
import os
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

from search_agent import SearchAgent
from search_agent.utils.config import Config
from search_agent.exceptions.custom_exceptions import SearchAgentError


class TestSearchAgentIntegration:
    """Integration tests for the complete search agent system."""
    
    @pytest.fixture
    def test_config(self):
        """Create test configuration."""
        return {
            "google_api_key": "test_google_key",
            "google_cse_id": "test_cse_id",
            "brave_api_key": "test_brave_key",
            "gemini_api_key": "test_gemini_key",
            "core_api_key": "test_core_key",
            "log_level": "INFO",
            "cache_type": "memory",
            "rate_limit_enabled": False,
            "max_results_per_source": 3
        }
    
    @pytest.fixture
    def search_agent(self, test_config):
        """Create SearchAgent instance for testing."""
        return SearchAgent(config=test_config)
    
    @pytest.mark.asyncio
    async def test_end_to_end_search_flow(self, search_agent):
        """Test complete end-to-end search flow."""
        # Mock all external APIs
        with patch('search_agent.tools.wikipedia.wikipedia') as mock_wiki, \
             patch('search_agent.llm.gemini.genai') as mock_gemini, \
             patch('httpx.AsyncClient') as mock_client:
            
            # Setup Wikipedia mock
            mock_wiki.search.return_value = ["Artificial Intelligence"]
            mock_wiki.page.return_value.title = "Artificial Intelligence"
            mock_wiki.page.return_value.summary = "AI is the simulation of human intelligence processes by machines..."
            mock_wiki.page.return_value.url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
            
            # Setup Gemini mock
            mock_gemini.configure.return_value = None
            mock_model = Mock()
            mock_response = Mock()
            mock_response.text = '{"optimized_query": "artificial intelligence machine learning", "key_concepts": ["AI", "ML"], "search_strategy": "comprehensive"}'
            mock_model.generate_content.return_value = mock_response
            mock_gemini.GenerativeModel.return_value = mock_model
            
            # Setup HTTP client mocks for other APIs
            mock_http_response = Mock()
            mock_http_response.status_code = 200
            mock_http_response.json.return_value = {"items": []}
            mock_http_response.text = "<feed></feed>"
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_http_response
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_http_response
            
            # Execute search
            result = await search_agent.search("artificial intelligence")
            
            # Verify result structure
            assert isinstance(result, dict)
            assert "query" in result
            assert "results" in result
            assert "metadata" in result
            assert result["query"] == "artificial intelligence"
    
    @pytest.mark.asyncio
    async def test_batch_search_integration(self, search_agent):
        """Test batch search functionality."""
        queries = [
            "artificial intelligence",
            "machine learning",
            "deep learning"
        ]
        
        # Mock all external dependencies
        with patch('search_agent.tools.wikipedia.wikipedia') as mock_wiki, \
             patch('search_agent.llm.gemini.genai') as mock_gemini, \
             patch('httpx.AsyncClient') as mock_client:
            
            # Setup basic mocks
            mock_wiki.search.return_value = ["Test Topic"]
            mock_wiki.page.return_value.title = "Test Topic"
            mock_wiki.page.return_value.summary = "Test summary"
            mock_wiki.page.return_value.url = "https://test.url"
            
            mock_gemini.configure.return_value = None
            mock_model = Mock()
            mock_response = Mock()
            mock_response.text = '{"optimized_query": "test query", "key_concepts": ["test"], "search_strategy": "basic"}'
            mock_model.generate_content.return_value = mock_response
            mock_gemini.GenerativeModel.return_value = mock_model
            
            mock_http_response = Mock()
            mock_http_response.status_code = 200
            mock_http_response.json.return_value = {"items": []}
            mock_http_response.text = "<feed></feed>"
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_http_response
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_http_response
            
            # Execute batch search
            results = await search_agent.batch_search(queries, max_concurrent=2)
            
            # Verify results
            assert len(results) == 3
            assert all(result is not None for result in results)
            assert all("query" in result for result in results if result)
    
    @pytest.mark.asyncio
    async def test_search_with_different_sources(self, search_agent):
        """Test search with specific source selection."""
        sources = ["wikipedia", "arxiv"]
        
        with patch('search_agent.tools.wikipedia.wikipedia') as mock_wiki, \
             patch('search_agent.llm.gemini.genai') as mock_gemini, \
             patch('httpx.AsyncClient') as mock_client:
            
            # Setup mocks
            mock_wiki.search.return_value = ["Test"]
            mock_wiki.page.return_value.title = "Test"
            mock_wiki.page.return_value.summary = "Test summary"
            mock_wiki.page.return_value.url = "https://test.url"
            
            mock_gemini.configure.return_value = None
            mock_model = Mock()
            mock_response = Mock()
            mock_response.text = '{"optimized_query": "test", "key_concepts": ["test"], "search_strategy": "basic"}'
            mock_model.generate_content.return_value = mock_response
            mock_gemini.GenerativeModel.return_value = mock_model
            
            mock_http_response = Mock()
            mock_http_response.status_code = 200
            mock_http_response.text = """<?xml version="1.0"?>
            <feed xmlns="http://www.w3.org/2005/Atom">
                <entry>
                    <title>Test Paper</title>
                    <summary>Test abstract</summary>
                    <author><name>Test Author</name></author>
                    <link href="https://arxiv.org/abs/test"/>
                </entry>
            </feed>"""
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_http_response
            
            # Execute search with specific sources
            result = await search_agent.search("test query", sources=sources)
            
            # Verify that only specified sources were used
            assert "results" in result
            used_sources = set()
            for source_results in result["results"].values():
                for item in source_results:
                    used_sources.add(item["source"])
            
            # Should only contain wikipedia and arxiv results
            assert used_sources.issubset({"wikipedia", "arxiv"})
    
    @pytest.mark.asyncio
    async def test_caching_integration(self, search_agent):
        """Test caching functionality integration."""
        query = "test caching query"
        
        with patch('search_agent.tools.wikipedia.wikipedia') as mock_wiki, \
             patch('search_agent.llm.gemini.genai') as mock_gemini:
            
            # Setup mocks
            mock_wiki.search.return_value = ["Test"]
            mock_wiki.page.return_value.title = "Test"
            mock_wiki.page.return_value.summary = "Test summary"
            mock_wiki.page.return_value.url = "https://test.url"
            
            mock_gemini.configure.return_value = None
            mock_model = Mock()
            mock_response = Mock()
            mock_response.text = '{"optimized_query": "test", "key_concepts": ["test"], "search_strategy": "basic"}'
            mock_model.generate_content.return_value = mock_response
            mock_gemini.GenerativeModel.return_value = mock_model
            
            # First search - should call APIs
            result1 = await search_agent.search(query)
            
            # Second search - should use cache (if caching is enabled)
            result2 = await search_agent.search(query)
            
            # Both results should be valid
            assert result1 is not None
            assert result2 is not None
            assert result1["query"] == result2["query"]
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self, search_agent):
        """Test error handling across the system."""
        # Test with API failures
        with patch('search_agent.tools.wikipedia.wikipedia') as mock_wiki, \
             patch('search_agent.llm.gemini.genai') as mock_gemini:
            
            # Make Wikipedia fail
            mock_wiki.search.side_effect = Exception("Wikipedia API error")
            
            # Make Gemini work to ensure partial functionality
            mock_gemini.configure.return_value = None
            mock_model = Mock()
            mock_response = Mock()
            mock_response.text = '{"optimized_query": "test", "key_concepts": ["test"], "search_strategy": "basic"}'
            mock_model.generate_content.return_value = mock_response
            mock_gemini.GenerativeModel.return_value = mock_model
            
            # Search should still work (graceful degradation)
            result = await search_agent.search("test query")
            
            # Should get a result despite Wikipedia failure
            assert result is not None
            assert "query" in result
    
    @pytest.mark.asyncio
    async def test_performance_monitoring_integration(self, search_agent):
        """Test performance monitoring integration."""
        with patch('search_agent.tools.wikipedia.wikipedia') as mock_wiki, \
             patch('search_agent.llm.gemini.genai') as mock_gemini:
            
            # Setup mocks
            mock_wiki.search.return_value = ["Test"]
            mock_wiki.page.return_value.title = "Test"
            mock_wiki.page.return_value.summary = "Test summary"
            mock_wiki.page.return_value.url = "https://test.url"
            
            mock_gemini.configure.return_value = None
            mock_model = Mock()
            mock_response = Mock()
            mock_response.text = '{"optimized_query": "test", "key_concepts": ["test"], "search_strategy": "basic"}'
            mock_model.generate_content.return_value = mock_response
            mock_gemini.GenerativeModel.return_value = mock_model
            
            # Execute search
            result = await search_agent.search("performance test")
            
            # Check that performance metrics are recorded
            assert "metadata" in result
            assert "search_time" in result["metadata"]
            assert isinstance(result["metadata"]["search_time"], (int, float))
            assert result["metadata"]["search_time"] >= 0
    
    def test_health_check_integration(self, search_agent):
        """Test health check functionality."""
        # Health check should work without external API calls
        health = asyncio.run(search_agent.health_check())
        
        assert isinstance(health, dict)
        assert "status" in health
        assert "tools" in health
        assert "system" in health
    
    def test_configuration_integration(self, test_config):
        """Test configuration integration across components."""
        # Create agent with custom config
        agent = SearchAgent(config=test_config)
        
        # Verify configuration is propagated
        assert agent.config.log_level == "INFO"
        assert agent.config.cache_type == "memory"
        assert agent.config.max_results_per_source == 3


class TestFileSystemIntegration:
    """Test file system integration (caching, logging, etc.)."""
    
    def test_file_cache_integration(self):
        """Test file-based caching integration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "cache_type": "file",
                "cache_dir": temp_dir,
                "google_api_key": "test_key",
                "gemini_api_key": "test_key"
            }
            
            agent = SearchAgent(config=config)
            
            # Verify cache directory setup
            cache_path = Path(temp_dir)
            assert cache_path.exists()
    
    def test_logging_integration(self):
        """Test logging integration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create agent with logging to temp directory
            with patch('search_agent.utils.logging.Path') as mock_path:
                mock_path.return_value.mkdir.return_value = None
                
                config = {
                    "log_level": "DEBUG",
                    "google_api_key": "test_key",
                    "gemini_api_key": "test_key"
                }
                
                agent = SearchAgent(config=config)
                
                # Verify logging is configured
                assert agent.logger is not None
    
    def test_output_file_integration(self):
        """Test output file generation integration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "google_api_key": "test_key",
                "gemini_api_key": "test_key"
            }
            
            agent = SearchAgent(config=config)
            
            # Test result saving (would need to implement save functionality)
            sample_result = {
                "query": "test query",
                "results": {"wikipedia": []},
                "metadata": {"search_time": 1.0, "timestamp": datetime.now().isoformat()}
            }
            
            # This would test saving functionality if implemented
            output_file = Path(temp_dir) / "test_results.json"
            # agent.save_results(sample_result, output_file)
            # assert output_file.exists()


class TestConcurrencyIntegration:
    """Test concurrent operations and thread safety."""
    
    @pytest.mark.asyncio
    async def test_concurrent_searches(self):
        """Test multiple concurrent searches."""
        config = {
            "google_api_key": "test_key",
            "gemini_api_key": "test_key",
            "rate_limit_enabled": False  # Disable for testing
        }
        
        agent = SearchAgent(config=config)
        
        # Mock all external dependencies
        with patch('search_agent.tools.wikipedia.wikipedia') as mock_wiki, \
             patch('search_agent.llm.gemini.genai') as mock_gemini, \
             patch('httpx.AsyncClient') as mock_client:
            
            # Setup mocks
            mock_wiki.search.return_value = ["Test"]
            mock_wiki.page.return_value.title = "Test"
            mock_wiki.page.return_value.summary = "Test summary"
            mock_wiki.page.return_value.url = "https://test.url"
            
            mock_gemini.configure.return_value = None
            mock_model = Mock()
            mock_response = Mock()
            mock_response.text = '{"optimized_query": "test", "key_concepts": ["test"], "search_strategy": "basic"}'
            mock_model.generate_content.return_value = mock_response
            mock_gemini.GenerativeModel.return_value = mock_model
            
            mock_http_response = Mock()
            mock_http_response.status_code = 200
            mock_http_response.json.return_value = {"items": []}
            mock_http_response.text = "<feed></feed>"
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_http_response
            
            # Create multiple concurrent search tasks
            tasks = [
                agent.search(f"concurrent query {i}")
                for i in range(5)
            ]
            
            # Execute concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Verify all searches completed
            assert len(results) == 5
            assert all(not isinstance(result, Exception) for result in results)
            assert all("query" in result for result in results)
    
    @pytest.mark.asyncio
    async def test_rate_limiting_integration(self):
        """Test rate limiting integration."""
        config = {
            "google_api_key": "test_key",
            "gemini_api_key": "test_key",
            "rate_limit_enabled": True,
            "rate_limit_requests_per_minute": 2
        }
        
        agent = SearchAgent(config=config)
        
        # This would test rate limiting if properly implemented
        # For now, just verify the agent can be created with rate limiting enabled
        assert agent.config.rate_limit_enabled is True
