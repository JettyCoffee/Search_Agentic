"""
Unit tests for the SearchAgent core functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
from typing import Dict, Any

from search_agent.core.agent import SearchAgent
from search_agent.core.state import SearchState
from search_agent.exceptions.custom_exceptions import (
    SearchAgentError,
    ConfigurationError,
    SearchToolError
)


class TestSearchAgent:
    """Test cases for the SearchAgent class."""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Mock all dependencies for SearchAgent."""
        with patch('search_agent.core.agent.Config') as mock_config, \
             patch('search_agent.core.agent.SearchWorkflow') as mock_workflow, \
             patch('search_agent.core.agent.get_logger') as mock_logger, \
             patch('search_agent.core.agent.get_performance_monitor') as mock_perf, \
             patch('search_agent.core.agent.get_search_metrics') as mock_metrics:
            
            # Configure mocks
            mock_config.return_value.validate.return_value = True
            mock_workflow.return_value.execute.return_value = {
                "query": "test query",
                "results": {"web_search": []},
                "metadata": {"search_time": 1.0, "sources_used": ["test"]}
            }
            
            yield {
                "config": mock_config,
                "workflow": mock_workflow,
                "logger": mock_logger,
                "performance": mock_perf,
                "metrics": mock_metrics
            }
    
    def test_initialization(self, mock_dependencies):
        """Test SearchAgent initialization."""
        agent = SearchAgent()
        
        assert agent is not None
        assert hasattr(agent, 'config')
        assert hasattr(agent, 'workflow')
        assert hasattr(agent, 'search_history')
        assert len(agent.search_history) == 0
    
    def test_initialization_with_custom_config(self, mock_dependencies):
        """Test SearchAgent initialization with custom config."""
        custom_config = {"log_level": "DEBUG"}
        agent = SearchAgent(config=custom_config)
        
        assert agent is not None
        mock_dependencies["config"].assert_called_with(custom_config)
    
    @pytest.mark.asyncio
    async def test_search_success(self, mock_dependencies):
        """Test successful search operation."""
        agent = SearchAgent()
        
        result = await agent.search("test query")
        
        assert result is not None
        assert "query" in result
        assert "results" in result
        assert "metadata" in result
        assert len(agent.search_history) == 1
    
    @pytest.mark.asyncio
    async def test_search_with_sources(self, mock_dependencies):
        """Test search with specific sources."""
        agent = SearchAgent()
        sources = ["wikipedia", "google_search"]
        
        result = await agent.search("test query", sources=sources)
        
        assert result is not None
        mock_dependencies["workflow"].return_value.execute.assert_called_once()
        
        # Check that sources were passed to workflow
        call_args = mock_dependencies["workflow"].return_value.execute.call_args
        state = call_args[0][0]  # First argument should be the state
        assert "sources" in state or hasattr(state, 'sources')
    
    @pytest.mark.asyncio
    async def test_search_with_options(self, mock_dependencies):
        """Test search with additional options."""
        agent = SearchAgent()
        options = {
            "max_results": 20,
            "include_abstracts": True,
            "language": "en"
        }
        
        result = await agent.search("test query", **options)
        
        assert result is not None
        mock_dependencies["workflow"].return_value.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_error_handling(self, mock_dependencies):
        """Test search error handling."""
        agent = SearchAgent()
        
        # Mock workflow to raise an exception
        mock_dependencies["workflow"].return_value.execute.side_effect = SearchToolError("Test error")
        
        with pytest.raises(SearchAgentError):
            await agent.search("test query")
    
    @pytest.mark.asyncio
    async def test_batch_search_success(self, mock_dependencies):
        """Test successful batch search operation."""
        agent = SearchAgent()
        queries = ["query 1", "query 2", "query 3"]
        
        results = await agent.batch_search(queries)
        
        assert len(results) == 3
        assert all("query" in result for result in results)
        assert len(agent.search_history) == 3
    
    @pytest.mark.asyncio
    async def test_batch_search_with_concurrency_limit(self, mock_dependencies):
        """Test batch search with concurrency limit."""
        agent = SearchAgent()
        queries = ["query 1", "query 2", "query 3", "query 4"]
        
        results = await agent.batch_search(queries, max_concurrent=2)
        
        assert len(results) == 4
        assert len(agent.search_history) == 4
    
    @pytest.mark.asyncio
    async def test_batch_search_partial_failure(self, mock_dependencies):
        """Test batch search with some failures."""
        agent = SearchAgent()
        queries = ["good query", "bad query"]
        
        # Mock workflow to fail on second query
        def side_effect(state):
            if "bad query" in state.get("query", ""):
                raise SearchToolError("Simulated error")
            return {
                "query": state.get("query"),
                "results": {"web_search": []},
                "metadata": {"search_time": 1.0, "sources_used": ["test"]}
            }
        
        mock_dependencies["workflow"].return_value.execute.side_effect = side_effect
        
        results = await agent.batch_search(queries)
        
        assert len(results) == 2
        assert results[0] is not None  # First query succeeded
        assert results[1] is None      # Second query failed
    
    def test_get_search_history(self, mock_dependencies):
        """Test getting search history."""
        agent = SearchAgent()
        
        # Add some mock history
        agent.search_history = [
            {"query": "query 1", "timestamp": datetime.now()},
            {"query": "query 2", "timestamp": datetime.now()}
        ]
        
        history = agent.get_search_history()
        assert len(history) == 2
        assert all("query" in item for item in history)
    
    def test_get_search_history_with_limit(self, mock_dependencies):
        """Test getting search history with limit."""
        agent = SearchAgent()
        
        # Add mock history
        agent.search_history = [
            {"query": f"query {i}", "timestamp": datetime.now()}
            for i in range(10)
        ]
        
        history = agent.get_search_history(limit=5)
        assert len(history) == 5
    
    def test_clear_search_history(self, mock_dependencies):
        """Test clearing search history."""
        agent = SearchAgent()
        
        # Add some mock history
        agent.search_history = [{"query": "test", "timestamp": datetime.now()}]
        
        agent.clear_search_history()
        assert len(agent.search_history) == 0
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, mock_dependencies):
        """Test successful health check."""
        agent = SearchAgent()
        
        # Mock all tools as healthy
        with patch.object(agent, '_check_tool_health') as mock_check:
            mock_check.return_value = True
            
            health = await agent.health_check()
            
            assert health["status"] == "healthy"
            assert "tools" in health
            assert "system" in health
    
    @pytest.mark.asyncio
    async def test_health_check_with_unhealthy_tools(self, mock_dependencies):
        """Test health check with some unhealthy tools."""
        agent = SearchAgent()
        
        # Mock some tools as unhealthy
        def mock_health_check(tool_name):
            return tool_name != "problematic_tool"
        
        with patch.object(agent, '_check_tool_health', side_effect=mock_health_check):
            health = await agent.health_check()
            
            assert "tools" in health
            assert any(not status for status in health["tools"].values())
    
    def test_get_stats(self, mock_dependencies):
        """Test getting agent statistics."""
        agent = SearchAgent()
        
        # Add some mock history
        agent.search_history = [
            {"query": "query 1", "timestamp": datetime.now()},
            {"query": "query 2", "timestamp": datetime.now()}
        ]
        
        stats = agent.get_stats()
        
        assert "total_searches" in stats
        assert "search_history_size" in stats
        assert stats["total_searches"] == 2
        assert stats["search_history_size"] == 2


class TestSearchAgentValidation:
    """Test cases for SearchAgent input validation."""
    
    @pytest.fixture
    def agent(self):
        """Create a SearchAgent instance for testing."""
        with patch('search_agent.core.agent.Config'), \
             patch('search_agent.core.agent.SearchWorkflow'), \
             patch('search_agent.core.agent.get_logger'), \
             patch('search_agent.core.agent.get_performance_monitor'), \
             patch('search_agent.core.agent.get_search_metrics'):
            return SearchAgent()
    
    @pytest.mark.asyncio
    async def test_search_empty_query(self, agent):
        """Test search with empty query."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            await agent.search("")
    
    @pytest.mark.asyncio
    async def test_search_none_query(self, agent):
        """Test search with None query."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            await agent.search(None)
    
    @pytest.mark.asyncio
    async def test_search_whitespace_query(self, agent):
        """Test search with whitespace-only query."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            await agent.search("   ")
    
    @pytest.mark.asyncio
    async def test_search_invalid_sources(self, agent):
        """Test search with invalid sources."""
        with pytest.raises(ValueError, match="Invalid source"):
            await agent.search("test query", sources=["invalid_source"])
    
    @pytest.mark.asyncio
    async def test_batch_search_empty_queries(self, agent):
        """Test batch search with empty queries list."""
        with pytest.raises(ValueError, match="Queries list cannot be empty"):
            await agent.batch_search([])
    
    @pytest.mark.asyncio
    async def test_batch_search_invalid_concurrent_limit(self, agent):
        """Test batch search with invalid concurrency limit."""
        with pytest.raises(ValueError, match="max_concurrent must be positive"):
            await agent.batch_search(["query1", "query2"], max_concurrent=0)


class TestSearchAgentIntegration:
    """Integration tests for SearchAgent with real components."""
    
    @pytest.mark.asyncio
    async def test_search_with_real_state(self):
        """Test search with real SearchState object."""
        with patch('search_agent.core.agent.Config') as mock_config, \
             patch('search_agent.core.agent.SearchWorkflow') as mock_workflow:
            
            mock_config.return_value.validate.return_value = True
            
            # Configure workflow to work with real state
            def execute_workflow(state):
                assert isinstance(state, dict)
                assert "query" in state
                return {
                    "query": state["query"],
                    "results": {"web_search": []},
                    "metadata": {"search_time": 1.0, "sources_used": ["test"]}
                }
            
            mock_workflow.return_value.execute.side_effect = execute_workflow
            
            agent = SearchAgent()
            result = await agent.search("integration test query")
            
            assert result["query"] == "integration test query"
    
    def test_performance_monitoring_integration(self):
        """Test integration with performance monitoring."""
        with patch('search_agent.core.agent.Config'), \
             patch('search_agent.core.agent.SearchWorkflow'), \
             patch('search_agent.core.agent.get_performance_monitor') as mock_perf:
            
            mock_monitor = Mock()
            mock_perf.return_value = mock_monitor
            
            agent = SearchAgent()
            
            # Verify performance monitor is used
            assert agent.performance_monitor == mock_monitor
            mock_perf.assert_called_once()
    
    def test_metrics_integration(self):
        """Test integration with search metrics."""
        with patch('search_agent.core.agent.Config'), \
             patch('search_agent.core.agent.SearchWorkflow'), \
             patch('search_agent.core.agent.get_search_metrics') as mock_metrics:
            
            mock_search_metrics = Mock()
            mock_metrics.return_value = mock_search_metrics
            
            agent = SearchAgent()
            
            # Verify search metrics is used
            assert agent.search_metrics == mock_search_metrics
            mock_metrics.assert_called_once()
