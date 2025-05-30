"""
Main Search Agent implementation that orchestrates the entire search process.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import json

from ..workflow.search_workflow import SearchWorkflow
from ..output.json_formatter import JSONFormatter
from ..output.report_generator import ReportGenerator
from ..utils.config import Config
from ..utils.cache import CacheManager
from ..utils.rate_limiter import APIRateLimitManager
from ..utils.data_validation import QueryValidator, DataCleaner
from ..exceptions.custom_exceptions import (
    SearchAgentError, ConfigurationError, WorkflowError
)

logger = logging.getLogger(__name__)


class SearchAgent:
    """
    Main Search Agent class that provides high-level interface for multi-source intelligent search.
    
    This class orchestrates the entire search process including:
    - Query validation and preprocessing
    - Multi-source search execution
    - Result synthesis and formatting
    - Caching and rate limiting
    """
    
    def __init__(self, config: Optional[Union[Dict[str, Any], Config]] = None):
        """
        Initialize the Search Agent.
        
        Args:
            config: Configuration dictionary or Config object
        """
        try:
            # Initialize configuration
            if isinstance(config, Config):
                self.config = config
            else:
                self.config = Config(config or {})
            
            # Initialize utilities
            self.cache_manager = CacheManager(self.config.get_all())
            self.rate_limit_manager = APIRateLimitManager(self.config.get_all())
            
            # Initialize output formatters
            self.json_formatter = JSONFormatter(self.config.get_all())
            self.report_generator = ReportGenerator(self.config.get_all())
            
            # Initialize search workflow
            self.workflow = SearchWorkflow(self.config)
            
            # Agent state
            self.is_initialized = True
            self.search_history = []
            
            logger.info("Search Agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Search Agent: {str(e)}")
            raise ConfigurationError(f"Search Agent initialization failed: {str(e)}")
    
    async def search(
        self,
        query: str,
        output_format: str = "json",
        save_to_file: Optional[str] = None,
        include_raw_results: bool = True,
        max_results_per_source: int = 10,
        use_cache: bool = True,
        timeout: float = 300.0
    ) -> Dict[str, Any]:
        """
        Perform a comprehensive multi-source search.
        
        Args:
            query: Search query
            output_format: Output format ("json", "markdown", "text", "summary")
            save_to_file: Optional file path to save results
            include_raw_results: Whether to include raw search results
            max_results_per_source: Maximum results per source
            use_cache: Whether to use caching
            timeout: Search timeout in seconds
            
        Returns:
            Formatted search results
        """
        if not self.is_initialized:
            raise SearchAgentError("Search Agent is not properly initialized")
        
        start_time = datetime.now()
        search_id = f"search_{int(start_time.timestamp())}"
        
        try:
            logger.info(f"Starting search [{search_id}]: {query}")
            
            # Validate query
            validation_result = QueryValidator.validate_query(query)
            if not validation_result["is_valid"]:
                raise SearchAgentError(f"Invalid query: {', '.join(validation_result['issues'])}")
            
            processed_query = validation_result["processed_query"]
            
            # Check cache if enabled
            if use_cache:
                cached_result = await self._get_cached_result(processed_query)
                if cached_result:
                    logger.info(f"Returning cached result for [{search_id}]")
                    return await self._format_output(
                        cached_result, output_format, save_to_file, include_raw_results
                    )
            
            # Configure search options
            search_config = {
                "max_results_per_source": max_results_per_source,
                "use_cache": use_cache,
                "timeout": timeout
            }
            
            # Execute search workflow
            workflow_result = await asyncio.wait_for(
                self.workflow.run_search(processed_query, search_config),
                timeout=timeout
            )
            
            # Post-process results
            workflow_result = await self._post_process_results(workflow_result)
            
            # Cache results if enabled
            if use_cache:
                await self._cache_result(processed_query, workflow_result)
            
            # Add to search history
            self._add_to_history(search_id, query, processed_query, workflow_result)
            
            # Format output
            formatted_result = await self._format_output(
                workflow_result, output_format, save_to_file, include_raw_results
            )
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            logger.info(f"Search completed [{search_id}] in {execution_time:.2f}s")
            
            return formatted_result
            
        except asyncio.TimeoutError:
            logger.error(f"Search timeout [{search_id}] after {timeout}s")
            raise SearchAgentError(f"Search timed out after {timeout} seconds")
        
        except Exception as e:
            logger.error(f"Search failed [{search_id}]: {str(e)}")
            raise SearchAgentError(f"Search execution failed: {str(e)}")
    
    async def quick_search(self, query: str) -> str:
        """
        Perform a quick search and return a summary.
        
        Args:
            query: Search query
            
        Returns:
            Quick summary string
        """
        try:
            result = await self.search(
                query=query,
                output_format="summary",
                include_raw_results=False,
                max_results_per_source=5,
                timeout=60.0
            )
            
            return self.report_generator.generate_quick_summary(result)
            
        except Exception as e:
            logger.error(f"Quick search failed: {str(e)}")
            return f"Quick search failed: {str(e)}"
    
    async def batch_search(
        self,
        queries: List[str],
        output_format: str = "json",
        save_to_directory: Optional[str] = None,
        max_concurrent: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Perform batch searches with concurrency control.
        
        Args:
            queries: List of search queries
            output_format: Output format for each search
            save_to_directory: Directory to save individual results
            max_concurrent: Maximum concurrent searches
            
        Returns:
            List of search results
        """
        if not queries:
            return []
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def _search_with_semaphore(query: str, index: int) -> Dict[str, Any]:
            async with semaphore:
                try:
                    save_path = None
                    if save_to_directory:
                        import os
                        os.makedirs(save_to_directory, exist_ok=True)
                        safe_query = "".join(c for c in query[:50] if c.isalnum() or c in (' ', '-', '_')).strip()
                        save_path = os.path.join(save_to_directory, f"search_{index:03d}_{safe_query}.json")
                    
                    result = await self.search(
                        query=query,
                        output_format=output_format,
                        save_to_file=save_path,
                        max_results_per_source=5
                    )
                    
                    return {"query": query, "index": index, "result": result, "status": "success"}
                    
                except Exception as e:
                    logger.error(f"Batch search failed for query {index}: {str(e)}")
                    return {"query": query, "index": index, "error": str(e), "status": "failed"}
        
        logger.info(f"Starting batch search for {len(queries)} queries with max {max_concurrent} concurrent")
        
        tasks = [_search_with_semaphore(query, i) for i, query in enumerate(queries)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful_results = []
        failed_count = 0
        
        for result in results:
            if isinstance(result, Exception):
                failed_count += 1
                logger.error(f"Batch search task failed: {str(result)}")
            elif result.get("status") == "success":
                successful_results.append(result["result"])
            else:
                failed_count += 1
        
        logger.info(f"Batch search completed: {len(successful_results)} successful, {failed_count} failed")
        return successful_results
    
    async def get_search_suggestions(self, partial_query: str) -> List[str]:
        """
        Get search suggestions based on partial query.
        
        Args:
            partial_query: Partial search query
            
        Returns:
            List of suggested queries
        """
        try:
            # Validate partial query
            validation_result = QueryValidator.validate_query(partial_query)
            
            suggestions = validation_result.get("suggestions", [])
            
            # Add query-specific suggestions
            if len(partial_query.split()) < 3:
                suggestions.append("Consider adding more specific terms")
            
            # Add domain-specific suggestions based on detected indicators
            metadata = validation_result.get("metadata", {})
            if metadata.get("academic_indicators"):
                suggestions.extend([
                    "Try academic databases for research papers",
                    "Consider adding year restrictions for recent research"
                ])
            
            return suggestions[:10]  # Limit to top 10 suggestions
            
        except Exception as e:
            logger.error(f"Failed to get search suggestions: {str(e)}")
            return []
    
    def get_search_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent search history.
        
        Args:
            limit: Maximum number of history items to return
            
        Returns:
            List of recent searches
        """
        return self.search_history[-limit:] if self.search_history else []
    
    def clear_search_history(self) -> None:
        """Clear search history."""
        self.search_history.clear()
        logger.info("Search history cleared")
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return await self.cache_manager.get_cache_stats()
    
    async def clear_cache(self, cache_type: Optional[str] = None) -> bool:
        """
        Clear cache.
        
        Args:
            cache_type: Type of cache to clear (None for all)
            
        Returns:
            True if successful
        """
        return await self.cache_manager.clear_cache(cache_type)
    
    def get_rate_limit_stats(self) -> Dict[str, Any]:
        """Get rate limiting statistics."""
        return self.rate_limit_manager.get_all_stats()
    
    def reset_rate_limiters(self) -> None:
        """Reset all rate limiters."""
        self.rate_limit_manager.reset_all_limiters()
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check of the search agent.
        
        Returns:
            Health status information
        """
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {},
            "issues": []
        }
        
        try:
            # Check configuration
            try:
                self.config.validate()
                health_status["components"]["config"] = "healthy"
            except Exception as e:
                health_status["components"]["config"] = "unhealthy"
                health_status["issues"].append(f"Configuration issue: {str(e)}")
            
            # Check cache
            try:
                cache_stats = await self.cache_manager.get_cache_stats()
                health_status["components"]["cache"] = "healthy"
                health_status["cache_stats"] = cache_stats
            except Exception as e:
                health_status["components"]["cache"] = "unhealthy"
                health_status["issues"].append(f"Cache issue: {str(e)}")
            
            # Check rate limiters
            try:
                rate_stats = self.rate_limit_manager.get_all_stats()
                health_status["components"]["rate_limiters"] = "healthy"
                health_status["rate_limit_stats"] = rate_stats
            except Exception as e:
                health_status["components"]["rate_limiters"] = "unhealthy"
                health_status["issues"].append(f"Rate limiter issue: {str(e)}")
            
            # Overall status
            if health_status["issues"]:
                health_status["status"] = "degraded" if len(health_status["issues"]) < 3 else "unhealthy"
            
        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["issues"].append(f"Health check failed: {str(e)}")
        
        return health_status
    
    async def _get_cached_result(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached search result."""
        try:
            # Create a cache key for the complete search
            cache_key = f"complete_search_{hash(query)}"
            
            # This would need to be implemented in the cache manager
            # For now, return None to skip caching of complete searches
            return None
            
        except Exception as e:
            logger.warning(f"Failed to retrieve cached result: {str(e)}")
            return None
    
    async def _cache_result(self, query: str, result: Dict[str, Any]) -> None:
        """Cache search result."""
        try:
            # For now, skip caching of complete search results
            # This could be implemented later if needed
            pass
            
        except Exception as e:
            logger.warning(f"Failed to cache result: {str(e)}")
    
    async def _post_process_results(self, workflow_result: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process workflow results."""
        try:
            # Clean and deduplicate results
            search_results = workflow_result.get("search_results", {})
            
            for source, results in search_results.items():
                if results:
                    # Remove duplicates
                    cleaned_results = DataCleaner.remove_duplicates(results)
                    
                    # Filter by relevance
                    query = workflow_result.get("query", "")
                    if query:
                        cleaned_results = DataCleaner.filter_by_relevance(cleaned_results, query)
                    
                    search_results[source] = cleaned_results
            
            workflow_result["search_results"] = search_results
            
            return workflow_result
            
        except Exception as e:
            logger.warning(f"Post-processing failed: {str(e)}")
            return workflow_result
    
    async def _format_output(
        self,
        workflow_result: Dict[str, Any],
        output_format: str,
        save_to_file: Optional[str],
        include_raw_results: bool
    ) -> Dict[str, Any]:
        """Format output according to specified format."""
        try:
            # Configure formatter
            formatter_config = {
                "include_raw_results": include_raw_results,
                "max_results_per_source": 10
            }
            
            # Format based on requested format
            if output_format == "json":
                self.json_formatter.config.update(formatter_config)
                formatted_result = self.json_formatter.format_output(workflow_result)
                
                if save_to_file:
                    self.json_formatter.save_to_file(workflow_result, save_to_file)
                
                return formatted_result
            
            elif output_format == "summary":
                formatted_result = self.json_formatter.create_summary_output(workflow_result)
                
                if save_to_file:
                    with open(save_to_file, 'w', encoding='utf-8') as f:
                        json.dump(formatted_result, f, indent=2, ensure_ascii=False)
                
                return formatted_result
            
            elif output_format in ["markdown", "text"]:
                self.report_generator.config.update(formatter_config)
                formatted_result = self.json_formatter.format_output(workflow_result)
                
                if save_to_file:
                    self.report_generator.save_report(formatted_result, save_to_file, output_format)
                
                return formatted_result
            
            else:
                raise SearchAgentError(f"Unsupported output format: {output_format}")
                
        except Exception as e:
            logger.error(f"Output formatting failed: {str(e)}")
            raise SearchAgentError(f"Output formatting failed: {str(e)}")
    
    def _add_to_history(
        self,
        search_id: str,
        original_query: str,
        processed_query: str,
        result: Dict[str, Any]
    ) -> None:
        """Add search to history."""
        try:
            history_entry = {
                "search_id": search_id,
                "timestamp": datetime.now().isoformat(),
                "original_query": original_query,
                "processed_query": processed_query,
                "total_results": result.get("metadata", {}).get("total_results", 0),
                "sources_searched": result.get("metadata", {}).get("sources_searched", []),
                "execution_time": result.get("metadata", {}).get("execution_time_seconds", 0),
                "has_errors": bool(result.get("metadata", {}).get("errors", []))
            }
            
            self.search_history.append(history_entry)
            
            # Keep only last 100 searches
            if len(self.search_history) > 100:
                self.search_history = self.search_history[-100:]
                
        except Exception as e:
            logger.warning(f"Failed to add search to history: {str(e)}")
    
    def __repr__(self) -> str:
        """String representation of SearchAgent."""
        return f"SearchAgent(initialized={self.is_initialized}, history_count={len(self.search_history)})"
