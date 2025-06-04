"""
Main Search Agent implementation that orchestrates the entire search process.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import json
import os

from ..workflow.search_workflow import SearchWorkflow
from ..output.json_formatter import JSONFormatter
from ..output.report_generator import ReportGenerator
from ..utils.config import get_config
from ..utils.cache import CacheManager
from ..utils.rate_limiter import APIRateLimitManager
from ..utils.data_validation import QueryValidator, DataCleaner
from ..exceptions.custom_exceptions import (
    SearchAgentError, ConfigurationError, WorkflowError
)

from langchain_community.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage

from ..tools.google_search import GoogleSearchTool
from ..tools.brave_search import BraveSearchTool
from ..tools.wikipedia import WikipediaTool

logger = logging.getLogger(__name__)


class MultiSourceSearchAgent:
    """多源搜索代理"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        self._setup_proxy()
        self._check_environment()
        self._initialize_llm()
        self._initialize_tools()
    
    def _setup_logging(self):
        """配置日志系统"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _setup_proxy(self):
        """配置代理"""
        # Windows主机IP（WSL默认网关）通常是172.x.x.x
        host_ip = "172.21.48.1"  # 这里需要根据实际情况修改
        proxy_port = "7890"  # Clash默认端口
        
        # 设置HTTP和HTTPS代理
        os.environ["HTTP_PROXY"] = f"http://{host_ip}:{proxy_port}"
        os.environ["HTTPS_PROXY"] = f"http://{host_ip}:{proxy_port}"
        
        self.logger.info(f"代理已配置: http://{host_ip}:{proxy_port}")
    
    def _check_environment(self):
        """检查环境变量"""
        required_vars = {
            "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
            "GOOGLE_CSE_ID": os.getenv("GOOGLE_CSE_ID"),
            "BRAVE_API_KEY": os.getenv("BRAVE_API_KEY"),
            "DEFAULT_LLM": os.getenv("DEFAULT_LLM", "gemini")
        }
        
        missing_vars = [key for key, value in required_vars.items() if not value]
        if missing_vars:
            self.logger.error(f"缺少必要的环境变量: {', '.join(missing_vars)}")
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        self.logger.info("环境变量检查通过")
        self.logger.debug(f"当前LLM设置: {required_vars['DEFAULT_LLM']}")
    
    def _initialize_llm(self):
        """初始化语言模型"""
        default_llm = os.getenv("DEFAULT_LLM", "gemini")
        self.logger.info(f"正在初始化LLM: {default_llm}")
        
        try:
            if default_llm == "gemini":
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-pro",
                    google_api_key=os.getenv("GOOGLE_API_KEY"),
                    temperature=0.7
                )
            elif default_llm == "claude":
                from langchain_anthropic import ChatAnthropic
                self.llm = ChatAnthropic(
                    model=os.getenv("CLAUDE_MODEL", "claude-3-sonnet-20240229"),
                    anthropic_api_key=os.getenv("CLAUDE_API_KEY"),
                    temperature=0.7
                )
            else:
                raise ValueError(f"Unsupported LLM: {default_llm}")
            
            self.logger.info("LLM初始化成功")
            
        except Exception as e:
            self.logger.error(f"LLM初始化失败: {str(e)}")
            raise
    
    def _initialize_tools(self):
        """初始化搜索工具"""
        self.logger.info("正在初始化搜索工具...")
        try:
            self.tools = {}
            
            # 初始化Google搜索
            try:
                self.tools["google"] = GoogleSearchTool()
                self.logger.info("Google搜索工具初始化成功")
            except Exception as e:
                self.logger.error(f"Google搜索工具初始化失败: {str(e)}")
            
            # 初始化Brave搜索
            try:
                self.tools["brave"] = BraveSearchTool()
                self.logger.info("Brave搜索工具初始化成功")
            except Exception as e:
                self.logger.error(f"Brave搜索工具初始化失败: {str(e)}")
            
            # 初始化Wikipedia
            try:
                self.tools["wikipedia"] = WikipediaTool()
                self.logger.info("Wikipedia工具初始化成功")
            except Exception as e:
                self.logger.error(f"Wikipedia工具初始化失败: {str(e)}")
            
            if not self.tools:
                raise ValueError("No search tools were successfully initialized")
                
        except Exception as e:
            self.logger.error(f"搜索工具初始化失败: {str(e)}")
            raise
    
    def search(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        执行多源搜索
        
        Args:
            query: 搜索查询
            **kwargs: 额外的搜索参数
        
        Returns:
            Dict[str, Any]: 搜索结果
        """
        self.logger.info(f"开始搜索: {query}")
        
        try:
            # 1. 使用LLM分析查询
            self.logger.info("正在分析查询...")
            analyzed_query = self._analyze_query(query)
            self.logger.info(f"优化后的查询: {analyzed_query}")
            
            # 2. 获取Wikipedia背景信息
            self.logger.info("正在获取Wikipedia上下文...")
            wiki_context = self.tools["wikipedia"].search(query)
            self.logger.info("已获取Wikipedia上下文")
            
            # 3. 执行多源搜索
            search_results = {}
            for source, tool in self.tools.items():
                try:
                    self.logger.info(f"正在从 {source} 搜索...")
                    results = tool.search(analyzed_query)
                    search_results[source] = {
                        "status": "success",
                        "results": results
                    }
                    self.logger.info(f"{source} 搜索完成，获取到 {len(results)} 条结果")
                except Exception as e:
                    self.logger.error(f"{source} 搜索失败: {str(e)}")
                    search_results[source] = {
                        "status": "error",
                        "error": str(e)
                    }
            
            # 4. 整理返回结果
            result = {
                "query_info": {
                    "original_query": query,
                    "analyzed_query": analyzed_query,
                    "timestamp": datetime.now().isoformat(),
                    "status": "success"
                },
                "context": {
                    "wikipedia": wiki_context
                },
                "sources": search_results
            }
            
            self.logger.info("搜索完成")
            return result
            
        except Exception as e:
            self.logger.error(f"搜索过程出错: {str(e)}")
            return {
                "query_info": {
                    "original_query": query,
                    "timestamp": datetime.now().isoformat(),
                    "status": "error",
                    "error": str(e)
                }
            }
    
    def _analyze_query(self, query: str) -> str:
        """使用LLM分析和优化查询"""
        try:
            self.logger.info("正在使用LLM分析查询...")
            response = self.llm.invoke(
                [HumanMessage(content=f"请分析并优化以下搜索查询，使其更适合搜索引擎：{query}")]
            )
            self.logger.info("查询分析完成")
            return response.content
        except Exception as e:
            self.logger.warning(f"查询分析失败，使用原始查询: {str(e)}")
            return query

class SearchAgent:
    """
    Main Search Agent class that provides high-level interface for multi-source intelligent search.
    
    This class orchestrates the entire search process including:
    - Query validation and preprocessing
    - Multi-source search execution
    - Result synthesis and formatting
    - Caching and rate limiting
    """
    
    def __init__(self):
        """
        Initialize the Search Agent.
        """
        try:
            # Initialize configuration
            self.config = get_config()
            
            # Initialize utilities
            self.cache_manager = CacheManager()
            self.rate_limit_manager = APIRateLimitManager()
            
            # Initialize output formatters
            self.json_formatter = JSONFormatter()
            self.report_generator = ReportGenerator()
            
            # Initialize search workflow
            self.workflow = SearchWorkflow()
            
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
