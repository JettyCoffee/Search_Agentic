"""
LangGraph workflow implementation for orchestrating the multi-source search process.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, TypedDict
from langgraph.graph import Graph, START, END
from langgraph.checkpoint.memory import MemorySaver

from ..core.state import SearchState
from ..llm.gemini import GeminiLLM
from ..tools.wikipedia import WikipediaSearchTool
from ..tools.google_search import GoogleSearchTool
from ..tools.brave_search import BraveSearchTool
from ..tools.semantic_scholar import SemanticScholarTool
from ..tools.arxiv import ArXivTool
from ..tools.core_api import CoreAPITool
from ..utils.config import Config
from ..exceptions.custom_exceptions import WorkflowError, SearchToolError

logger = logging.getLogger(__name__)


class SearchWorkflow:
    """LangGraph workflow for orchestrating multi-source search operations."""
    
    def __init__(self, config: Config):
        """Initialize the search workflow with configuration."""
        self.config = config
        
        # Initialize LLM
        self.llm = GeminiLLM(config)
        
        # Initialize search tools
        self.tools = {
            'wikipedia': WikipediaSearchTool(config),
            'google': GoogleSearchTool(config),
            'brave': BraveSearchTool(config),
            'semantic_scholar': SemanticScholarTool(config),
            'arxiv': ArXivTool(config),
            'core': CoreAPITool(config)
        }
        
        # Create the workflow graph
        self.graph = self._build_workflow_graph()
        
        # Set up checkpointing for state persistence
        self.checkpointer = MemorySaver()
        
        logger.info("Search workflow initialized successfully")
    
    def _build_workflow_graph(self) -> Graph:
        """Build the LangGraph workflow."""
        # Create workflow builder
        workflow = Graph()
        
        # Add workflow nodes
        workflow.add_node("initialize", self._initialize_search)
        workflow.add_node("get_context", self._get_wikipedia_context)
        workflow.add_node("optimize_query", self._optimize_search_queries)
        workflow.add_node("search_academic", self._search_academic_sources)
        workflow.add_node("search_web", self._search_web_sources)
        workflow.add_node("synthesize_results", self._synthesize_search_results)
        workflow.add_node("format_output", self._format_final_output)
        
        # Define workflow edges
        workflow.add_edge(START, "initialize")
        workflow.add_edge("initialize", "get_context")
        workflow.add_edge("get_context", "optimize_query")
        workflow.add_edge("optimize_query", "search_academic")
        workflow.add_edge("optimize_query", "search_web")
        workflow.add_edge("search_academic", "synthesize_results")
        workflow.add_edge("search_web", "synthesize_results")
        workflow.add_edge("synthesize_results", "format_output")
        workflow.add_edge("format_output", END)
        
        # Add conditional routing for parallel searches
        workflow.add_conditional_edges(
            "optimize_query",
            self._should_run_parallel_searches,
            {
                "academic": "search_academic",
                "web": "search_web",
                "both": ["search_academic", "search_web"]
            }
        )
        
        return workflow.compile(checkpointer=self.checkpointer)
    
    async def run_search(self, query: str, config_overrides: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Run the complete search workflow.
        
        Args:
            query: The search query
            config_overrides: Optional configuration overrides
            
        Returns:
            Complete search results with synthesis
        """
        try:
            # Initialize state
            initial_state = SearchState(
                original_query=query,
                current_step="initialize",
                search_results={},
                optimized_queries={},
                context="",
                synthesis={},
                final_output={},
                errors=[],
                metadata={
                    "start_time": asyncio.get_event_loop().time(),
                    "config_overrides": config_overrides or {}
                }
            )
            
            # Run the workflow
            result = await self.graph.ainvoke(initial_state)
            
            # Return the final output
            return result.get("final_output", {})
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            raise WorkflowError(f"Search workflow failed: {str(e)}")
    
    async def _initialize_search(self, state: SearchState) -> SearchState:
        """Initialize the search process."""
        logger.info(f"Starting search for query: {state['original_query']}")
        
        state["current_step"] = "initialize"
        state["metadata"]["tools_available"] = list(self.tools.keys())
        
        # Extract key concepts from the query
        try:
            key_concepts = await self.llm.extract_key_concepts(state["original_query"])
            state["metadata"]["key_concepts"] = key_concepts
        except Exception as e:
            logger.warning(f"Failed to extract key concepts: {str(e)}")
            state["errors"].append(f"Key concept extraction failed: {str(e)}")
        
        return state
    
    async def _get_wikipedia_context(self, state: SearchState) -> SearchState:
        """Get background context from Wikipedia."""
        state["current_step"] = "get_context"
        
        try:
            logger.info("Fetching Wikipedia context")
            
            # Search Wikipedia for background context
            wikipedia_results = await self.tools['wikipedia'].search(
                state["original_query"], 
                limit=3
            )
            
            if wikipedia_results:
                # Combine Wikipedia content for context
                context_parts = []
                for result in wikipedia_results:
                    if result.get('content'):
                        context_parts.append(result['content'][:500])  # Limit length
                
                state["context"] = " ".join(context_parts)
                state["search_results"]["wikipedia"] = wikipedia_results
                
                logger.info(f"Retrieved context from {len(wikipedia_results)} Wikipedia articles")
            else:
                logger.warning("No Wikipedia context found")
                state["context"] = ""
                
        except Exception as e:
            logger.error(f"Wikipedia context retrieval failed: {str(e)}")
            state["errors"].append(f"Wikipedia context failed: {str(e)}")
            state["context"] = ""
        
        return state
    
    async def _optimize_search_queries(self, state: SearchState) -> SearchState:
        """Optimize search queries for different sources."""
        state["current_step"] = "optimize_query"
        
        try:
            logger.info("Optimizing search queries")
            
            # Use LLM to optimize queries
            optimization_result = await self.llm.optimize_query(
                state["original_query"],
                state["context"]
            )
            
            state["optimized_queries"] = optimization_result
            logger.info("Query optimization completed")
            
        except Exception as e:
            logger.error(f"Query optimization failed: {str(e)}")
            state["errors"].append(f"Query optimization failed: {str(e)}")
            
            # Fallback: use original query
            state["optimized_queries"] = {
                "academic_query": state["original_query"],
                "web_query": state["original_query"],
                "wikipedia_query": state["original_query"],
                "key_concepts": [],
                "alternative_terms": [],
                "broader_terms": [],
                "narrower_terms": []
            }
        
        return state
    
    async def _search_academic_sources(self, state: SearchState) -> SearchState:
        """Search academic sources in parallel."""
        state["current_step"] = "search_academic"
        
        logger.info("Searching academic sources")
        
        # Get optimized academic query
        academic_query = state["optimized_queries"].get("academic_query", state["original_query"])
        
        # Define academic search tasks
        academic_tasks = []
        
        # Semantic Scholar
        if "semantic_scholar" in self.tools:
            academic_tasks.append(
                self._safe_search("semantic_scholar", academic_query, limit=5)
            )
        
        # ArXiv
        if "arxiv" in self.tools:
            academic_tasks.append(
                self._safe_search("arxiv", academic_query, limit=5)
            )
        
        # CORE API
        if "core" in self.tools:
            academic_tasks.append(
                self._safe_search("core", academic_query, limit=5)
            )
        
        # Execute academic searches in parallel
        if academic_tasks:
            results = await asyncio.gather(*academic_tasks, return_exceptions=True)
            
            # Process results
            tool_names = ["semantic_scholar", "arxiv", "core"]
            for i, result in enumerate(results):
                if i < len(tool_names):
                    tool_name = tool_names[i]
                    if isinstance(result, Exception):
                        logger.error(f"{tool_name} search failed: {str(result)}")
                        state["errors"].append(f"{tool_name} search failed: {str(result)}")
                        state["search_results"][tool_name] = []
                    else:
                        state["search_results"][tool_name] = result
        
        logger.info(f"Academic search completed for {len(academic_tasks)} sources")
        return state
    
    async def _search_web_sources(self, state: SearchState) -> SearchState:
        """Search web sources in parallel."""
        state["current_step"] = "search_web"
        
        logger.info("Searching web sources")
        
        # Get optimized web query
        web_query = state["optimized_queries"].get("web_query", state["original_query"])
        
        # Define web search tasks
        web_tasks = []
        
        # Google Search
        if "google" in self.tools:
            web_tasks.append(
                self._safe_search("google", web_query, limit=8)
            )
        
        # Brave Search
        if "brave" in self.tools:
            web_tasks.append(
                self._safe_search("brave", web_query, limit=8)
            )
        
        # Execute web searches in parallel
        if web_tasks:
            results = await asyncio.gather(*web_tasks, return_exceptions=True)
            
            # Process results
            tool_names = ["google", "brave"]
            for i, result in enumerate(results):
                if i < len(tool_names):
                    tool_name = tool_names[i]
                    if isinstance(result, Exception):
                        logger.error(f"{tool_name} search failed: {str(result)}")
                        state["errors"].append(f"{tool_name} search failed: {str(result)}")
                        state["search_results"][tool_name] = []
                    else:
                        state["search_results"][tool_name] = result
        
        logger.info(f"Web search completed for {len(web_tasks)} sources")
        return state
    
    async def _synthesize_search_results(self, state: SearchState) -> SearchState:
        """Synthesize results from all sources."""
        state["current_step"] = "synthesize_results"
        
        try:
            logger.info("Synthesizing search results")
            
            # Use LLM to synthesize results
            synthesis = await self.llm.synthesize_results(
                state["original_query"],
                state["search_results"]
            )
            
            state["synthesis"] = synthesis
            logger.info("Result synthesis completed")
            
        except Exception as e:
            logger.error(f"Result synthesis failed: {str(e)}")
            state["errors"].append(f"Result synthesis failed: {str(e)}")
            
            # Fallback synthesis
            state["synthesis"] = {
                "summary": "Search completed but synthesis failed",
                "key_findings": [],
                "confidence_level": "low",
                "limitations": "Synthesis processing encountered errors"
            }
        
        return state
    
    async def _format_final_output(self, state: SearchState) -> SearchState:
        """Format the final output."""
        state["current_step"] = "format_output"
        
        try:
            # Calculate execution time
            end_time = asyncio.get_event_loop().time()
            start_time = state["metadata"].get("start_time", end_time)
            execution_time = end_time - start_time
            
            # Count total results
            total_results = sum(len(results) for results in state["search_results"].values())
            
            # Format final output
            final_output = {
                "query": state["original_query"],
                "synthesis": state["synthesis"],
                "search_results": state["search_results"],
                "optimized_queries": state["optimized_queries"],
                "metadata": {
                    "execution_time_seconds": round(execution_time, 2),
                    "total_results": total_results,
                    "sources_searched": list(state["search_results"].keys()),
                    "errors": state["errors"],
                    "context_used": bool(state["context"]),
                    "timestamp": end_time
                }
            }
            
            state["final_output"] = final_output
            logger.info(f"Search completed in {execution_time:.2f}s with {total_results} total results")
            
        except Exception as e:
            logger.error(f"Output formatting failed: {str(e)}")
            state["errors"].append(f"Output formatting failed: {str(e)}")
            
            # Minimal fallback output
            state["final_output"] = {
                "query": state["original_query"],
                "synthesis": state["synthesis"],
                "search_results": state["search_results"],
                "metadata": {"errors": state["errors"]}
            }
        
        return state
    
    async def _safe_search(self, tool_name: str, query: str, **kwargs) -> List[Dict]:
        """Safely execute a search with error handling."""
        try:
            if tool_name in self.tools:
                return await self.tools[tool_name].search(query, **kwargs)
            else:
                logger.warning(f"Tool {tool_name} not available")
                return []
        except Exception as e:
            logger.error(f"Search failed for {tool_name}: {str(e)}")
            raise SearchToolError(f"{tool_name} search failed: {str(e)}")
    
    def _should_run_parallel_searches(self, state: SearchState) -> str:
        """Determine which searches to run based on the query and configuration."""
        # This could be made more sophisticated based on query analysis
        # For now, always run both academic and web searches
        return "both"
