"""
LangGraph workflow implementation for orchestrating the multi-source search process.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, TypedDict
from langgraph.graph import Graph, START, END
from langgraph.checkpoint.memory import MemorySaver

from ..core.state import AgentState, StateManager, ExecutionStatus
from ..llm.gemini import GeminiLLM
from ..tools.wikipedia import WikipediaSearchTool
from ..tools.google_search import GoogleSearchTool
from ..tools.brave_search import BraveSearchTool
from ..tools.semantic_scholar import SemanticScholarTool
from ..tools.arxiv import ArXivSearchTool
from ..tools.core_api import CORESearchTool
from ..utils.config import get_config
from ..exceptions.custom_exceptions import WorkflowError, SearchToolError

logger = logging.getLogger(__name__)


class SearchWorkflow:
    """LangGraph workflow for orchestrating multi-source search operations."""
    
    def __init__(self):
        """Initialize the search workflow with configuration."""
        self.config = get_config()
        
        # Initialize LLM
        self.llm = GeminiLLM()
        
        # Initialize search tools
        self.tools = {
            'wikipedia': WikipediaSearchTool(),
            'google': GoogleSearchTool(),
            'brave': BraveSearchTool(),
            'semantic_scholar': SemanticScholarTool(),
            'arxiv': ArXivSearchTool(),
            'core': CORESearchTool()
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
            initial_state = StateManager.create_initial_state(query)
            
            # Run the workflow
            result = await self.graph.ainvoke(initial_state)
            
            # Return the final output from execution metadata
            return result.get("execution_metadata", {}).get("final_output", {})
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            raise WorkflowError(f"Search workflow failed: {str(e)}")
    
    async def _initialize_search(self, state: AgentState) -> AgentState:
        """Initialize the search process."""
        logger.info(f"Starting search for query: {state['original_query']}")
        
        # Update execution metadata
        state = StateManager.update_execution_metadata(state, "initialize")
        state["execution_metadata"]["tools_available"] = list(self.tools.keys())
        
        # Extract key concepts from the query
        try:
            key_concepts = await self.llm.extract_key_concepts(state["original_query"])
            state["execution_metadata"]["key_concepts"] = key_concepts
        except Exception as e:
            logger.warning(f"Failed to extract key concepts: {str(e)}")
            state["errors"]["key_concepts"] = f"Key concept extraction failed: {str(e)}"
        
        return state
    
    async def _get_wikipedia_context(self, state: AgentState) -> AgentState:
        """Get background context from Wikipedia."""
        # Update execution metadata
        state = StateManager.update_execution_metadata(state, "get_context")
        
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
                
                state["wikipedia_summary"] = " ".join(context_parts)
                state["raw_search_results"]["wikipedia"] = wikipedia_results
                
                logger.info(f"Retrieved context from {len(wikipedia_results)} Wikipedia articles")
            else:
                logger.warning("No Wikipedia context found")
                state["wikipedia_summary"] = ""
                
        except Exception as e:
            logger.error(f"Wikipedia context retrieval failed: {str(e)}")
            state["errors"]["wikipedia"] = f"Wikipedia context failed: {str(e)}"
            state["wikipedia_error"] = str(e)
            state["wikipedia_summary"] = ""
        
        return state
    
    async def _optimize_search_queries(self, state: AgentState) -> AgentState:
        """Optimize search queries for different sources."""
        # Update execution metadata
        state = StateManager.update_execution_metadata(state, "optimize_query")
        
        try:
            logger.info("Optimizing search queries")
            
            # Use LLM to optimize queries
            optimization_result = await self.llm.optimize_query(
                state["original_query"],
                state["wikipedia_summary"]
            )
            
            # Store refined queries for different sources
            if isinstance(optimization_result, dict):
                for source_name in ["arxiv", "semantic_scholar", "google", "brave"]:
                    query_key = f"{source_name}_query"
                    if query_key in optimization_result:
                        state["refined_queries"][source_name] = optimization_result[query_key]
                    else:
                        state["refined_queries"][source_name] = state["original_query"]
            else:
                # Fallback: use original query for all sources
                for source_name in ["arxiv", "semantic_scholar", "google", "brave"]:
                    state["refined_queries"][source_name] = state["original_query"]
            
            logger.info("Query optimization completed")
            
        except Exception as e:
            logger.error(f"Query optimization failed: {str(e)}")
            state["errors"]["query_optimization"] = f"Query optimization failed: {str(e)}"
            
            # Fallback: use original query for all sources
            for source_name in ["arxiv", "semantic_scholar", "google", "brave"]:
                state["refined_queries"][source_name] = state["original_query"]
        
        return state
    
    async def _search_academic_sources(self, state: AgentState) -> AgentState:
        """Search academic sources in parallel."""
        # Update execution metadata
        state = StateManager.update_execution_metadata(state, "search_academic")
        
        logger.info("Searching academic sources")
        
        # Get refined queries or fallback to original
        arxiv_query = state["refined_queries"].get("arxiv", state["original_query"])
        semantic_query = state["refined_queries"].get("semantic_scholar", state["original_query"])
        
        # Define academic search tasks
        academic_tasks = []
        
        # Semantic Scholar
        if "semantic_scholar" in self.tools:
            academic_tasks.append(
                self._safe_search("semantic_scholar", semantic_query, limit=5)
            )
        
        # ArXiv
        if "arxiv" in self.tools:
            academic_tasks.append(
                self._safe_search("arxiv", arxiv_query, limit=5)
            )
        
        # CORE API
        if "core" in self.tools:
            academic_tasks.append(
                self._safe_search("core", semantic_query, limit=5)
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
                        state["errors"][tool_name] = f"{tool_name} search failed: {str(result)}"
                        state["raw_search_results"][tool_name] = []
                    else:
                        state["raw_search_results"][tool_name] = result
        
        logger.info(f"Academic search completed for {len(academic_tasks)} sources")
        return state
    
    async def _search_web_sources(self, state: AgentState) -> AgentState:
        """Search web sources in parallel."""
        # Update execution metadata
        state = StateManager.update_execution_metadata(state, "search_web")
        
        logger.info("Searching web sources")
        
        # Get refined queries or fallback to original
        google_query = state["refined_queries"].get("google", state["original_query"])
        brave_query = state["refined_queries"].get("brave", state["original_query"])
        
        # Define web search tasks
        web_tasks = []
        
        # Google Search
        if "google" in self.tools:
            web_tasks.append(
                self._safe_search("google", google_query, limit=8)
            )
        
        # Brave Search
        if "brave" in self.tools:
            web_tasks.append(
                self._safe_search("brave", brave_query, limit=8)
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
                        state["errors"][tool_name] = f"{tool_name} search failed: {str(result)}"
                        state["raw_search_results"][tool_name] = []
                    else:
                        state["raw_search_results"][tool_name] = result
        
        logger.info(f"Web search completed for {len(web_tasks)} sources")
        return state
    
    async def _synthesize_search_results(self, state: AgentState) -> AgentState:
        """Synthesize results from all sources."""
        # Update execution metadata
        state = StateManager.update_execution_metadata(state, "synthesize_results")
        
        try:
            logger.info("Synthesizing search results")
            
            # Use LLM to synthesize results
            synthesis = await self.llm.synthesize_results(
                state["original_query"],
                state["raw_search_results"]
            )
            
            # Store synthesis result in execution metadata
            state["execution_metadata"]["synthesis"] = synthesis
            logger.info("Result synthesis completed")
            
        except Exception as e:
            logger.error(f"Result synthesis failed: {str(e)}")
            state["errors"]["synthesis"] = f"Result synthesis failed: {str(e)}"
            
            # Fallback synthesis
            state["execution_metadata"]["synthesis"] = {
                "summary": "Search completed but synthesis failed",
                "key_findings": [],
                "confidence_level": "low",
                "limitations": "Synthesis processing encountered errors"
            }
        
        return state
    
    async def _format_final_output(self, state: AgentState) -> AgentState:
        """Format the final output."""
        # Update execution metadata and finalize state
        state = StateManager.update_execution_metadata(state, "format_output")
        state = StateManager.finalize_state(state)
        
        try:
            # Count total results
            total_results = sum(len(results) for results in state["raw_search_results"].values())
            
            # Format final output
            final_output = {
                "query": state["original_query"],
                "synthesis": state["execution_metadata"].get("synthesis", {}),
                "search_results": state["raw_search_results"],
                "refined_queries": state["refined_queries"],
                "wikipedia_context": state["wikipedia_summary"],
                "metadata": {
                    "execution_time_seconds": state["total_execution_time"],
                    "total_results": total_results,
                    "sources_searched": list(state["raw_search_results"].keys()),
                    "successful_sources": StateManager.get_successful_sources(state),
                    "failed_sources": StateManager.get_failed_sources(state),
                    "errors": state["errors"],
                    "context_used": bool(state["wikipedia_summary"]),
                    "timestamp": state["execution_timestamp"],
                    "final_status": state["final_status"]
                }
            }
            
            # Store final output in execution metadata
            state["execution_metadata"]["final_output"] = final_output
            logger.info(f"Search completed in {state['total_execution_time']:.2f}s with {total_results} total results")
            
        except Exception as e:
            logger.error(f"Output formatting failed: {str(e)}")
            state["errors"]["output_formatting"] = f"Output formatting failed: {str(e)}"
            
            # Minimal fallback output
            state["execution_metadata"]["final_output"] = {
                "query": state["original_query"],
                "synthesis": {},
                "search_results": state["raw_search_results"],
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
    
    def _should_run_parallel_searches(self, state: AgentState) -> str:
        """Determine which searches to run based on the query and configuration."""
        # This could be made more sophisticated based on query analysis
        # For now, always run both academic and web searches
        return "both"
