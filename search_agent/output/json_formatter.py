"""
JSON output formatting for search results.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)


class SearchResult(BaseModel):
    """Model for individual search results."""
    title: str
    url: Optional[str] = None
    snippet: Optional[str] = None
    source: str
    relevance_score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SynthesisResult(BaseModel):
    """Model for synthesis results."""
    summary: str
    key_findings: List[str] = Field(default_factory=list)
    academic_insights: Optional[str] = None
    general_information: Optional[str] = None
    background_context: Optional[str] = None
    confidence_level: str = "medium"
    limitations: Optional[str] = None
    suggested_followup: List[str] = Field(default_factory=list)


class SearchMetadata(BaseModel):
    """Model for search metadata."""
    execution_time_seconds: float
    total_results: int
    sources_searched: List[str]
    errors: List[str] = Field(default_factory=list)
    context_used: bool
    timestamp: float
    query_optimization_used: bool = True


class FormattedSearchOutput(BaseModel):
    """Complete formatted search output."""
    query: str
    synthesis: SynthesisResult
    search_results: Dict[str, List[SearchResult]]
    optimized_queries: Dict[str, Any]
    metadata: SearchMetadata


class JSONFormatter:
    """Formats search results into structured JSON output."""
    
    def __init__(self):
        """Initialize JSON formatter with configuration."""
        self.include_raw_results = True
        self.max_results_per_source = 10
        self.include_metadata = True
    
    def format_output(self, workflow_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format workflow output into structured JSON.
        
        Args:
            workflow_result: Raw output from search workflow
            
        Returns:
            Formatted JSON output
        """
        try:
            # Extract components
            query = workflow_result.get("query", "")
            synthesis = workflow_result.get("synthesis", {})
            raw_search_results = workflow_result.get("search_results", {})
            optimized_queries = workflow_result.get("optimized_queries", {})
            metadata = workflow_result.get("metadata", {})
            
            # Format synthesis
            formatted_synthesis = self._format_synthesis(synthesis)
            
            # Format search results
            formatted_results = self._format_search_results(raw_search_results)
            
            # Format metadata
            formatted_metadata = self._format_metadata(metadata)
            
            # Create formatted output
            formatted_output = FormattedSearchOutput(
                query=query,
                synthesis=formatted_synthesis,
                search_results=formatted_results,
                optimized_queries=optimized_queries,
                metadata=formatted_metadata
            )
            
            # Convert to dict and apply filtering
            output_dict = formatted_output.model_dump()
            
            # Apply configuration-based filtering
            if not self.include_raw_results:
                output_dict.pop("search_results", None)
            
            if not self.include_metadata:
                output_dict.pop("metadata", None)
            
            return output_dict
            
        except Exception as e:
            logger.error(f"Output formatting failed: {str(e)}")
            return self._create_error_output(workflow_result, str(e))
    
    def format_to_json_string(self, workflow_result: Dict[str, Any], indent: int = 2) -> str:
        """
        Format workflow output to JSON string.
        
        Args:
            workflow_result: Raw output from search workflow
            indent: JSON indentation level
            
        Returns:
            Formatted JSON string
        """
        try:
            formatted_output = self.format_output(workflow_result)
            return json.dumps(formatted_output, indent=indent, ensure_ascii=False)
        except Exception as e:
            logger.error(f"JSON string formatting failed: {str(e)}")
            error_output = {"error": f"Formatting failed: {str(e)}", "raw_data": workflow_result}
            return json.dumps(error_output, indent=indent, ensure_ascii=False)
    
    def save_to_file(self, workflow_result: Dict[str, Any], filepath: str) -> bool:
        """
        Save formatted output to JSON file.
        
        Args:
            workflow_result: Raw output from search workflow
            filepath: Path to save the JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            formatted_output = self.format_output(workflow_result)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(formatted_output, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Output saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save output to {filepath}: {str(e)}")
            return False
    
    def _format_synthesis(self, synthesis: Dict[str, Any]) -> SynthesisResult:
        """Format synthesis results."""
        try:
            return SynthesisResult(
                summary=synthesis.get("summary", ""),
                key_findings=synthesis.get("key_findings", []),
                academic_insights=synthesis.get("academic_insights"),
                general_information=synthesis.get("general_information"),
                background_context=synthesis.get("background_context"),
                confidence_level=synthesis.get("confidence_level", "medium"),
                limitations=synthesis.get("limitations"),
                suggested_followup=synthesis.get("suggested_followup", [])
            )
        except ValidationError as e:
            logger.warning(f"Synthesis validation failed: {str(e)}")
            return SynthesisResult(
                summary=synthesis.get("summary", "Synthesis formatting failed"),
                confidence_level="low"
            )
    
    def _format_search_results(self, raw_results: Dict[str, List[Dict]]) -> Dict[str, List[SearchResult]]:
        """Format search results from all sources."""
        formatted_results = {}
        
        for source, results in raw_results.items():
            if not isinstance(results, list):
                continue
            
            source_results = []
            for result in results[:self.max_results_per_source]:
                try:
                    formatted_result = SearchResult(
                        title=result.get("title", "No title"),
                        url=result.get("url"),
                        snippet=result.get("snippet", result.get("abstract", result.get("content", ""))),
                        source=source,
                        relevance_score=result.get("relevance_score"),
                        metadata=self._extract_result_metadata(result, source)
                    )
                    source_results.append(formatted_result)
                except ValidationError as e:
                    logger.warning(f"Result validation failed for {source}: {str(e)}")
                    continue
            
            formatted_results[source] = source_results
        
        return formatted_results
    
    def _format_metadata(self, metadata: Dict[str, Any]) -> SearchMetadata:
        """Format metadata."""
        try:
            return SearchMetadata(
                execution_time_seconds=metadata.get("execution_time_seconds", 0.0),
                total_results=metadata.get("total_results", 0),
                sources_searched=metadata.get("sources_searched", []),
                errors=metadata.get("errors", []),
                context_used=metadata.get("context_used", False),
                timestamp=metadata.get("timestamp", datetime.now().timestamp()),
                query_optimization_used=metadata.get("query_optimization_used", True)
            )
        except ValidationError as e:
            logger.warning(f"Metadata validation failed: {str(e)}")
            return SearchMetadata(
                execution_time_seconds=0.0,
                total_results=0,
                sources_searched=[],
                timestamp=datetime.now().timestamp()
            )
    
    def _extract_result_metadata(self, result: Dict[str, Any], source: str) -> Dict[str, Any]:
        """Extract relevant metadata from a search result."""
        metadata = {}
        
        # Common metadata fields
        if "published_date" in result:
            metadata["published_date"] = result["published_date"]
        
        if "authors" in result:
            metadata["authors"] = result["authors"]
        
        if "doi" in result:
            metadata["doi"] = result["doi"]
        
        if "venue" in result:
            metadata["venue"] = result["venue"]
        
        if "citations" in result:
            metadata["citations"] = result["citations"]
        
        if "categories" in result:
            metadata["categories"] = result["categories"]
        
        # Source-specific metadata
        if source == "arxiv":
            if "arxiv_id" in result:
                metadata["arxiv_id"] = result["arxiv_id"]
            if "primary_category" in result:
                metadata["primary_category"] = result["primary_category"]
        
        elif source == "semantic_scholar":
            if "paper_id" in result:
                metadata["paper_id"] = result["paper_id"]
            if "influential_citation_count" in result:
                metadata["influential_citation_count"] = result["influential_citation_count"]
        
        elif source in ["google", "brave"]:
            if "displayed_url" in result:
                metadata["displayed_url"] = result["displayed_url"]
            if "favicon" in result:
                metadata["favicon"] = result["favicon"]
        
        return metadata
    
    def _create_error_output(self, workflow_result: Dict[str, Any], error_message: str) -> Dict[str, Any]:
        """Create error output when formatting fails."""
        return {
            "query": workflow_result.get("query", "Unknown"),
            "error": error_message,
            "synthesis": {
                "summary": "Output formatting failed",
                "key_findings": [],
                "confidence_level": "low",
                "limitations": f"Formatting error: {error_message}"
            },
            "search_results": {},
            "metadata": {
                "execution_time_seconds": 0.0,
                "total_results": 0,
                "sources_searched": [],
                "errors": [error_message],
                "context_used": False,
                "timestamp": datetime.now().timestamp()
            }
        }
    
    def create_summary_output(self, workflow_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary-only output for quick consumption."""
        formatted_output = self.format_output(workflow_result)
        
        summary_output = {
            "query": formatted_output["query"],
            "summary": formatted_output["synthesis"]["summary"],
            "key_findings": formatted_output["synthesis"]["key_findings"],
            "confidence_level": formatted_output["synthesis"]["confidence_level"],
            "sources_count": {
                source: len(results) 
                for source, results in formatted_output["search_results"].items()
            },
            "execution_time": formatted_output["metadata"]["execution_time_seconds"]
        }
        
        return summary_output
