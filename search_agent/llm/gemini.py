"""
Google Gemini API integration for query optimization and response processing.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from tenacity import retry, stop_after_attempt, wait_exponential

from ..exceptions.custom_exceptions import LLMConfigError, LLMAPIError, QueryOptimizationError
from ..utils.config import Config

logger = logging.getLogger(__name__)


class GeminiLLM:
    """Google Gemini API integration for query processing and optimization."""
    
    def __init__(self, config: Config):
        """Initialize Gemini LLM with configuration."""
        self.config = config
        
        # Initialize Gemini API
        api_key = config.get_api_key("GEMINI_API_KEY")
        if not api_key:
            raise LLMConfigError("Gemini API key not found in configuration")
        
        genai.configure(api_key=api_key)
        
        # Configure model
        self.model_name = config.get("gemini_model", "gemini-1.5-flash")
        self.generation_config = {
            "temperature": config.get("gemini_temperature", 0.3),
            "top_p": config.get("gemini_top_p", 0.95),
            "top_k": config.get("gemini_top_k", 40),
            "max_output_tokens": config.get("gemini_max_tokens", 2048),
        }
        
        # Safety settings
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }
        
        # Initialize model
        try:
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
        except Exception as e:
            raise LLMConfigError(f"Failed to initialize Gemini model: {str(e)}")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _make_api_call(self, prompt: str) -> str:
        """Make an API call to Gemini with retry logic."""
        try:
            # Run the blocking API call in an executor
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: self.model.generate_content(prompt)
            )
            
            if not response.text:
                raise LLMAPIError("Empty response from Gemini API")
            
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Gemini API call failed: {str(e)}")
            raise LLMAPIError(f"Gemini API error: {str(e)}")
    
    async def optimize_query(self, query: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Optimize a search query for better results across multiple sources.
        
        Args:
            query: Original search query
            context: Optional context from Wikipedia or previous searches
            
        Returns:
            Dictionary containing optimized queries for different sources
        """
        try:
            # Build context-aware prompt
            prompt = self._build_query_optimization_prompt(query, context)
            
            # Get optimized queries
            response = await self._make_api_call(prompt)
            
            # Parse the JSON response
            try:
                result = json.loads(response)
                return self._validate_query_optimization(result)
            except json.JSONDecodeError:
                # Fallback: extract JSON from markdown code blocks
                return self._extract_json_from_response(response)
                
        except Exception as e:
            logger.error(f"Query optimization failed: {str(e)}")
            raise QueryOptimizationError(f"Failed to optimize query: {str(e)}")
    
    async def synthesize_results(self, query: str, search_results: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """
        Synthesize search results from multiple sources into a coherent response.
        
        Args:
            query: Original search query
            search_results: Results from different search tools
            
        Returns:
            Synthesized response with key findings and sources
        """
        try:
            # Build synthesis prompt
            prompt = self._build_synthesis_prompt(query, search_results)
            
            # Get synthesis
            response = await self._make_api_call(prompt)
            
            # Parse the JSON response
            try:
                result = json.loads(response)
                return self._validate_synthesis_result(result)
            except json.JSONDecodeError:
                return self._extract_json_from_response(response)
                
        except Exception as e:
            logger.error(f"Result synthesis failed: {str(e)}")
            raise LLMAPIError(f"Failed to synthesize results: {str(e)}")
    
    async def extract_key_concepts(self, text: str) -> List[str]:
        """
        Extract key concepts and entities from text for enhanced searching.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of key concepts and entities
        """
        try:
            prompt = f"""
            Extract the key concepts, entities, and important terms from the following text.
            Focus on named entities, technical terms, topics, and searchable concepts.
            Return as a JSON array of strings.
            
            Text: {text}
            
            Return only the JSON array, no additional text.
            """
            
            response = await self._make_api_call(prompt)
            
            try:
                concepts = json.loads(response)
                return concepts if isinstance(concepts, list) else []
            except json.JSONDecodeError:
                # Fallback: try to extract from response
                return self._extract_list_from_response(response)
                
        except Exception as e:
            logger.error(f"Concept extraction failed: {str(e)}")
            return []
    
    def _build_query_optimization_prompt(self, query: str, context: Optional[str] = None) -> str:
        """Build prompt for query optimization."""
        context_section = ""
        if context:
            context_section = f"""
            Additional Context: {context[:1000]}
            """
        
        return f"""
        You are a search query optimization expert. Your task is to transform a user query into optimized search queries for different types of sources.
        
        Original Query: {query}
        {context_section}
        
        Create optimized search queries for the following sources:
        1. Academic papers (Semantic Scholar, ArXiv, CORE) - focus on technical terms, research keywords
        2. General web search (Google, Brave) - broader terms, related concepts
        3. Wikipedia - encyclopedic terms, background concepts
        
        Also provide:
        - Key concepts and entities to focus on
        - Alternative phrasings and synonyms
        - Broader and narrower search terms
        
        Return your response as a JSON object with this structure:
        {{
            "academic_query": "optimized query for academic sources",
            "web_query": "optimized query for web search",
            "wikipedia_query": "optimized query for Wikipedia",
            "key_concepts": ["concept1", "concept2", "concept3"],
            "alternative_terms": ["term1", "term2", "term3"],
            "broader_terms": ["broad1", "broad2"],
            "narrower_terms": ["narrow1", "narrow2"]
        }}
        
        Return only the JSON object, no additional text.
        """
    
    def _build_synthesis_prompt(self, query: str, search_results: Dict[str, List[Dict]]) -> str:
        """Build prompt for result synthesis."""
        # Prepare search results summary
        results_summary = []
        for source, results in search_results.items():
            if results:
                source_summary = f"\n{source.upper()} RESULTS ({len(results)} items):"
                for i, result in enumerate(results[:3], 1):  # Top 3 results per source
                    title = result.get('title', 'No title')
                    snippet = result.get('snippet', result.get('abstract', 'No description'))[:200]
                    source_summary += f"\n{i}. {title}\n   {snippet}..."
                results_summary.append(source_summary)
        
        results_text = "\n".join(results_summary)
        
        return f"""
        You are an expert research analyst. Synthesize the following search results to provide a comprehensive answer to the user's query.
        
        User Query: {query}
        
        Search Results:
        {results_text[:4000]}  # Limit to prevent token overflow
        
        Provide a synthesis in JSON format with this structure:
        {{
            "summary": "A comprehensive summary answering the user's query",
            "key_findings": [
                "Finding 1 with source information",
                "Finding 2 with source information",
                "Finding 3 with source information"
            ],
            "academic_insights": "Insights from academic sources",
            "general_information": "Information from web sources", 
            "background_context": "Background information from Wikipedia",
            "confidence_level": "high|medium|low",
            "limitations": "Any limitations or gaps in the available information",
            "suggested_followup": ["Follow-up question 1", "Follow-up question 2"]
        }}
        
        Return only the JSON object, no additional text.
        """
    
    def _validate_query_optimization(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate query optimization result structure."""
        required_fields = ['academic_query', 'web_query', 'wikipedia_query', 'key_concepts']
        
        for field in required_fields:
            if field not in result:
                result[field] = ""
        
        # Ensure lists are actually lists
        list_fields = ['key_concepts', 'alternative_terms', 'broader_terms', 'narrower_terms']
        for field in list_fields:
            if field not in result:
                result[field] = []
            elif not isinstance(result[field], list):
                result[field] = []
        
        return result
    
    def _validate_synthesis_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate synthesis result structure."""
        required_fields = ['summary', 'key_findings', 'confidence_level']
        
        for field in required_fields:
            if field not in result:
                if field == 'key_findings':
                    result[field] = []
                elif field == 'confidence_level':
                    result[field] = 'medium'
                else:
                    result[field] = ""
        
        # Ensure lists are actually lists
        if not isinstance(result.get('key_findings'), list):
            result['key_findings'] = []
        
        if not isinstance(result.get('suggested_followup'), list):
            result['suggested_followup'] = []
        
        return result
    
    def _extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON from response that might contain markdown code blocks."""
        try:
            # Try to find JSON in markdown code blocks
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                if end != -1:
                    json_str = response[start:end].strip()
                    return json.loads(json_str)
            
            # Try to find JSON in the response directly
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
            
            # Fallback: return empty structure
            return {}
            
        except json.JSONDecodeError:
            logger.warning("Failed to extract JSON from LLM response")
            return {}
    
    def _extract_list_from_response(self, response: str) -> List[str]:
        """Extract list from response that might contain markdown."""
        try:
            # Try to find JSON array
            if "[" in response and "]" in response:
                start = response.find("[")
                end = response.rfind("]") + 1
                if start != -1 and end > start:
                    list_str = response[start:end]
                    return json.loads(list_str)
            
            # Fallback: extract lines that look like list items
            lines = response.split('\n')
            items = []
            for line in lines:
                line = line.strip()
                if line.startswith('-') or line.startswith('*') or line.startswith('•'):
                    item = line[1:].strip()
                    if item:
                        items.append(item)
            
            return items[:10]  # Limit to 10 items
            
        except Exception:
            return []
