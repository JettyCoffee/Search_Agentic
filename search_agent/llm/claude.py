"""Claude LLM integration for the Multi-Source Search Agent."""

import logging
import json
from typing import Dict, List, Any, Optional
import asyncio
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from ..exceptions.custom_exceptions import LLMConfigError, LLMAPIError, QueryOptimizationError

logger = logging.getLogger(__name__)


class ClaudeLLM:
    """Claude LLM integration using OpenAI-compatible API."""
    
    def __init__(self):
        """Initialize Claude LLM with configuration."""
        from ..utils.config import get_config
        self.config = get_config()
        
        # Claude API配置
        self.api_key = self.config.api.claude_api_key
        self.base_url = self.config.api.claude_base_url
        self.model_name = self.config.api.claude_model
        
        if not self.api_key:
            raise LLMConfigError("Claude API key not configured")
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        logger.info("Claude LLM initialized successfully")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _make_api_request(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """Make API request to Claude using OpenAI client."""
        try:
            # 使用同步客户端，在asyncio中运行
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=2000
                )
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise LLMAPIError(f"Claude API request failed: {str(e)}")
    
    async def extract_key_concepts(self, query: str) -> List[str]:
        """Extract key concepts from a search query."""
        messages = [
            {
                "role": "system",
                "content": "You are a research assistant. Extract key concepts from the user's query and return them as a JSON list."
            },
            {
                "role": "user",
                "content": f"""
                Please extract the key concepts from this search query: "{query}"
                
                Return only a JSON array of key concepts, like: ["concept1", "concept2", "concept3"]
                Focus on the most important terms that would be useful for searching.
                """
            }
        ]
        
        try:
            response = await self._make_api_request(messages, temperature=0.3)
            # 尝试解析JSON响应
            concepts = json.loads(response.strip())
            if isinstance(concepts, list):
                return concepts
            else:
                return [query]  # 回退到原始查询
        except Exception as e:
            logger.warning(f"Key concept extraction failed: {str(e)}")
            return [query]
    
    async def optimize_query(self, original_query: str, context: str = "") -> Dict[str, str]:
        """Optimize search queries for different sources."""
        messages = [
            {
                "role": "system",
                "content": """You are a search optimization expert. Given a user query and optional context, 
                create optimized search queries for different search engines and databases."""
            },
            {
                "role": "user",
                "content": f"""
                Original query: "{original_query}"
                Context: "{context}"
                
                Please create optimized search queries for these sources:
                1. arxiv - for academic papers
                2. semantic_scholar - for academic research
                3. google - for general web search
                4. brave - for alternative web search
                
                Return your response as a JSON object with these exact keys:
                {{
                    "arxiv": "optimized query for arxiv",
                    "semantic_scholar": "optimized query for semantic scholar",
                    "google": "optimized query for google search",
                    "brave": "optimized query for brave search"
                }}
                
                Make the queries specific to each platform's strengths.
                """
            }
        ]
        
        try:
            response = await self._make_api_request(messages, temperature=0.5)
            optimized_queries = json.loads(response.strip())
            
            # 验证响应格式
            required_keys = ["arxiv", "semantic_scholar", "google", "brave"]
            if all(key in optimized_queries for key in required_keys):
                return optimized_queries
            else:
                raise QueryOptimizationError("Invalid optimization response format")
                
        except Exception as e:
            logger.error(f"Query optimization failed: {str(e)}")
            # 回退策略：返回原始查询
            return {
                "arxiv": original_query,
                "semantic_scholar": original_query,
                "google": original_query,
                "brave": original_query
            }
    
    async def synthesize_results(self, query: str, search_results: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Synthesize search results from multiple sources."""
        # 准备搜索结果摘要
        results_summary = []
        for source, results in search_results.items():
            if results:
                results_summary.append(f"**{source.upper()}**: {len(results)} results found")
                for i, result in enumerate(results[:3]):  # 只显示前3个结果
                    title = result.get('title', 'No title')
                    snippet = result.get('snippet', result.get('description', 'No description'))[:200]
                    results_summary.append(f"  {i+1}. {title}: {snippet}")
        
        results_text = "\n".join(results_summary)
        
        messages = [
            {
                "role": "system",
                "content": """You are a research analyst. Synthesize search results from multiple sources 
                into a comprehensive summary. Focus on key findings, patterns, and insights."""
            },
            {
                "role": "user",
                "content": f"""
                Query: "{query}"
                
                Search Results:
                {results_text}
                
                Please provide a comprehensive synthesis in this JSON format:
                {{
                    "summary": "Overall summary of findings",
                    "key_findings": ["finding 1", "finding 2", "finding 3"],
                    "confidence_level": "high/medium/low",
                    "limitations": "Any limitations or gaps in the data",
                    "recommendations": ["recommendation 1", "recommendation 2"]
                }}
                """
            }
        ]
        
        try:
            response = await self._make_api_request(messages, temperature=0.6)
            synthesis = json.loads(response.strip())
            
            # 验证响应格式
            required_keys = ["summary", "key_findings", "confidence_level"]
            if all(key in synthesis for key in required_keys):
                return synthesis
            else:
                raise Exception("Invalid synthesis response format")
                
        except Exception as e:
            logger.error(f"Result synthesis failed: {str(e)}")
            # 回退合成
            total_results = sum(len(results) for results in search_results.values())
            return {
                "summary": f"Found {total_results} results across {len(search_results)} sources for query: {query}",
                "key_findings": [f"Search completed across {len(search_results)} sources"],
                "confidence_level": "low",
                "limitations": "Synthesis processing encountered errors",
                "recommendations": ["Review individual search results for detailed information"]
            }
