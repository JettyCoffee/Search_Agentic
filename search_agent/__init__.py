"""
Search Agent - A comprehensive multi-source intelligent search framework.

This package provides a complete solution for intelligent search across
multiple sources including web search engines, academic databases, and
knowledge bases. It leverages LangChain and LangGraph for orchestration,
Google Gemini for query optimization, and provides structured JSON outputs.

Main Components:
- SearchAgent: Main interface for search operations
- Multiple search tools (Wikipedia, Google, Brave, Semantic Scholar, ArXiv, CORE)
- LangGraph workflow orchestration
- Google Gemini LLM integration
- Comprehensive output formatting and reporting
- Caching, rate limiting, and performance monitoring

Example Usage:
    from search_agent import SearchAgent
    
    agent = SearchAgent()
    results = await agent.search("artificial intelligence machine learning")
    print(results)
"""

from .core.agent import SearchAgent
from .utils.config import Config
from .utils.logging import setup_logging, get_logger
from .exceptions.custom_exceptions import (
    SearchAgentError,
    ConfigurationError,
    SearchToolError,
    LLMError,
    RateLimitError,
    ValidationError,
    CacheError,
)

# Version information
__version__ = "1.0.0"
__author__ = "Search Agent Team"
__email__ = "contact@searchagent.ai"
__license__ = "MIT"

# Package metadata
__title__ = "search-agent"
__description__ = "A comprehensive multi-source intelligent search framework"
__url__ = "https://github.com/your-org/search-agent"

# Export main classes and functions
__all__ = [
    # Main classes
    "SearchAgent",
    "Config",
    
    # Utility functions
    "setup_logging",
    "get_logger",
    
    # Exceptions
    "SearchAgentError",
    "ConfigurationError", 
    "SearchToolError",
    "LLMError",
    "RateLimitError",
    "ValidationError",
    "CacheError",
    
    # Version info
    "__version__",
]

# Initialize logging by default
setup_logging()

# Module-level logger
logger = get_logger(__name__)
