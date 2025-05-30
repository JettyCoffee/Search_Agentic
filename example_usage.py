#!/usr/bin/env python3
"""
Example usage script for the Search Agent.

This script demonstrates how to use the search agent to perform
intelligent searches across multiple sources.
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from search_agent import SearchAgent, setup_logging, get_logger


async def basic_search_example():
    """Demonstrate basic search functionality."""
    print("🔍 Basic Search Example")
    print("-" * 50)
    
    # Create search agent with configuration
    config = {
        "log_level": "INFO",
        "max_results_per_source": 3,
        "cache_type": "memory"
    }
    
    agent = SearchAgent(config=config)
    logger = get_logger("example")
    
    try:
        # Perform a search
        query = "artificial intelligence in healthcare"
        logger.info(f"Searching for: {query}")
        
        # Mock the search since we don't have real API keys
        print(f"Would search for: '{query}'")
        print("This would return results from multiple sources:")
        print("- Wikipedia: Background information")
        print("- Academic papers: Latest research")
        print("- Web search: Current news and articles")
        
        # In a real scenario with API keys:
        # result = await agent.search(query)
        # print(json.dumps(result, indent=2))
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        print(f"❌ Search failed: {e}")


async def batch_search_example():
    """Demonstrate batch search functionality."""
    print("\n📚 Batch Search Example")
    print("-" * 50)
    
    agent = SearchAgent()
    logger = get_logger("batch_example")
    
    queries = [
        "machine learning algorithms",
        "quantum computing applications", 
        "blockchain technology",
        "renewable energy solutions"
    ]
    
    try:
        logger.info(f"Performing batch search for {len(queries)} queries")
        
        print("Would perform batch search for:")
        for i, query in enumerate(queries, 1):
            print(f"{i}. {query}")
        
        print("\nThis would execute searches concurrently and return aggregated results.")
        
        # In a real scenario:
        # results = await agent.batch_search(queries, max_concurrent=2)
        # for i, result in enumerate(results):
        #     print(f"\nResult {i+1}: {result['query']}")
        #     print(f"Sources: {result['metadata']['sources_used']}")
        
    except Exception as e:
        logger.error(f"Batch search failed: {e}")
        print(f"❌ Batch search failed: {e}")


async def advanced_search_example():
    """Demonstrate advanced search features."""
    print("\n🚀 Advanced Search Example")
    print("-" * 50)
    
    agent = SearchAgent()
    logger = get_logger("advanced_example")
    
    try:
        query = "deep learning neural networks"
        sources = ["wikipedia", "arxiv", "semantic_scholar"]
        
        logger.info(f"Advanced search: {query} using sources: {sources}")
        
        print(f"Query: {query}")
        print(f"Specific sources: {', '.join(sources)}")
        print("Additional options:")
        print("- Max results: 5 per source")
        print("- Include abstracts: Yes")
        print("- Language: English")
        
        # In a real scenario:
        # result = await agent.search(
        #     query,
        #     sources=sources,
        #     max_results=5,
        #     include_abstracts=True,
        #     language="en"
        # )
        
        print("\nThis would return structured results with:")
        print("- Query optimization using Google Gemini")
        print("- Context from Wikipedia")
        print("- Academic papers from ArXiv and Semantic Scholar")
        print("- Comprehensive summary and analysis")
        
    except Exception as e:
        logger.error(f"Advanced search failed: {e}")
        print(f"❌ Advanced search failed: {e}")


async def health_check_example():
    """Demonstrate health check functionality."""
    print("\n🏥 Health Check Example")
    print("-" * 50)
    
    agent = SearchAgent()
    logger = get_logger("health_example")
    
    try:
        # Perform health check
        health = await agent.health_check()
        
        print("System Health Status:")
        print(f"Overall Status: {health.get('status', 'unknown')}")
        
        if 'tools' in health:
            print("\nTool Status:")
            for tool, status in health['tools'].items():
                status_icon = "✅" if status else "❌"
                print(f"  {status_icon} {tool}: {'OK' if status else 'ERROR'}")
        
        if 'system' in health:
            print(f"\nSystem Info:")
            for key, value in health['system'].items():
                print(f"  {key}: {value}")
                
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        print(f"❌ Health check failed: {e}")


def configuration_example():
    """Demonstrate configuration options."""
    print("\n⚙️  Configuration Example")
    print("-" * 50)
    
    # Example configurations
    configs = {
        "Development": {
            "log_level": "DEBUG",
            "cache_type": "memory",
            "rate_limit_enabled": False,
            "max_results_per_source": 5
        },
        "Production": {
            "log_level": "INFO",
            "cache_type": "file",
            "cache_ttl": 3600,
            "rate_limit_enabled": True,
            "rate_limit_requests_per_minute": 60,
            "max_results_per_source": 10
        },
        "Research": {
            "log_level": "INFO",
            "cache_type": "file",
            "default_sources": ["arxiv", "semantic_scholar", "wikipedia"],
            "max_results_per_source": 20,
            "include_abstracts": True
        }
    }
    
    for env_name, config in configs.items():
        print(f"\n{env_name} Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    print("\nEnvironment Variables (required):")
    required_vars = [
        "GOOGLE_API_KEY",
        "GOOGLE_CSE_ID", 
        "GEMINI_API_KEY",
        "BRAVE_API_KEY",
        "CORE_API_KEY"
    ]
    
    for var in required_vars:
        status = "✅ Set" if os.getenv(var) else "❌ Missing"
        print(f"  {var}: {status}")


async def metrics_example():
    """Demonstrate metrics and monitoring."""
    print("\n📊 Metrics and Monitoring Example")
    print("-" * 50)
    
    agent = SearchAgent()
    
    # Simulate some search history
    agent.search_history = [
        {
            "query": "artificial intelligence", 
            "timestamp": datetime.now(),
            "sources_used": ["wikipedia", "arxiv"],
            "search_time": 2.3
        },
        {
            "query": "machine learning",
            "timestamp": datetime.now(),
            "sources_used": ["google_search", "semantic_scholar"],
            "search_time": 1.8
        }
    ]
    
    # Get statistics
    stats = agent.get_stats()
    
    print("Search Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nRecent Search History:")
    history = agent.get_search_history(limit=5)
    for i, item in enumerate(history, 1):
        print(f"  {i}. {item.get('query', 'Unknown')}")


async def main():
    """Run all examples."""
    # Setup logging
    setup_logging(log_level="INFO")
    
    print("🤖 Search Agent - Example Usage")
    print("=" * 60)
    
    # Run examples
    await basic_search_example()
    await batch_search_example()
    await advanced_search_example()
    await health_check_example()
    configuration_example()
    await metrics_example()
    
    print("\n" + "=" * 60)
    print("✨ Examples completed!")
    print("\nTo use the search agent with real API keys:")
    print("1. Copy .env.example to .env")
    print("2. Fill in your API keys")
    print("3. Run: python -m search_agent.cli search 'your query'")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
