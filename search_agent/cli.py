"""
Command Line Interface for the Search Agent.
"""

import asyncio
import argparse
import json
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from ..core.agent import SearchAgent
from ..utils.config import Config
from ..exceptions.custom_exceptions import SearchAgentError, ConfigurationError


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Setup logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup handlers
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        handlers=handlers,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


async def search_command(args: argparse.Namespace) -> None:
    """Execute search command."""
    try:
        # Load configuration
        config = None
        if args.config:
            config_path = Path(args.config)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
            else:
                print(f"Warning: Config file {args.config} not found, using defaults")
        
        # Initialize agent
        agent = SearchAgent(config)
        
        # Perform search
        result = await agent.search(
            query=args.query,
            output_format=args.format,
            save_to_file=args.output,
            include_raw_results=not args.no_raw_results,
            max_results_per_source=args.max_results,
            use_cache=not args.no_cache,
            timeout=args.timeout
        )
        
        # Display results
        if args.format == "json":
            if args.output:
                print(f"Results saved to: {args.output}")
            else:
                print(json.dumps(result, indent=2, ensure_ascii=False))
        
        elif args.format == "summary":
            summary = agent.report_generator.generate_quick_summary(result)
            print(summary)
        
        else:
            if args.output:
                print(f"Report saved to: {args.output}")
            else:
                # Generate and display report
                if args.format == "markdown":
                    report = agent.report_generator.generate_markdown_report(result)
                else:
                    report = agent.report_generator.generate_text_report(result)
                print(report)
        
        # Show summary stats
        metadata = result.get("metadata", {})
        print(f"\n--- Search Summary ---")
        print(f"Total results: {metadata.get('total_results', 0)}")
        print(f"Sources searched: {', '.join(metadata.get('sources_searched', []))}")
        print(f"Execution time: {metadata.get('execution_time_seconds', 0):.2f}s")
        
        if metadata.get('errors'):
            print(f"Errors: {len(metadata['errors'])}")
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


async def batch_search_command(args: argparse.Namespace) -> None:
    """Execute batch search command."""
    try:
        # Load queries
        queries_file = Path(args.queries_file)
        if not queries_file.exists():
            print(f"Error: Queries file {args.queries_file} not found", file=sys.stderr)
            sys.exit(1)
        
        with open(queries_file, 'r') as f:
            if args.queries_file.endswith('.json'):
                queries = json.load(f)
            else:
                queries = [line.strip() for line in f if line.strip()]
        
        if not queries:
            print("Error: No queries found in file", file=sys.stderr)
            sys.exit(1)
        
        # Load configuration
        config = None
        if args.config:
            config_path = Path(args.config)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
        
        # Initialize agent
        agent = SearchAgent(config)
        
        # Perform batch search
        print(f"Starting batch search for {len(queries)} queries...")
        
        results = await agent.batch_search(
            queries=queries,
            output_format=args.format,
            save_to_directory=args.output_dir,
            max_concurrent=args.max_concurrent
        )
        
        # Save summary
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(exist_ok=True)
            
            summary = {
                "total_queries": len(queries),
                "successful_searches": len(results),
                "failed_searches": len(queries) - len(results),
                "results": results
            }
            
            summary_file = output_dir / "batch_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            print(f"Batch search completed. Results saved to: {args.output_dir}")
            print(f"Summary saved to: {summary_file}")
        
        print(f"\nBatch Search Summary:")
        print(f"Total queries: {len(queries)}")
        print(f"Successful: {len(results)}")
        print(f"Failed: {len(queries) - len(results)}")
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


async def health_check_command(args: argparse.Namespace) -> None:
    """Execute health check command."""
    try:
        # Load configuration
        config = None
        if args.config:
            config_path = Path(args.config)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
        
        # Initialize agent
        agent = SearchAgent(config)
        
        # Perform health check
        health_status = await agent.health_check()
        
        # Display health status
        print(json.dumps(health_status, indent=2, ensure_ascii=False))
        
        # Exit with appropriate code
        if health_status["status"] == "healthy":
            sys.exit(0)
        elif health_status["status"] == "degraded":
            sys.exit(1)
        else:
            sys.exit(2)
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(3)


def create_config_command(args: argparse.Namespace) -> None:
    """Create a sample configuration file."""
    try:
        config_template = {
            "api_keys": {
                "GOOGLE_API_KEY": "your_google_api_key_here",
                "GOOGLE_CSE_ID": "your_google_cse_id_here",
                "BRAVE_API_KEY": "your_brave_api_key_here",
                "GEMINI_API_KEY": "your_gemini_api_key_here"
            },
            "search_settings": {
                "max_results_per_source": 10,
                "timeout_seconds": 300,
                "use_cache": True,
                "cache_ttl_hours": 24
            },
            "gemini_config": {
                "model": "gemini-1.5-flash",
                "temperature": 0.3,
                "max_tokens": 2048
            },
            "rate_limits": {
                "google_search": {
                    "requests_per_minute": 100,
                    "requests_per_hour": 1000,
                    "requests_per_day": 10000
                },
                "brave_search": {
                    "requests_per_minute": 60,
                    "requests_per_hour": 1000,
                    "requests_per_day": 5000
                },
                "gemini": {
                    "requests_per_minute": 60,
                    "requests_per_hour": 1000,
                    "requests_per_day": 10000
                }
            },
            "cache_settings": {
                "cache_directory": ".cache",
                "max_cache_size_mb": 100,
                "cache_enabled": True
            },
            "output_settings": {
                "include_raw_results": True,
                "include_metadata": True,
                "max_results_per_source": 10
            }
        }
        
        output_file = args.output or "config.json"
        
        with open(output_file, 'w') as f:
            json.dump(config_template, f, indent=2)
        
        print(f"Sample configuration created: {output_file}")
        print("Please update the API keys and settings as needed.")
        
    except Exception as e:
        print(f"Error creating config: {str(e)}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Source Intelligent Search Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple search
  search-agent search "machine learning transformers"
  
  # Search with custom output format
  search-agent search "climate change" --format markdown --output report.md
  
  # Batch search from file
  search-agent batch-search queries.txt --output-dir results/
  
  # Health check
  search-agent health-check
  
  # Create sample config
  search-agent create-config --output my_config.json
        """
    )
    
    # Global arguments
    parser.add_argument("--config", "-c", help="Configuration file path")
    parser.add_argument("--log-level", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="Logging level")
    parser.add_argument("--log-file", help="Log file path")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Perform a search")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--format", "-f", default="json",
                              choices=["json", "markdown", "text", "summary"],
                              help="Output format")
    search_parser.add_argument("--output", "-o", help="Output file path")
    search_parser.add_argument("--max-results", type=int, default=10,
                              help="Maximum results per source")
    search_parser.add_argument("--timeout", type=float, default=300.0,
                              help="Search timeout in seconds")
    search_parser.add_argument("--no-cache", action="store_true",
                              help="Disable caching")
    search_parser.add_argument("--no-raw-results", action="store_true",
                              help="Exclude raw search results from output")
    
    # Batch search command
    batch_parser = subparsers.add_parser("batch-search", help="Perform batch searches")
    batch_parser.add_argument("queries_file", help="File containing queries (one per line or JSON)")
    batch_parser.add_argument("--output-dir", "-d", help="Output directory for results")
    batch_parser.add_argument("--format", "-f", default="json",
                             choices=["json", "markdown", "text"],
                             help="Output format for each search")
    batch_parser.add_argument("--max-concurrent", type=int, default=3,
                             help="Maximum concurrent searches")
    
    # Health check command
    health_parser = subparsers.add_parser("health-check", help="Check agent health")
    
    # Create config command
    config_parser = subparsers.add_parser("create-config", help="Create sample configuration file")
    config_parser.add_argument("--output", "-o", help="Output file path (default: config.json)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    # Execute command
    if args.command == "search":
        asyncio.run(search_command(args))
    elif args.command == "batch-search":
        asyncio.run(batch_search_command(args))
    elif args.command == "health-check":
        asyncio.run(health_check_command(args))
    elif args.command == "create-config":
        create_config_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
