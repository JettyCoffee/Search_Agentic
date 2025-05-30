"""
Caching utilities for the search agent.
"""

import json
import hashlib
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Union
from pathlib import Path
import aiofiles
import pickle

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages caching for search results and API responses."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize cache manager with configuration."""
        self.config = config
        self.cache_dir = Path(config.get("cache_directory", ".cache"))
        self.cache_ttl = config.get("cache_ttl_hours", 24)  # TTL in hours
        self.max_cache_size = config.get("max_cache_size_mb", 100)  # Max size in MB
        self.enabled = config.get("cache_enabled", True)
        
        # Create cache directory
        self.cache_dir.mkdir(exist_ok=True)
        
        # Separate subdirectories for different types of cache
        self.search_cache_dir = self.cache_dir / "search_results"
        self.llm_cache_dir = self.cache_dir / "llm_responses"
        self.context_cache_dir = self.cache_dir / "context"
        
        for cache_subdir in [self.search_cache_dir, self.llm_cache_dir, self.context_cache_dir]:
            cache_subdir.mkdir(exist_ok=True)
        
        # In-memory cache for frequently accessed items
        self.memory_cache = {}
        self.memory_cache_ttl = {}
        
        logger.info(f"Cache manager initialized - Directory: {self.cache_dir}, TTL: {self.cache_ttl}h")
    
    async def get_search_result(self, tool_name: str, query: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Get cached search result.
        
        Args:
            tool_name: Name of the search tool
            query: Search query
            **kwargs: Additional search parameters
            
        Returns:
            Cached result if available and valid, None otherwise
        """
        if not self.enabled:
            return None
        
        try:
            cache_key = self._generate_search_cache_key(tool_name, query, **kwargs)
            
            # Check memory cache first
            memory_result = await self._get_from_memory_cache(cache_key)
            if memory_result is not None:
                return memory_result
            
            # Check file cache
            cache_file = self.search_cache_dir / f"{cache_key}.json"
            
            if cache_file.exists():
                # Check if cache is still valid
                if await self._is_cache_valid(cache_file):
                    async with aiofiles.open(cache_file, 'r', encoding='utf-8') as f:
                        content = await f.read()
                        result = json.loads(content)
                    
                    # Store in memory cache for faster access
                    await self._store_in_memory_cache(cache_key, result)
                    
                    logger.debug(f"Cache hit for {tool_name} search: {query[:50]}...")
                    return result
                else:
                    # Remove expired cache
                    await self._remove_cache_file(cache_file)
            
            return None
            
        except Exception as e:
            logger.warning(f"Error retrieving from cache: {str(e)}")
            return None
    
    async def store_search_result(self, tool_name: str, query: str, result: Dict[str, Any], **kwargs) -> bool:
        """
        Store search result in cache.
        
        Args:
            tool_name: Name of the search tool
            query: Search query
            result: Search result to cache
            **kwargs: Additional search parameters
            
        Returns:
            True if stored successfully, False otherwise
        """
        if not self.enabled or not result:
            return False
        
        try:
            cache_key = self._generate_search_cache_key(tool_name, query, **kwargs)
            
            # Add metadata to the cached result
            cached_data = {
                "data": result,
                "cached_at": datetime.now().isoformat(),
                "tool_name": tool_name,
                "query": query,
                "parameters": kwargs
            }
            
            # Store in file cache
            cache_file = self.search_cache_dir / f"{cache_key}.json"
            async with aiofiles.open(cache_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(cached_data, indent=2, ensure_ascii=False))
            
            # Store in memory cache
            await self._store_in_memory_cache(cache_key, cached_data)
            
            logger.debug(f"Cached {tool_name} search result: {query[:50]}...")
            
            # Clean up old cache files if needed
            await self._cleanup_cache_if_needed()
            
            return True
            
        except Exception as e:
            logger.warning(f"Error storing in cache: {str(e)}")
            return False
    
    async def get_llm_response(self, prompt: str, model_config: Dict[str, Any]) -> Optional[str]:
        """
        Get cached LLM response.
        
        Args:
            prompt: LLM prompt
            model_config: Model configuration parameters
            
        Returns:
            Cached response if available and valid, None otherwise
        """
        if not self.enabled:
            return None
        
        try:
            cache_key = self._generate_llm_cache_key(prompt, model_config)
            
            # Check memory cache first
            memory_result = await self._get_from_memory_cache(cache_key)
            if memory_result is not None:
                return memory_result.get("response")
            
            # Check file cache
            cache_file = self.llm_cache_dir / f"{cache_key}.json"
            
            if cache_file.exists():
                if await self._is_cache_valid(cache_file):
                    async with aiofiles.open(cache_file, 'r', encoding='utf-8') as f:
                        content = await f.read()
                        result = json.loads(content)
                    
                    await self._store_in_memory_cache(cache_key, result)
                    
                    logger.debug(f"Cache hit for LLM response: {prompt[:50]}...")
                    return result.get("response")
                else:
                    await self._remove_cache_file(cache_file)
            
            return None
            
        except Exception as e:
            logger.warning(f"Error retrieving LLM cache: {str(e)}")
            return None
    
    async def store_llm_response(self, prompt: str, response: str, model_config: Dict[str, Any]) -> bool:
        """
        Store LLM response in cache.
        
        Args:
            prompt: LLM prompt
            response: LLM response
            model_config: Model configuration parameters
            
        Returns:
            True if stored successfully, False otherwise
        """
        if not self.enabled or not response:
            return False
        
        try:
            cache_key = self._generate_llm_cache_key(prompt, model_config)
            
            cached_data = {
                "response": response,
                "cached_at": datetime.now().isoformat(),
                "prompt_hash": hashlib.md5(prompt.encode()).hexdigest(),
                "model_config": model_config
            }
            
            cache_file = self.llm_cache_dir / f"{cache_key}.json"
            async with aiofiles.open(cache_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(cached_data, indent=2, ensure_ascii=False))
            
            await self._store_in_memory_cache(cache_key, cached_data)
            
            logger.debug(f"Cached LLM response: {prompt[:50]}...")
            
            await self._cleanup_cache_if_needed()
            
            return True
            
        except Exception as e:
            logger.warning(f"Error storing LLM cache: {str(e)}")
            return False
    
    async def clear_cache(self, cache_type: Optional[str] = None) -> bool:
        """
        Clear cache files.
        
        Args:
            cache_type: Type of cache to clear ("search", "llm", "context", or None for all)
            
        Returns:
            True if cleared successfully, False otherwise
        """
        try:
            if cache_type == "search":
                await self._clear_directory(self.search_cache_dir)
            elif cache_type == "llm":
                await self._clear_directory(self.llm_cache_dir)
            elif cache_type == "context":
                await self._clear_directory(self.context_cache_dir)
            else:
                # Clear all caches
                await self._clear_directory(self.search_cache_dir)
                await self._clear_directory(self.llm_cache_dir)
                await self._clear_directory(self.context_cache_dir)
            
            # Clear memory cache
            self.memory_cache.clear()
            self.memory_cache_ttl.clear()
            
            logger.info(f"Cache cleared: {cache_type or 'all'}")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            return False
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            stats = {
                "enabled": self.enabled,
                "cache_directory": str(self.cache_dir),
                "ttl_hours": self.cache_ttl,
                "max_size_mb": self.max_cache_size
            }
            
            # Count files and calculate sizes
            for cache_type, cache_dir in [
                ("search", self.search_cache_dir),
                ("llm", self.llm_cache_dir),
                ("context", self.context_cache_dir)
            ]:
                file_count = len(list(cache_dir.glob("*.json")))
                size_bytes = sum(f.stat().st_size for f in cache_dir.glob("*.json"))
                size_mb = size_bytes / (1024 * 1024)
                
                stats[f"{cache_type}_files"] = file_count
                stats[f"{cache_type}_size_mb"] = round(size_mb, 2)
            
            # Total stats
            total_files = sum(stats[f"{t}_files"] for t in ["search", "llm", "context"])
            total_size = sum(stats[f"{t}_size_mb"] for t in ["search", "llm", "context"])
            
            stats["total_files"] = total_files
            stats["total_size_mb"] = round(total_size, 2)
            stats["memory_cache_items"] = len(self.memory_cache)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {str(e)}")
            return {"error": str(e)}
    
    def _generate_search_cache_key(self, tool_name: str, query: str, **kwargs) -> str:
        """Generate cache key for search results."""
        # Create a consistent key from tool name, query, and parameters
        key_data = {
            "tool": tool_name,
            "query": query.strip().lower(),
            "params": sorted(kwargs.items())
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _generate_llm_cache_key(self, prompt: str, model_config: Dict[str, Any]) -> str:
        """Generate cache key for LLM responses."""
        key_data = {
            "prompt": prompt.strip(),
            "config": sorted(model_config.items())
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def _is_cache_valid(self, cache_file: Path) -> bool:
        """Check if cache file is still valid based on TTL."""
        try:
            file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            expiry_time = file_time + timedelta(hours=self.cache_ttl)
            return datetime.now() < expiry_time
        except Exception:
            return False
    
    async def _get_from_memory_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get item from memory cache if valid."""
        if cache_key not in self.memory_cache:
            return None
        
        # Check TTL
        if cache_key in self.memory_cache_ttl:
            if datetime.now() > self.memory_cache_ttl[cache_key]:
                # Expired, remove from memory cache
                self.memory_cache.pop(cache_key, None)
                self.memory_cache_ttl.pop(cache_key, None)
                return None
        
        return self.memory_cache[cache_key]
    
    async def _store_in_memory_cache(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Store item in memory cache with TTL."""
        self.memory_cache[cache_key] = data
        self.memory_cache_ttl[cache_key] = datetime.now() + timedelta(hours=1)  # 1 hour memory cache
        
        # Limit memory cache size (keep last 100 items)
        if len(self.memory_cache) > 100:
            oldest_key = min(self.memory_cache_ttl.keys(), key=lambda k: self.memory_cache_ttl[k])
            self.memory_cache.pop(oldest_key, None)
            self.memory_cache_ttl.pop(oldest_key, None)
    
    async def _remove_cache_file(self, cache_file: Path) -> None:
        """Remove a cache file."""
        try:
            cache_file.unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Error removing cache file {cache_file}: {str(e)}")
    
    async def _clear_directory(self, directory: Path) -> None:
        """Clear all files in a directory."""
        for file_path in directory.glob("*.json"):
            await self._remove_cache_file(file_path)
    
    async def _cleanup_cache_if_needed(self) -> None:
        """Clean up cache if it exceeds size limits."""
        try:
            # Calculate total cache size
            total_size = 0
            all_files = []
            
            for cache_dir in [self.search_cache_dir, self.llm_cache_dir, self.context_cache_dir]:
                for file_path in cache_dir.glob("*.json"):
                    size = file_path.stat().st_size
                    total_size += size
                    all_files.append((file_path, size, file_path.stat().st_mtime))
            
            # Check if cleanup is needed
            size_mb = total_size / (1024 * 1024)
            if size_mb > self.max_cache_size:
                logger.info(f"Cache size ({size_mb:.2f} MB) exceeds limit ({self.max_cache_size} MB), cleaning up...")
                
                # Sort files by modification time (oldest first)
                all_files.sort(key=lambda x: x[2])
                
                # Remove oldest files until under limit
                current_size = total_size
                for file_path, file_size, _ in all_files:
                    if current_size / (1024 * 1024) <= self.max_cache_size * 0.8:  # Clean to 80% of limit
                        break
                    
                    await self._remove_cache_file(file_path)
                    current_size -= file_size
                
                logger.info("Cache cleanup completed")
                
        except Exception as e:
            logger.warning(f"Error during cache cleanup: {str(e)}")
