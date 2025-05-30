"""
Rate limiting utilities for API calls.
"""

import asyncio
import time
import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_limit: int = 10  # Maximum requests in burst
    burst_window_seconds: int = 10  # Burst window duration


class RateLimiter:
    """Rate limiter for API calls with multiple time windows."""
    
    def __init__(self, config: RateLimitConfig, name: str = "default"):
        """Initialize rate limiter with configuration."""
        self.config = config
        self.name = name
        
        # Track requests in different time windows
        self.minute_requests = deque()
        self.hour_requests = deque()
        self.day_requests = deque()
        self.burst_requests = deque()
        
        # Locks for thread safety
        self.lock = asyncio.Lock()
        
        logger.info(f"Rate limiter '{name}' initialized: {config.requests_per_minute}/min, {config.requests_per_hour}/hour")
    
    async def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire permission to make a request.
        
        Args:
            timeout: Maximum time to wait for permission (None for no timeout)
            
        Returns:
            True if permission granted, False if timeout
        """
        start_time = time.time()
        
        while True:
            async with self.lock:
                now = time.time()
                
                # Clean up old requests
                self._cleanup_old_requests(now)
                
                # Check all rate limits
                if self._can_make_request(now):
                    # Record the request
                    self._record_request(now)
                    return True
                
                # Calculate wait time
                wait_time = self._calculate_wait_time(now)
            
            # Check timeout
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    logger.warning(f"Rate limiter '{self.name}' timeout after {elapsed:.2f}s")
                    return False
                
                # Don't wait longer than remaining timeout
                wait_time = min(wait_time, timeout - elapsed)
            
            # Wait before retrying
            if wait_time > 0:
                logger.debug(f"Rate limiter '{self.name}' waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
            else:
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.1)
    
    async def acquire_multiple(self, count: int, timeout: Optional[float] = None) -> bool:
        """
        Acquire permission for multiple requests.
        
        Args:
            count: Number of requests to acquire
            timeout: Maximum time to wait for permission
            
        Returns:
            True if all permissions granted, False if timeout
        """
        start_time = time.time()
        
        while True:
            async with self.lock:
                now = time.time()
                self._cleanup_old_requests(now)
                
                if self._can_make_multiple_requests(now, count):
                    # Record all requests
                    for _ in range(count):
                        self._record_request(now)
                    return True
                
                wait_time = self._calculate_wait_time_for_multiple(now, count)
            
            # Check timeout
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    return False
                wait_time = min(wait_time, timeout - elapsed)
            
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            else:
                await asyncio.sleep(0.1)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current rate limiter statistics."""
        now = time.time()
        self._cleanup_old_requests(now)
        
        return {
            "name": self.name,
            "config": {
                "requests_per_minute": self.config.requests_per_minute,
                "requests_per_hour": self.config.requests_per_hour,
                "requests_per_day": self.config.requests_per_day,
                "burst_limit": self.config.burst_limit,
                "burst_window_seconds": self.config.burst_window_seconds
            },
            "current_usage": {
                "last_minute": len(self.minute_requests),
                "last_hour": len(self.hour_requests),
                "last_day": len(self.day_requests),
                "burst_window": len(self.burst_requests)
            },
            "remaining": {
                "minute": max(0, self.config.requests_per_minute - len(self.minute_requests)),
                "hour": max(0, self.config.requests_per_hour - len(self.hour_requests)),
                "day": max(0, self.config.requests_per_day - len(self.day_requests)),
                "burst": max(0, self.config.burst_limit - len(self.burst_requests))
            }
        }
    
    def reset(self) -> None:
        """Reset all rate limiting counters."""
        self.minute_requests.clear()
        self.hour_requests.clear()
        self.day_requests.clear()
        self.burst_requests.clear()
        logger.info(f"Rate limiter '{self.name}' reset")
    
    def _cleanup_old_requests(self, now: float) -> None:
        """Remove old requests from tracking queues."""
        # Clean minute requests (older than 60 seconds)
        while self.minute_requests and now - self.minute_requests[0] > 60:
            self.minute_requests.popleft()
        
        # Clean hour requests (older than 3600 seconds)
        while self.hour_requests and now - self.hour_requests[0] > 3600:
            self.hour_requests.popleft()
        
        # Clean day requests (older than 86400 seconds)
        while self.day_requests and now - self.day_requests[0] > 86400:
            self.day_requests.popleft()
        
        # Clean burst requests (older than burst window)
        while self.burst_requests and now - self.burst_requests[0] > self.config.burst_window_seconds:
            self.burst_requests.popleft()
    
    def _can_make_request(self, now: float) -> bool:
        """Check if a single request can be made."""
        return (
            len(self.minute_requests) < self.config.requests_per_minute and
            len(self.hour_requests) < self.config.requests_per_hour and
            len(self.day_requests) < self.config.requests_per_day and
            len(self.burst_requests) < self.config.burst_limit
        )
    
    def _can_make_multiple_requests(self, now: float, count: int) -> bool:
        """Check if multiple requests can be made."""
        return (
            len(self.minute_requests) + count <= self.config.requests_per_minute and
            len(self.hour_requests) + count <= self.config.requests_per_hour and
            len(self.day_requests) + count <= self.config.requests_per_day and
            len(self.burst_requests) + count <= self.config.burst_limit
        )
    
    def _record_request(self, now: float) -> None:
        """Record a request in all tracking queues."""
        self.minute_requests.append(now)
        self.hour_requests.append(now)
        self.day_requests.append(now)
        self.burst_requests.append(now)
    
    def _calculate_wait_time(self, now: float) -> float:
        """Calculate minimum wait time before next request."""
        wait_times = []
        
        # Check each limit
        if len(self.minute_requests) >= self.config.requests_per_minute:
            oldest = self.minute_requests[0]
            wait_times.append(60 - (now - oldest))
        
        if len(self.hour_requests) >= self.config.requests_per_hour:
            oldest = self.hour_requests[0]
            wait_times.append(3600 - (now - oldest))
        
        if len(self.day_requests) >= self.config.requests_per_day:
            oldest = self.day_requests[0]
            wait_times.append(86400 - (now - oldest))
        
        if len(self.burst_requests) >= self.config.burst_limit:
            oldest = self.burst_requests[0]
            wait_times.append(self.config.burst_window_seconds - (now - oldest))
        
        return max(wait_times) if wait_times else 0
    
    def _calculate_wait_time_for_multiple(self, now: float, count: int) -> float:
        """Calculate wait time for multiple requests."""
        wait_times = []
        
        # For multiple requests, we need to wait until enough slots are available
        if len(self.minute_requests) + count > self.config.requests_per_minute:
            # Find when enough slots will be available
            needed_slots = len(self.minute_requests) + count - self.config.requests_per_minute
            if needed_slots <= len(self.minute_requests):
                target_time = self.minute_requests[needed_slots - 1]
                wait_times.append(60 - (now - target_time))
        
        # Similar calculations for other time windows...
        if len(self.hour_requests) + count > self.config.requests_per_hour:
            needed_slots = len(self.hour_requests) + count - self.config.requests_per_hour
            if needed_slots <= len(self.hour_requests):
                target_time = self.hour_requests[needed_slots - 1]
                wait_times.append(3600 - (now - target_time))
        
        if len(self.day_requests) + count > self.config.requests_per_day:
            needed_slots = len(self.day_requests) + count - self.config.requests_per_day
            if needed_slots <= len(self.day_requests):
                target_time = self.day_requests[needed_slots - 1]
                wait_times.append(86400 - (now - target_time))
        
        if len(self.burst_requests) + count > self.config.burst_limit:
            needed_slots = len(self.burst_requests) + count - self.config.burst_limit
            if needed_slots <= len(self.burst_requests):
                target_time = self.burst_requests[needed_slots - 1]
                wait_times.append(self.config.burst_window_seconds - (now - target_time))
        
        return max(wait_times) if wait_times else 0


class APIRateLimitManager:
    """Manages rate limiters for different APIs."""
    
    def __init__(self):
        """Initialize rate limit manager with configuration."""
        from .config import get_config
        config = get_config()
        
        self.limiters: Dict[str, RateLimiter] = {}
        
        # Default rate limit configurations for different APIs
        self.default_configs = {
            "google_search": RateLimitConfig(
                requests_per_minute=100,
                requests_per_hour=1000,
                requests_per_day=10000,
                burst_limit=5,
                burst_window_seconds=10
            ),
            "brave_search": RateLimitConfig(
                requests_per_minute=60,
                requests_per_hour=1000,
                requests_per_day=5000,
                burst_limit=5,
                burst_window_seconds=10
            ),
            "semantic_scholar": RateLimitConfig(
                requests_per_minute=100,
                requests_per_hour=1000,
                requests_per_day=10000,
                burst_limit=10,
                burst_window_seconds=10
            ),
            "arxiv": RateLimitConfig(
                requests_per_minute=30,
                requests_per_hour=1000,
                requests_per_day=5000,
                burst_limit=3,
                burst_window_seconds=10
            ),
            "core_api": RateLimitConfig(
                requests_per_minute=60,
                requests_per_hour=1000,
                requests_per_day=5000,
                burst_limit=5,
                burst_window_seconds=10
            ),
            "wikipedia": RateLimitConfig(
                requests_per_minute=200,
                requests_per_hour=5000,
                requests_per_day=50000,
                burst_limit=20,
                burst_window_seconds=10
            ),
            "gemini": RateLimitConfig(
                requests_per_minute=60,
                requests_per_hour=1000,
                requests_per_day=10000,
                burst_limit=5,
                burst_window_seconds=30
            )
        }
        
        # Initialize rate limiters
        self._initialize_limiters()
        
        logger.info(f"Rate limit manager initialized for {len(self.limiters)} APIs")
    
    def get_limiter(self, api_name: str) -> RateLimiter:
        """Get rate limiter for a specific API."""
        if api_name not in self.limiters:
            # Create a default limiter
            default_config = RateLimitConfig()
            self.limiters[api_name] = RateLimiter(default_config, api_name)
            logger.warning(f"Created default rate limiter for unknown API: {api_name}")
        
        return self.limiters[api_name]
    
    async def acquire(self, api_name: str, timeout: Optional[float] = None) -> bool:
        """Acquire permission for an API call."""
        limiter = self.get_limiter(api_name)
        return await limiter.acquire(timeout)
    
    async def acquire_multiple(self, api_name: str, count: int, timeout: Optional[float] = None) -> bool:
        """Acquire permission for multiple API calls."""
        limiter = self.get_limiter(api_name)
        return await limiter.acquire_multiple(count, timeout)
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all rate limiters."""
        return {name: limiter.get_stats() for name, limiter in self.limiters.items()}
    
    def reset_limiter(self, api_name: str) -> bool:
        """Reset a specific rate limiter."""
        if api_name in self.limiters:
            self.limiters[api_name].reset()
            return True
        return False
    
    def reset_all_limiters(self) -> None:
        """Reset all rate limiters."""
        for limiter in self.limiters.values():
            limiter.reset()
        logger.info("All rate limiters reset")
    
    def update_config(self, api_name: str, config: RateLimitConfig) -> None:
        """Update configuration for a specific API."""
        self.limiters[api_name] = RateLimiter(config, api_name)
        logger.info(f"Updated rate limiter config for {api_name}")
    
    def _initialize_limiters(self) -> None:
        """Initialize rate limiters based on configuration."""
        # 使用默认配置创建限流器
        for api_name, default_config in self.default_configs.items():
            self.limiters[api_name] = RateLimiter(default_config, api_name)
            
        logger.info(f"Initialized rate limiters for {len(self.limiters)} APIs")
            
            self.limiters[api_name] = RateLimiter(config, api_name)
