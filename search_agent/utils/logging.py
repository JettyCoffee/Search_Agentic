"""
Logging configuration and utilities for the search agent.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any

import structlog
from structlog import processors


class SearchAgentLogger:
    """Custom logger for the search agent with structured logging support."""
    
    def __init__(self, name: str = "search_agent", log_level: str = "INFO"):
        self.name = name
        self.log_level = log_level.upper()
        self._setup_logging()
    
    def _setup_logging(self):
        """Set up structured logging with both console and file handlers."""
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Configure structlog
        structlog.configure(
            processors=[
                processors.TimeStamper(fmt="ISO"),
                processors.add_log_level,
                processors.add_logger_name,
                processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(
                getattr(logging, self.log_level)
            ),
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        
        # Set up standard logging
        logger = logging.getLogger(self.name)
        logger.setLevel(getattr(logging, self.log_level))
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Console handler with color formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, self.log_level))
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / f"{self.name}.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Error file handler
        error_handler = logging.handlers.RotatingFileHandler(
            log_dir / f"{self.name}_errors.log",
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        logger.addHandler(error_handler)
        
        self.logger = logger
        self.struct_logger = structlog.get_logger(self.name)
    
    def get_logger(self) -> logging.Logger:
        """Get the standard logger instance."""
        return self.logger
    
    def get_struct_logger(self) -> structlog.BoundLogger:
        """Get the structured logger instance."""
        return self.struct_logger


class PerformanceMonitor:
    """Monitor and log performance metrics."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("performance")
        self.metrics: Dict[str, Any] = {}
        self.start_times: Dict[str, datetime] = {}
    
    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        self.start_times[operation] = datetime.now()
        self.logger.debug(f"Started timing operation: {operation}")
    
    def end_timer(self, operation: str) -> float:
        """End timing an operation and return duration in seconds."""
        if operation not in self.start_times:
            self.logger.warning(f"No start time found for operation: {operation}")
            return 0.0
        
        duration = (datetime.now() - self.start_times[operation]).total_seconds()
        self.metrics[operation] = duration
        
        self.logger.info(f"Operation '{operation}' completed in {duration:.2f}s")
        del self.start_times[operation]
        return duration
    
    def record_metric(self, name: str, value: Any) -> None:
        """Record a custom metric."""
        self.metrics[name] = value
        self.logger.debug(f"Recorded metric {name}: {value}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all recorded metrics."""
        return self.metrics.copy()
    
    def log_system_info(self) -> None:
        """Log system information."""
        import platform
        import psutil
        
        system_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
        }
        
        self.logger.info(f"System information: {system_info}")
        self.metrics.update(system_info)


class SearchMetrics:
    """Track search-specific metrics."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("search_metrics")
        self.search_count = 0
        self.source_usage: Dict[str, int] = {}
        self.error_count: Dict[str, int] = {}
        self.response_times: Dict[str, list] = {}
    
    def record_search(self, query: str, sources: list, duration: float) -> None:
        """Record a search operation."""
        self.search_count += 1
        
        for source in sources:
            self.source_usage[source] = self.source_usage.get(source, 0) + 1
            if source not in self.response_times:
                self.response_times[source] = []
            self.response_times[source].append(duration)
        
        self.logger.info(
            f"Search recorded - Query: '{query[:50]}...', "
            f"Sources: {sources}, Duration: {duration:.2f}s"
        )
    
    def record_error(self, source: str, error_type: str) -> None:
        """Record an error for a specific source."""
        error_key = f"{source}_{error_type}"
        self.error_count[error_key] = self.error_count.get(error_key, 0) + 1
        
        self.logger.warning(f"Error recorded - Source: {source}, Type: {error_type}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of search metrics."""
        avg_response_times = {}
        for source, times in self.response_times.items():
            avg_response_times[source] = sum(times) / len(times) if times else 0
        
        return {
            "total_searches": self.search_count,
            "source_usage": self.source_usage,
            "error_count": self.error_count,
            "average_response_times": avg_response_times,
        }
    
    def log_summary(self) -> None:
        """Log a summary of metrics."""
        summary = self.get_summary()
        self.logger.info(f"Search metrics summary: {summary}")


# Global logger instances
_main_logger = None
_performance_monitor = None
_search_metrics = None


def get_logger(name: str = "search_agent", log_level: str = "INFO") -> logging.Logger:
    """Get or create a logger instance."""
    global _main_logger
    if _main_logger is None:
        _main_logger = SearchAgentLogger(name, log_level)
    return _main_logger.get_logger()


def get_struct_logger(name: str = "search_agent", log_level: str = "INFO") -> structlog.BoundLogger:
    """Get or create a structured logger instance."""
    global _main_logger
    if _main_logger is None:
        _main_logger = SearchAgentLogger(name, log_level)
    return _main_logger.get_struct_logger()


def get_performance_monitor() -> PerformanceMonitor:
    """Get or create a performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor(get_logger("performance"))
    return _performance_monitor


def get_search_metrics() -> SearchMetrics:
    """Get or create a search metrics instance."""
    global _search_metrics
    if _search_metrics is None:
        _search_metrics = SearchMetrics(get_logger("search_metrics"))
    return _search_metrics


def setup_logging(log_level: str = "INFO", name: str = "search_agent") -> None:
    """Set up logging for the entire application."""
    global _main_logger, _performance_monitor, _search_metrics
    
    _main_logger = SearchAgentLogger(name, log_level)
    _performance_monitor = PerformanceMonitor(_main_logger.get_logger())
    _search_metrics = SearchMetrics(_main_logger.get_logger())
    
    logger = _main_logger.get_logger()
    logger.info(f"Logging initialized with level: {log_level}")
    
    # Log system information
    _performance_monitor.log_system_info()
