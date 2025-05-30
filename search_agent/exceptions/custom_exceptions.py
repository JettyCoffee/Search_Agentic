"""Custom exceptions for the Multi-Source Search Agent."""

from typing import Optional, Dict, Any


class SearchAgentError(Exception):
    """Base exception for all search agent errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class APIError(SearchAgentError):
    """API相关错误"""
    pass


class APIQuotaExceededError(APIError):
    """API配额超限错误"""
    pass


class APIAuthenticationError(APIError):
    """API认证错误"""
    pass


class APITimeoutError(APIError):
    """API超时错误"""
    pass


class APIRateLimitError(APIError):
    """API速率限制错误"""
    pass


class DataValidationError(SearchAgentError):
    """数据验证错误"""
    pass


class QueryProcessingError(SearchAgentError):
    """查询处理错误"""
    pass


class WikipediaError(SearchAgentError):
    """Wikipedia搜索错误"""
    pass


class LLMError(SearchAgentError):
    """LLM处理错误"""
    pass


class LLMConfigError(LLMError):
    """LLM配置错误"""
    pass


class LLMAPIError(LLMError):
    """LLM API错误"""
    pass


class QueryOptimizationError(LLMError):
    """查询优化错误"""
    pass


class WorkflowError(SearchAgentError):
    """工作流错误"""
    pass


class SearchToolError(SearchAgentError):
    """搜索工具错误"""
    
    def __init__(self, tool_name: str, message: str, **kwargs):
        self.tool_name = tool_name
        super().__init__(f"[{tool_name}] {message}", **kwargs)


class OutputFormattingError(SearchAgentError):
    """输出格式化错误"""
    pass


class ConfigurationError(SearchAgentError):
    """配置错误"""
    pass
