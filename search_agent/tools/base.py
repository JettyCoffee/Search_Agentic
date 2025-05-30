"""Base class for all search tools."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import time
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

from ..core.state import SearchResult, ExecutionStatus
from ..exceptions.custom_exceptions import SearchToolError, APITimeoutError
from ..utils.config import config


class ToolConfig(BaseModel):
    """工具配置基类"""
    max_results: int = Field(default=10)
    timeout: int = Field(default=30)
    rate_limit: int = Field(default=10)
    retry_attempts: int = Field(default=3)


class BaseSearchTool(ABC):
    """搜索工具基类"""
    
    def __init__(self, tool_config: Optional[ToolConfig] = None):
        self.config = tool_config or ToolConfig()
        self.name = self.__class__.__name__
        self._last_request_time = 0
        
    @property
    @abstractmethod
    def tool_name(self) -> str:
        """工具名称"""
        pass
    
    @property
    @abstractmethod
    def tool_description(self) -> str:
        """工具描述，用于LLM理解"""
        pass
    
    @abstractmethod
    async def _execute_search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """执行搜索的核心方法，需要子类实现"""
        pass
    
    def _apply_rate_limit(self):
        """应用速率限制"""
        current_time = time.time()
        time_since_last_request = current_time - self._last_request_time
        min_interval = 1.0 / self.config.rate_limit
        
        if time_since_last_request < min_interval:
            sleep_time = min_interval - time_since_last_request
            time.sleep(sleep_time)
        
        self._last_request_time = time.time()
    
    def _standardize_result(self, raw_result: Dict[str, Any]) -> SearchResult:
        """标准化单个搜索结果"""
        return SearchResult(
            title=self._extract_title(raw_result),
            url=self._extract_url(raw_result),
            snippet=self._extract_snippet(raw_result),
            source=self.tool_name,
            metadata=self._extract_metadata(raw_result)
        )
    
    @abstractmethod
    def _extract_title(self, raw_result: Dict[str, Any]) -> str:
        """从原始结果中提取标题"""
        pass
    
    @abstractmethod
    def _extract_url(self, raw_result: Dict[str, Any]) -> str:
        """从原始结果中提取URL"""
        pass
    
    @abstractmethod
    def _extract_snippet(self, raw_result: Dict[str, Any]) -> str:
        """从原始结果中提取摘要"""
        pass
    
    def _extract_metadata(self, raw_result: Dict[str, Any]) -> Dict[str, Any]:
        """从原始结果中提取元数据，默认实现"""
        return {}
    
    def _validate_query(self, query: str) -> bool:
        """验证查询字符串"""
        if not query or not query.strip():
            raise SearchToolError(self.tool_name, "Query cannot be empty")
        
        if len(query) > 1000:  # 设置最大长度限制
            raise SearchToolError(self.tool_name, "Query too long (max 1000 characters)")
        
        return True
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def search(self, query: str, **kwargs) -> List[SearchResult]:
        """
        执行搜索并返回标准化结果
        
        Args:
            query: 搜索查询字符串
            **kwargs: 额外参数
            
        Returns:
            标准化的搜索结果列表
            
        Raises:
            SearchToolError: 搜索执行错误
        """
        # 验证查询
        self._validate_query(query)
        
        # 应用速率限制
        self._apply_rate_limit()
        
        try:
            # 设置超时
            raw_results = await asyncio.wait_for(
                self._execute_search(query, **kwargs),
                timeout=self.config.timeout
            )
            
            # 标准化结果
            standardized_results = []
            for raw_result in raw_results[:self.config.max_results]:
                try:
                    standardized_result = self._standardize_result(raw_result)
                    standardized_results.append(standardized_result)
                except Exception as e:
                    # 记录单个结果处理错误但继续处理其他结果
                    print(f"Warning: Failed to standardize result from {self.tool_name}: {e}")
            
            return standardized_results
            
        except asyncio.TimeoutError:
            raise APITimeoutError(f"Search timeout after {self.config.timeout} seconds")
        except Exception as e:
            raise SearchToolError(self.tool_name, f"Search failed: {str(e)}")
    
    def get_tool_info(self) -> Dict[str, Any]:
        """获取工具信息"""
        return {
            "name": self.tool_name,
            "description": self.tool_description,
            "config": self.config.dict(),
            "supported_parameters": self._get_supported_parameters()
        }
    
    def _get_supported_parameters(self) -> List[str]:
        """获取支持的参数列表，子类可重写"""
        return ["query", "max_results"]


class AcademicSearchTool(BaseSearchTool):
    """学术搜索工具基类"""
    
    def _extract_metadata(self, raw_result: Dict[str, Any]) -> Dict[str, Any]:
        """学术搜索的元数据提取"""
        metadata = super()._extract_metadata(raw_result)
        
        # 添加学术特定的元数据
        metadata.update({
            "authors": self._extract_authors(raw_result),
            "publication_year": self._extract_publication_year(raw_result),
            "doi": self._extract_doi(raw_result),
            "citation_count": self._extract_citation_count(raw_result),
            "venue": self._extract_venue(raw_result),
            "pdf_url": self._extract_pdf_url(raw_result)
        })
        
        return metadata
    
    def _extract_authors(self, raw_result: Dict[str, Any]) -> List[str]:
        """提取作者信息，子类需要实现"""
        return []
    
    def _extract_publication_year(self, raw_result: Dict[str, Any]) -> Optional[int]:
        """提取发表年份，子类需要实现"""
        return None
    
    def _extract_doi(self, raw_result: Dict[str, Any]) -> Optional[str]:
        """提取DOI，子类需要实现"""
        return None
    
    def _extract_citation_count(self, raw_result: Dict[str, Any]) -> Optional[int]:
        """提取引用数，子类需要实现"""
        return None
    
    def _extract_venue(self, raw_result: Dict[str, Any]) -> Optional[str]:
        """提取发表场所，子类需要实现"""
        return None
    
    def _extract_pdf_url(self, raw_result: Dict[str, Any]) -> Optional[str]:
        """提取PDF链接，子类需要实现"""
        return None
