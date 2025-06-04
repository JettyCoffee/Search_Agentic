"""Brave Search API tool."""

import aiohttp
from typing import List, Dict, Any, Optional
import asyncio
import os
import requests

from .base import BaseSearchTool
from ..exceptions.custom_exceptions import APIError, APIQuotaExceededError, APIAuthenticationError
from ..utils.config import Config, get_config


class BraveSearchTool(BaseSearchTool):
    """Brave搜索API工具"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 从配置工具获取配置
        self.config = get_config()
        self.api_key = self.config.api.brave_api_key
        self.base_url = "https://api.search.brave.com/res/v1/web/search"
        
        if not self.api_key:
            # Brave搜索是可选的，如果没有API密钥则不可用
            self.available = False
        else:
            self.available = True
    
    @property
    def tool_name(self) -> str:
        return "Brave Search"
    
    @property
    def tool_description(self) -> str:
        return """
        Search the web using Brave Search API.
        Privacy-focused search engine that returns relevant web content.
        Good alternative to Google search for general web information.
        """
    
    async def _execute_search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """执行Brave搜索"""
        if not self.available:
            raise APIAuthenticationError("Brave Search API key not configured")
        
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key
        }
        
        params = {
            "q": query,
            "count": min(kwargs.get("max_results", self.config.max_results), 20),  # Brave最多返回20个结果
            "offset": kwargs.get("offset", 0),
            "safesearch": kwargs.get("safesearch", "moderate"),
            "freshness": kwargs.get("freshness", ""),  # pd, pw, pm, py (past day, week, month, year)
            "text_decorations": False,
            "spellcheck": True
        }
        
        # 移除空值参数
        params = {k: v for k, v in params.items() if v != ""}
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(self.base_url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("web", {}).get("results", [])
                    
                    elif response.status == 401:
                        raise APIAuthenticationError("Brave Search API authentication failed")
                    
                    elif response.status == 429:
                        raise APIQuotaExceededError("Brave Search API rate limit exceeded")
                    
                    elif response.status == 402:
                        raise APIQuotaExceededError("Brave Search API quota exceeded")
                    
                    else:
                        error_text = await response.text()
                        raise APIError(f"Brave Search API error: {response.status} - {error_text}")
                        
            except aiohttp.ClientError as e:
                raise APIError(f"Network error during Brave search: {str(e)}")
    
    def _extract_title(self, raw_result: Dict[str, Any]) -> str:
        """提取标题"""
        return raw_result.get("title", "").strip()
    
    def _extract_url(self, raw_result: Dict[str, Any]) -> str:
        """提取URL"""
        return raw_result.get("url", "")
    
    def _extract_snippet(self, raw_result: Dict[str, Any]) -> str:
        """提取摘要"""
        return raw_result.get("description", "").strip()
    
    def _extract_metadata(self, raw_result: Dict[str, Any]) -> Dict[str, Any]:
        """提取Brave搜索特定的元数据"""
        return {
            "age": raw_result.get("age"),  # 页面年龄
            "language": raw_result.get("language"),
            "family_friendly": raw_result.get("family_friendly"),
            "type": raw_result.get("type"),  # search_result类型
            "subtype": raw_result.get("subtype"),
            "deep_results": raw_result.get("deep_results", {}),
            "profile": raw_result.get("profile", {}),
            "search_engine": "Brave Search"
        }
    
    def _get_supported_parameters(self) -> List[str]:
        """获取支持的参数"""
        return [
            "query", 
            "max_results", 
            "offset", 
            "safesearch", 
            "freshness"
        ]


class BraveNewsSearchTool(BraveSearchTool):
    """Brave新闻搜索工具"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_url = "https://api.search.brave.com/res/v1/news/search"
    
    @property
    def tool_name(self) -> str:
        return "Brave News Search"
    
    @property
    def tool_description(self) -> str:
        return """
        Search for news articles using Brave Search API.
        Returns recent news articles and current information.
        Good for finding up-to-date news and current events.
        """
    
    async def _execute_search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """执行Brave新闻搜索"""
        if not self.available:
            raise APIAuthenticationError("Brave Search API key not configured")
        
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key
        }
        
        params = {
            "q": query,
            "count": min(kwargs.get("max_results", self.config.max_results), 20),
            "offset": kwargs.get("offset", 0),
            "safesearch": kwargs.get("safesearch", "moderate"),
            "freshness": kwargs.get("freshness", "pd"),  # 默认过去一天的新闻
            "text_decorations": False,
            "spellcheck": True
        }
        
        # 移除空值参数
        params = {k: v for k, v in params.items() if v != ""}
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(self.base_url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("results", [])
                    
                    elif response.status == 401:
                        raise APIAuthenticationError("Brave News Search API authentication failed")
                    
                    elif response.status == 429:
                        raise APIQuotaExceededError("Brave News Search API rate limit exceeded")
                    
                    elif response.status == 402:
                        raise APIQuotaExceededError("Brave News Search API quota exceeded")
                    
                    else:
                        error_text = await response.text()
                        raise APIError(f"Brave News Search API error: {response.status} - {error_text}")
                        
            except aiohttp.ClientError as e:
                raise APIError(f"Network error during Brave news search: {str(e)}")
    
    def _extract_metadata(self, raw_result: Dict[str, Any]) -> Dict[str, Any]:
        """提取新闻特定的元数据"""
        metadata = super()._extract_metadata(raw_result)
        
        # 添加新闻特定的元数据
        metadata.update({
            "published_date": raw_result.get("age"),
            "source": raw_result.get("meta_url", {}).get("netloc"),
            "category": raw_result.get("category"),
            "breaking_news": raw_result.get("breaking", False),
            "search_engine": "Brave News Search"
        })
        
        return metadata


class BraveSearchTool:
    """Brave搜索工具"""
    
    def __init__(self):
        self.api_key = os.getenv("BRAVE_API_KEY")
        if not self.api_key:
            raise ValueError("Missing Brave API key")
            
        # 获取代理设置
        self.proxies = {
            "http": os.getenv("HTTP_PROXY"),
            "https": os.getenv("HTTPS_PROXY")
        }
    
    def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        执行Brave搜索
        
        Args:
            query: 搜索查询
            max_results: 最大结果数量
            
        Returns:
            List[Dict[str, Any]]: 搜索结果列表
        """
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key
        }
        params = {
            "q": query,
            "count": max_results
        }
        
        response = requests.get(url, headers=headers, params=params, proxies=self.proxies)
        response.raise_for_status()
        data = response.json()
        
        results = []
        if "web" in data and "results" in data["web"]:
            for item in data["web"]["results"]:
                results.append({
                    "title": item.get("title", ""),
                    "link": item.get("url", ""),
                    "snippet": item.get("description", ""),
                    "source": "brave"
                })
        
        return results
