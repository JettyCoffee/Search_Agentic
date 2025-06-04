"""Google Custom Search Engine tool."""

import aiohttp
from typing import List, Dict, Any, Optional
import asyncio
import os
import requests

from .base import BaseSearchTool
from ..exceptions.custom_exceptions import APIError, APIQuotaExceededError, APIAuthenticationError
from ..utils.config import Config, get_config


class GoogleSearchTool(BaseSearchTool):
    """Google自定义搜索引擎工具"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.cse_id = os.getenv("GOOGLE_CSE_ID")
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        
        if not self.api_key or self.api_key == "your_google_gemini_api_key_here":
            raise APIAuthenticationError("Google API key not configured")
        
        if not self.cse_id or self.cse_id == "your_google_custom_search_engine_id_here":
            raise APIAuthenticationError("Google Custom Search Engine ID not configured")
    
    @property
    def tool_name(self) -> str:
        return "Google Custom Search"
    
    @property
    def tool_description(self) -> str:
        return """
        Search the web using Google Custom Search Engine.
        Returns relevant web pages, news articles, and general information from across the internet.
        Best for finding current information, news, and general web content.
        """
    
    async def _execute_search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """执行Google自定义搜索"""
        config = get_config()
        params = {
            "key": self.api_key,
            "cx": self.cse_id,
            "q": query,
            "num": min(kwargs.get("max_results", config.agent.max_search_results), 10),  # Google CSE最多返回10个结果
            "safe": "medium",
            "fields": "items(title,link,snippet,displayLink,formattedUrl,htmlSnippet)"
        }
        
        # 可选参数
        if "date_restrict" in kwargs:
            params["dateRestrict"] = kwargs["date_restrict"]
        
        if "site_search" in kwargs:
            params["siteSearch"] = kwargs["site_search"]
        
        if "file_type" in kwargs:
            params["fileType"] = kwargs["file_type"]
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("items", [])
                    
                    elif response.status == 403:
                        error_data = await response.json()
                        error_reason = error_data.get("error", {}).get("errors", [{}])[0].get("reason", "")
                        
                        if "quotaExceeded" in error_reason or "dailyLimitExceeded" in error_reason:
                            raise APIQuotaExceededError("Google Custom Search API quota exceeded")
                        else:
                            raise APIAuthenticationError("Google Custom Search API authentication failed")
                    
                    elif response.status == 429:
                        raise APIQuotaExceededError("Google Custom Search API rate limit exceeded")
                    
                    else:
                        error_text = await response.text()
                        raise APIError(f"Google Custom Search API error: {response.status} - {error_text}")
                        
            except aiohttp.ClientError as e:
                raise APIError(f"Network error during Google search: {str(e)}")
    
    def _extract_title(self, raw_result: Dict[str, Any]) -> str:
        """提取标题"""
        return raw_result.get("title", "").strip()
    
    def _extract_url(self, raw_result: Dict[str, Any]) -> str:
        """提取URL"""
        return raw_result.get("link", "")
    
    def _extract_snippet(self, raw_result: Dict[str, Any]) -> str:
        """提取摘要"""
        # 优先使用纯文本snippet，如果没有则使用HTML snippet并清理
        snippet = raw_result.get("snippet", "")
        if not snippet:
            html_snippet = raw_result.get("htmlSnippet", "")
            # 简单的HTML标签清理
            import re
            snippet = re.sub(r'<[^>]+>', '', html_snippet)
        
        return snippet.strip()
    
    def _extract_metadata(self, raw_result: Dict[str, Any]) -> Dict[str, Any]:
        """提取Google搜索特定的元数据"""
        return {
            "display_link": raw_result.get("displayLink", ""),
            "formatted_url": raw_result.get("formattedUrl", ""),
            "has_html_snippet": bool(raw_result.get("htmlSnippet")),
            "search_engine": "Google Custom Search"
        }
    
    def _get_supported_parameters(self) -> List[str]:
        """获取支持的参数"""
        return [
            "query", 
            "max_results", 
            "date_restrict", 
            "site_search", 
            "file_type"
        ]


class GoogleImageSearchTool(GoogleSearchTool):
    """Google图片搜索工具"""
    
    @property
    def tool_name(self) -> str:
        return "Google Image Search"
    
    @property
    def tool_description(self) -> str:
        return """
        Search for images using Google Custom Search Engine.
        Returns relevant images with metadata and source information.
        """
    
    async def _execute_search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """执行Google图片搜索"""
        # 添加搜索类型参数
        kwargs["searchType"] = "image"
        return await super()._execute_search(query, **kwargs)
    
    def _extract_metadata(self, raw_result: Dict[str, Any]) -> Dict[str, Any]:
        """提取图片搜索特定的元数据"""
        metadata = super()._extract_metadata(raw_result)
        
        # 添加图片特定的元数据
        image_info = raw_result.get("image", {})
        metadata.update({
            "image_width": image_info.get("width"),
            "image_height": image_info.get("height"),
            "image_size": image_info.get("byteSize"),
            "image_format": image_info.get("format"),
            "thumbnail_url": image_info.get("thumbnailLink"),
            "context_link": image_info.get("contextLink")
        })
        
        return metadata


class GoogleSearchTool:
    """Google搜索工具"""
    
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.cse_id = os.getenv("GOOGLE_CSE_ID")
        if not self.api_key or not self.cse_id:
            raise ValueError("Missing Google API credentials")
        
        # 获取代理设置
        self.proxies = {
            "http": os.getenv("HTTP_PROXY"),
            "https": os.getenv("HTTPS_PROXY")
        }
    
    def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        执行Google搜索
        
        Args:
            query: 搜索查询
            max_results: 最大结果数量
            
        Returns:
            List[Dict[str, Any]]: 搜索结果列表
        """
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.api_key,
            "cx": self.cse_id,
            "q": query,
            "num": min(max_results, 10)  # Google CSE限制最大为10
        }
        
        response = requests.get(url, params=params, proxies=self.proxies)
        response.raise_for_status()
        data = response.json()
        
        results = []
        if "items" in data:
            for item in data["items"]:
                results.append({
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "source": "google"
                })
        
        return results
