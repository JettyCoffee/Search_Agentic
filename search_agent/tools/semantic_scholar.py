"""Semantic Scholar API tool for academic paper search."""

import aiohttp
from typing import List, Dict, Any, Optional
import asyncio

from .base import AcademicSearchTool
from ..exceptions.custom_exceptions import APIError, APIQuotaExceededError, APIAuthenticationError
from ..utils.config import Config, get_config


class SemanticScholarTool(AcademicSearchTool):
    """Semantic Scholar学术论文搜索工具"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        config = get_config()
        self.api_key = config.api.semantic_scholar_api_key  # 可选的API密钥
        self.base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        
        # Semantic Scholar有免费层级，API密钥是可选的
        self.available = True
    
    @property
    def tool_name(self) -> str:
        return "Semantic Scholar"
    
    @property
    def tool_description(self) -> str:
        return """
        Search academic papers using Semantic Scholar API.
        Returns research papers with abstracts, citation counts, author information, and publication venues.
        Best for finding peer-reviewed academic research and scientific papers.
        """
    
    async def _execute_search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """执行Semantic Scholar搜索"""
        headers = {
            "Accept": "application/json",
            "User-Agent": "Multi-Source-Search-Agent/1.0"
        }
        
        # 如果有API密钥，添加到headers
        if self.api_key:
            headers["x-api-key"] = self.api_key
        
        params = {
            "query": query,
            "limit": min(kwargs.get("max_results", self.config.max_results), 100),  # 最多100个结果
            "offset": kwargs.get("offset", 0),
            "fields": "paperId,title,abstract,authors,year,citationCount,referenceCount,publicationDate,journal,url,openAccessPdf,venue,externalIds,publicationTypes,publicationVenue"
        }
        
        # 可选过滤器
        if "year" in kwargs:
            params["year"] = kwargs["year"]
        
        if "venue" in kwargs:
            params["venue"] = kwargs["venue"]
        
        if "fields_of_study" in kwargs:
            params["fieldsOfStudy"] = kwargs["fields_of_study"]
        
        if "publication_types" in kwargs:
            params["publicationTypes"] = kwargs["publication_types"]
        
        if "min_citation_count" in kwargs:
            params["minCitationCount"] = kwargs["min_citation_count"]
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(self.base_url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("data", [])
                    
                    elif response.status == 401:
                        raise APIAuthenticationError("Semantic Scholar API authentication failed")
                    
                    elif response.status == 429:
                        raise APIQuotaExceededError("Semantic Scholar API rate limit exceeded")
                    
                    elif response.status == 400:
                        error_text = await response.text()
                        raise APIError(f"Semantic Scholar API bad request: {error_text}")
                    
                    else:
                        error_text = await response.text()
                        raise APIError(f"Semantic Scholar API error: {response.status} - {error_text}")
                        
            except aiohttp.ClientError as e:
                raise APIError(f"Network error during Semantic Scholar search: {str(e)}")
    
    def _extract_title(self, raw_result: Dict[str, Any]) -> str:
        """提取标题"""
        return raw_result.get("title", "").strip()
    
    def _extract_url(self, raw_result: Dict[str, Any]) -> str:
        """提取URL"""
        # 优先返回Semantic Scholar的页面URL
        paper_id = raw_result.get("paperId")
        if paper_id:
            return f"https://www.semanticscholar.org/paper/{paper_id}"
        return raw_result.get("url", "")
    
    def _extract_snippet(self, raw_result: Dict[str, Any]) -> str:
        """提取摘要"""
        abstract = raw_result.get("abstract", "")
        if abstract:
            # 如果摘要太长，截断到合理长度
            if len(abstract) > 500:
                abstract = abstract[:497] + "..."
        return abstract.strip()
    
    def _extract_authors(self, raw_result: Dict[str, Any]) -> List[str]:
        """提取作者信息"""
        authors = raw_result.get("authors", [])
        return [author.get("name", "") for author in authors if author.get("name")]
    
    def _extract_publication_year(self, raw_result: Dict[str, Any]) -> Optional[int]:
        """提取发表年份"""
        return raw_result.get("year")
    
    def _extract_doi(self, raw_result: Dict[str, Any]) -> Optional[str]:
        """提取DOI"""
        external_ids = raw_result.get("externalIds", {})
        return external_ids.get("DOI")
    
    def _extract_citation_count(self, raw_result: Dict[str, Any]) -> Optional[int]:
        """提取引用数"""
        return raw_result.get("citationCount")
    
    def _extract_venue(self, raw_result: Dict[str, Any]) -> Optional[str]:
        """提取发表场所"""
        # 优先使用venue字段，然后是journal
        venue = raw_result.get("venue")
        if venue:
            return venue
        
        journal = raw_result.get("journal")
        if journal and isinstance(journal, dict):
            return journal.get("name")
        elif isinstance(journal, str):
            return journal
        
        return None
    
    def _extract_pdf_url(self, raw_result: Dict[str, Any]) -> Optional[str]:
        """提取PDF链接"""
        open_access_pdf = raw_result.get("openAccessPdf")
        if open_access_pdf and isinstance(open_access_pdf, dict):
            return open_access_pdf.get("url")
        return None
    
    def _extract_metadata(self, raw_result: Dict[str, Any]) -> Dict[str, Any]:
        """提取Semantic Scholar特定的元数据"""
        metadata = super()._extract_metadata(raw_result)
        
        # 添加Semantic Scholar特定的元数据
        metadata.update({
            "paper_id": raw_result.get("paperId"),
            "reference_count": raw_result.get("referenceCount"),
            "publication_date": raw_result.get("publicationDate"),
            "publication_types": raw_result.get("publicationTypes", []),
            "fields_of_study": raw_result.get("fieldsOfStudy", []),
            "is_open_access": bool(raw_result.get("openAccessPdf")),
            "external_ids": raw_result.get("externalIds", {}),
            "publication_venue": raw_result.get("publicationVenue", {}),
            "search_engine": "Semantic Scholar"
        })
        
        return metadata
    
    def _get_supported_parameters(self) -> List[str]:
        """获取支持的参数"""
        return [
            "query", 
            "max_results", 
            "offset", 
            "year", 
            "venue", 
            "fields_of_study", 
            "publication_types", 
            "min_citation_count"
        ]
    
    async def get_paper_details(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """
        获取特定论文的详细信息
        
        Args:
            paper_id: Semantic Scholar论文ID
            
        Returns:
            论文详细信息
        """
        if not paper_id:
            return None
        
        url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
        headers = {
            "Accept": "application/json",
            "User-Agent": "Multi-Source-Search-Agent/1.0"
        }
        
        if self.api_key:
            headers["x-api-key"] = self.api_key
        
        params = {
            "fields": "paperId,title,abstract,authors,year,citationCount,referenceCount,publicationDate,journal,url,openAccessPdf,venue,externalIds,publicationTypes,publicationVenue,citations,references"
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return None
            except Exception:
                return None
