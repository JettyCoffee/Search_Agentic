"""CORE API tool for open access academic literature."""

import aiohttp
from typing import List, Dict, Any, Optional
import asyncio

from .base import AcademicSearchTool
from ..exceptions.custom_exceptions import APIError, APIQuotaExceededError, APIAuthenticationError
from ..utils.config import config


class CORESearchTool(AcademicSearchTool):
    """CORE开放获取学术文献搜索工具"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = config.api.core_api_key
        self.base_url = "https://api.core.ac.uk/v3/search/works"
        
        if not self.api_key:
            # CORE API需要API密钥
            self.available = False
        else:
            self.available = True
    
    @property
    def tool_name(self) -> str:
        return "CORE"
    
    @property
    def tool_description(self) -> str:
        return """
        Search open access academic literature using CORE API.
        Returns research papers with full-text availability and repository information.
        Best for finding openly accessible academic papers and theses.
        """
    
    async def _execute_search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """执行CORE搜索"""
        if not self.available:
            raise APIAuthenticationError("CORE API key not configured")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        
        # 构建搜索请求体
        request_body = {
            "q": query,
            "limit": min(kwargs.get("max_results", self.config.max_results), 100),
            "offset": kwargs.get("offset", 0),
            "sort": kwargs.get("sort", "relevance"),  # relevance, recency, citation_count
            "exclude_deleted": True,
            "exclude_withdrawn": True
        }
        
        # 可选过滤器
        filters = {}
        
        if "year_from" in kwargs or "year_to" in kwargs:
            year_filter = {}
            if "year_from" in kwargs:
                year_filter["gte"] = kwargs["year_from"]
            if "year_to" in kwargs:
                year_filter["lte"] = kwargs["year_to"]
            filters["year"] = year_filter
        
        if "repository" in kwargs:
            filters["repository.name"] = kwargs["repository"]
        
        if "language" in kwargs:
            filters["language.code"] = kwargs["language"]
        
        if "subject" in kwargs:
            filters["subjects"] = kwargs["subject"]
        
        if "has_fulltext" in kwargs:
            filters["fulltext"] = kwargs["has_fulltext"]
        
        if filters:
            request_body["filters"] = filters
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(self.base_url, headers=headers, json=request_body) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("results", [])
                    
                    elif response.status == 401:
                        raise APIAuthenticationError("CORE API authentication failed")
                    
                    elif response.status == 429:
                        raise APIQuotaExceededError("CORE API rate limit exceeded")
                    
                    elif response.status == 403:
                        raise APIQuotaExceededError("CORE API quota exceeded")
                    
                    else:
                        error_text = await response.text()
                        raise APIError(f"CORE API error: {response.status} - {error_text}")
                        
            except aiohttp.ClientError as e:
                raise APIError(f"Network error during CORE search: {str(e)}")
    
    def _extract_title(self, raw_result: Dict[str, Any]) -> str:
        """提取标题"""
        return raw_result.get("title", "").strip()
    
    def _extract_url(self, raw_result: Dict[str, Any]) -> str:
        """提取URL"""
        # 优先使用CORE的链接
        core_id = raw_result.get("id")
        if core_id:
            return f"https://core.ac.uk/works/{core_id}"
        
        # 备选：使用原始URL
        return raw_result.get("sourceFulltextUrls", [""])[0] or raw_result.get("urls", [""])[0]
    
    def _extract_snippet(self, raw_result: Dict[str, Any]) -> str:
        """提取摘要"""
        abstract = raw_result.get("abstract", "")
        if abstract and len(abstract) > 500:
            abstract = abstract[:497] + "..."
        return abstract.strip()
    
    def _extract_authors(self, raw_result: Dict[str, Any]) -> List[str]:
        """提取作者信息"""
        authors = raw_result.get("authors", [])
        author_names = []
        
        for author in authors:
            if isinstance(author, dict):
                name = author.get("name", "")
            elif isinstance(author, str):
                name = author
            else:
                continue
            
            if name:
                author_names.append(name)
        
        return author_names
    
    def _extract_publication_year(self, raw_result: Dict[str, Any]) -> Optional[int]:
        """提取发表年份"""
        year = raw_result.get("yearPublished")
        if year:
            try:
                return int(year)
            except (ValueError, TypeError):
                pass
        return None
    
    def _extract_doi(self, raw_result: Dict[str, Any]) -> Optional[str]:
        """提取DOI"""
        identifiers = raw_result.get("identifiers", [])
        for identifier in identifiers:
            if isinstance(identifier, dict) and identifier.get("type") == "doi":
                return identifier.get("identifier")
        return raw_result.get("doi")
    
    def _extract_citation_count(self, raw_result: Dict[str, Any]) -> Optional[int]:
        """提取引用数"""
        return raw_result.get("citationCount")
    
    def _extract_venue(self, raw_result: Dict[str, Any]) -> Optional[str]:
        """提取发表场所"""
        journal = raw_result.get("journal")
        if journal:
            if isinstance(journal, dict):
                return journal.get("title") or journal.get("name")
            elif isinstance(journal, str):
                return journal
        
        # 备选：使用publisher信息
        publisher = raw_result.get("publisher")
        if publisher:
            return publisher
        
        return None
    
    def _extract_pdf_url(self, raw_result: Dict[str, Any]) -> Optional[str]:
        """提取PDF链接"""
        # 检查全文URL
        fulltext_urls = raw_result.get("sourceFulltextUrls", [])
        for url in fulltext_urls:
            if url and url.lower().endswith('.pdf'):
                return url
        
        # 检查下载URL
        download_url = raw_result.get("downloadUrl")
        if download_url:
            return download_url
        
        # 检查其他URL
        urls = raw_result.get("urls", [])
        for url in urls:
            if url and url.lower().endswith('.pdf'):
                return url
        
        return None
    
    def _extract_metadata(self, raw_result: Dict[str, Any]) -> Dict[str, Any]:
        """提取CORE特定的元数据"""
        metadata = super()._extract_metadata(raw_result)
        
        # 添加CORE特定的元数据
        repository_info = raw_result.get("repositories", [{}])[0] if raw_result.get("repositories") else {}
        
        metadata.update({
            "core_id": raw_result.get("id"),
            "repository_name": repository_info.get("name"),
            "repository_id": repository_info.get("id"),
            "language": raw_result.get("language", {}).get("name"),
            "language_code": raw_result.get("language", {}).get("code"),
            "subjects": raw_result.get("subjects", []),
            "document_type": raw_result.get("documentType"),
            "has_fulltext": bool(raw_result.get("fullText")),
            "fulltext_identifier": raw_result.get("fulltextIdentifier"),
            "oai_identifier": raw_result.get("oai"),
            "mag_id": raw_result.get("magId"),
            "arxiv_id": raw_result.get("arxivId"),
            "date_published": raw_result.get("datePublished"),
            "search_engine": "CORE"
        })
        
        return metadata
    
    def _get_supported_parameters(self) -> List[str]:
        """获取支持的参数"""
        return [
            "query",
            "max_results",
            "offset",
            "sort",
            "year_from",
            "year_to", 
            "repository",
            "language",
            "subject",
            "has_fulltext"
        ]
    
    async def get_fulltext(self, core_id: str) -> Optional[str]:
        """
        获取论文的全文内容
        
        Args:
            core_id: CORE文档ID
            
        Returns:
            全文内容（如果可用）
        """
        if not self.available or not core_id:
            return None
        
        url = f"https://api.core.ac.uk/v3/works/{core_id}/fulltext"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json"
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("fullText")
                    else:
                        return None
            except Exception:
                return None
