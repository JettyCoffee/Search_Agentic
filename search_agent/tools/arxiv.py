"""ArXiv API tool for preprint paper search."""

import aiohttp
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
import re
from datetime import datetime

from .base import AcademicSearchTool
from ..exceptions.custom_exceptions import APIError
from ..utils.config import config


class ArXivSearchTool(AcademicSearchTool):
    """ArXiv预印本论文搜索工具"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_url = config.api.arxiv_api_url
        self.available = True  # ArXiv API是免费的
    
    @property
    def tool_name(self) -> str:
        return "ArXiv"
    
    @property
    def tool_description(self) -> str:
        return """
        Search preprint papers using ArXiv API.
        Returns research papers from physics, mathematics, computer science, and other fields.
        Best for finding the latest research and preprint papers before peer review.
        """
    
    async def _execute_search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """执行ArXiv搜索"""
        params = {
            "search_query": self._build_search_query(query, **kwargs),
            "start": kwargs.get("offset", 0),
            "max_results": min(kwargs.get("max_results", self.config.max_results), 30),  # ArXiv建议不超过30
            "sortBy": kwargs.get("sort_by", "relevance"),  # relevance, lastUpdatedDate, submittedDate
            "sortOrder": kwargs.get("sort_order", "descending")  # ascending, descending
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        xml_content = await response.text()
                        return self._parse_arxiv_xml(xml_content)
                    else:
                        error_text = await response.text()
                        raise APIError(f"ArXiv API error: {response.status} - {error_text}")
                        
            except aiohttp.ClientError as e:
                raise APIError(f"Network error during ArXiv search: {str(e)}")
    
    def _build_search_query(self, query: str, **kwargs) -> str:
        """构建ArXiv搜索查询"""
        # 基础查询
        search_parts = [f"all:{query}"]
        
        # 添加可选的搜索字段
        if "title" in kwargs:
            search_parts.append(f"ti:{kwargs['title']}")
        
        if "author" in kwargs:
            search_parts.append(f"au:{kwargs['author']}")
        
        if "abstract" in kwargs:
            search_parts.append(f"abs:{kwargs['abstract']}")
        
        if "category" in kwargs:
            search_parts.append(f"cat:{kwargs['category']}")
        
        if "journal_ref" in kwargs:
            search_parts.append(f"jr:{kwargs['journal_ref']}")
        
        # 日期范围
        if "submitted_date_from" in kwargs or "submitted_date_to" in kwargs:
            date_from = kwargs.get("submitted_date_from", "1991-01-01")
            date_to = kwargs.get("submitted_date_to", datetime.now().strftime("%Y-%m-%d"))
            search_parts.append(f"submittedDate:[{date_from} TO {date_to}]")
        
        return " AND ".join(search_parts)
    
    def _parse_arxiv_xml(self, xml_content: str) -> List[Dict[str, Any]]:
        """解析ArXiv XML响应"""
        try:
            root = ET.fromstring(xml_content)
            
            # 定义命名空间
            namespaces = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            entries = root.findall('atom:entry', namespaces)
            results = []
            
            for entry in entries:
                try:
                    result = self._parse_entry(entry, namespaces)
                    if result:
                        results.append(result)
                except Exception as e:
                    # 记录单个条目解析错误但继续处理其他条目
                    print(f"Warning: Failed to parse ArXiv entry: {e}")
                    continue
            
            return results
            
        except ET.ParseError as e:
            raise APIError(f"Failed to parse ArXiv XML response: {str(e)}")
    
    def _parse_entry(self, entry, namespaces: Dict[str, str]) -> Dict[str, Any]:
        """解析单个ArXiv条目"""
        # 提取基本信息
        title = self._get_text(entry, 'atom:title', namespaces)
        summary = self._get_text(entry, 'atom:summary', namespaces)
        
        # 获取ArXiv ID和URL
        arxiv_id = self._get_text(entry, 'atom:id', namespaces)
        if arxiv_id:
            arxiv_id = arxiv_id.split('/')[-1]  # 提取实际的ArXiv ID
        
        # 获取PDF链接
        pdf_url = None
        links = entry.findall('atom:link', namespaces)
        for link in links:
            if link.get('type') == 'application/pdf':
                pdf_url = link.get('href')
                break
        
        # 获取作者信息
        authors = []
        author_elements = entry.findall('atom:author', namespaces)
        for author in author_elements:
            name = self._get_text(author, 'atom:name', namespaces)
            if name:
                authors.append(name)
        
        # 获取发表日期
        published = self._get_text(entry, 'atom:published', namespaces)
        updated = self._get_text(entry, 'atom:updated', namespaces)
        
        # 获取分类
        categories = []
        category_elements = entry.findall('atom:category', namespaces)
        for cat in category_elements:
            term = cat.get('term')
            if term:
                categories.append(term)
        
        # 获取ArXiv特定信息
        comment = self._get_text(entry, 'arxiv:comment', namespaces)
        journal_ref = self._get_text(entry, 'arxiv:journal_ref', namespaces)
        doi = self._get_text(entry, 'arxiv:doi', namespaces)
        
        # 提取年份
        year = None
        if published:
            try:
                year = int(published.split('-')[0])
            except (ValueError, IndexError):
                pass
        
        return {
            "title": title,
            "summary": summary,
            "arxiv_id": arxiv_id,
            "pdf_url": pdf_url,
            "authors": authors,
            "published": published,
            "updated": updated,
            "categories": categories,
            "comment": comment,
            "journal_ref": journal_ref,
            "doi": doi,
            "year": year,
            "url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ""
        }
    
    def _get_text(self, element, path: str, namespaces: Dict[str, str]) -> Optional[str]:
        """安全地从XML元素获取文本"""
        try:
            elem = element.find(path, namespaces)
            if elem is not None:
                text = elem.text
                if text:
                    # 清理文本（移除多余的空白字符）
                    return ' '.join(text.split())
            return None
        except Exception:
            return None
    
    def _extract_title(self, raw_result: Dict[str, Any]) -> str:
        """提取标题"""
        return raw_result.get("title", "").strip()
    
    def _extract_url(self, raw_result: Dict[str, Any]) -> str:
        """提取URL"""
        return raw_result.get("url", "")
    
    def _extract_snippet(self, raw_result: Dict[str, Any]) -> str:
        """提取摘要"""
        summary = raw_result.get("summary", "")
        if len(summary) > 500:
            summary = summary[:497] + "..."
        return summary.strip()
    
    def _extract_authors(self, raw_result: Dict[str, Any]) -> List[str]:
        """提取作者信息"""
        return raw_result.get("authors", [])
    
    def _extract_publication_year(self, raw_result: Dict[str, Any]) -> Optional[int]:
        """提取发表年份"""
        return raw_result.get("year")
    
    def _extract_doi(self, raw_result: Dict[str, Any]) -> Optional[str]:
        """提取DOI"""
        return raw_result.get("doi")
    
    def _extract_citation_count(self, raw_result: Dict[str, Any]) -> Optional[int]:
        """提取引用数（ArXiv不提供引用数）"""
        return None
    
    def _extract_venue(self, raw_result: Dict[str, Any]) -> Optional[str]:
        """提取发表场所"""
        return raw_result.get("journal_ref")
    
    def _extract_pdf_url(self, raw_result: Dict[str, Any]) -> Optional[str]:
        """提取PDF链接"""
        return raw_result.get("pdf_url")
    
    def _extract_metadata(self, raw_result: Dict[str, Any]) -> Dict[str, Any]:
        """提取ArXiv特定的元数据"""
        metadata = super()._extract_metadata(raw_result)
        
        # 添加ArXiv特定的元数据
        metadata.update({
            "arxiv_id": raw_result.get("arxiv_id"),
            "categories": raw_result.get("categories", []),
            "primary_category": raw_result.get("categories", [None])[0],
            "published_date": raw_result.get("published"),
            "updated_date": raw_result.get("updated"),
            "comment": raw_result.get("comment"),
            "journal_reference": raw_result.get("journal_ref"),
            "search_engine": "ArXiv"
        })
        
        return metadata
    
    def _get_supported_parameters(self) -> List[str]:
        """获取支持的参数"""
        return [
            "query",
            "max_results",
            "offset",
            "title",
            "author", 
            "abstract",
            "category",
            "journal_ref",
            "submitted_date_from",
            "submitted_date_to",
            "sort_by",
            "sort_order"
        ]
