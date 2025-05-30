"""Wikipedia search tool for background context."""

import wikipedia
from typing import List, Dict, Any, Optional
import asyncio

from .base import BaseSearchTool
from ..exceptions.custom_exceptions import WikipediaError


class WikipediaSearchTool(BaseSearchTool):
    """Wikipedia搜索工具，用于获取背景信息和上下文"""
    
    def __init__(self, language: str = "en", **kwargs):
        super().__init__(**kwargs)
        self.language = language
        wikipedia.set_lang(language)
    
    @property
    def tool_name(self) -> str:
        return "Wikipedia"
    
    @property
    def tool_description(self) -> str:
        return """
        Search Wikipedia for background information and context about the query topic.
        Returns comprehensive summaries and related information to enhance query understanding.
        """
    
    async def _execute_search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """执行Wikipedia搜索"""
        try:
            # 在异步上下文中运行同步的Wikipedia API调用
            loop = asyncio.get_event_loop()
            
            # 搜索相关页面
            search_results = await loop.run_in_executor(
                None, wikipedia.search, query, kwargs.get('max_results', 5)
            )
            
            results = []
            for title in search_results:
                try:
                    # 获取页面摘要
                    summary = await loop.run_in_executor(
                        None, wikipedia.summary, title, 3  # 3句摘要
                    )
                    
                    # 获取页面对象以获取URL
                    page = await loop.run_in_executor(
                        None, wikipedia.page, title
                    )
                    
                    result = {
                        "title": title,
                        "summary": summary,
                        "url": page.url,
                        "content": summary,  # 对于标准化，使用summary作为content
                        "categories": getattr(page, 'categories', [])[:5],  # 前5个分类
                        "links": list(page.links)[:10] if hasattr(page, 'links') else [],  # 前10个链接
                        "language": self.language
                    }
                    results.append(result)
                    
                except wikipedia.exceptions.DisambiguationError as e:
                    # 处理消歧义页面，选择第一个选项
                    try:
                        first_option = e.options[0]
                        summary = await loop.run_in_executor(
                            None, wikipedia.summary, first_option, 3
                        )
                        page = await loop.run_in_executor(
                            None, wikipedia.page, first_option
                        )
                        
                        result = {
                            "title": first_option,
                            "summary": summary,
                            "url": page.url,
                            "content": summary,
                            "categories": getattr(page, 'categories', [])[:5],
                            "links": list(page.links)[:10] if hasattr(page, 'links') else [],
                            "language": self.language,
                            "disambiguation_resolved": True,
                            "original_query": title
                        }
                        results.append(result)
                    except Exception:
                        # 如果消歧义解决失败，跳过这个结果
                        continue
                        
                except wikipedia.exceptions.PageError:
                    # 页面不存在，跳过
                    continue
                except Exception as e:
                    # 其他错误，记录但继续
                    print(f"Warning: Error processing Wikipedia page '{title}': {e}")
                    continue
            
            return results
            
        except Exception as e:
            raise WikipediaError(f"Wikipedia search failed: {str(e)}")
    
    def _extract_title(self, raw_result: Dict[str, Any]) -> str:
        """提取标题"""
        return raw_result.get("title", "")
    
    def _extract_url(self, raw_result: Dict[str, Any]) -> str:
        """提取URL"""
        return raw_result.get("url", "")
    
    def _extract_snippet(self, raw_result: Dict[str, Any]) -> str:
        """提取摘要"""
        return raw_result.get("summary", "")
    
    def _extract_metadata(self, raw_result: Dict[str, Any]) -> Dict[str, Any]:
        """提取Wikipedia特定的元数据"""
        return {
            "categories": raw_result.get("categories", []),
            "related_links": raw_result.get("links", []),
            "language": raw_result.get("language", self.language),
            "disambiguation_resolved": raw_result.get("disambiguation_resolved", False),
            "content_length": len(raw_result.get("summary", ""))
        }
    
    async def get_context_summary(self, query: str, max_sentences: int = 5) -> str:
        """
        获取查询的上下文摘要，专门用于上下文增强
        
        Args:
            query: 搜索查询
            max_sentences: 最大句子数
            
        Returns:
            上下文摘要字符串
        """
        try:
            results = await self.search(query, max_results=3)
            
            if not results:
                return f"No Wikipedia context found for query: {query}"
            
            # 合并前几个结果的摘要
            context_parts = []
            total_sentences = 0
            
            for result in results:
                if total_sentences >= max_sentences:
                    break
                
                snippet = result.snippet
                sentences = snippet.split('. ')
                
                remaining_sentences = max_sentences - total_sentences
                if len(sentences) > remaining_sentences:
                    sentences = sentences[:remaining_sentences]
                
                context_parts.extend(sentences)
                total_sentences += len(sentences)
            
            context = '. '.join(context_parts)
            if context and not context.endswith('.'):
                context += '.'
            
            return context
            
        except Exception as e:
            return f"Error retrieving Wikipedia context: {str(e)}"
    
    def _get_supported_parameters(self) -> List[str]:
        """获取支持的参数"""
        return ["query", "max_results", "max_sentences"]
