#!/usr/bin/env python
"""
测试搜索代理基本功能
"""

import asyncio
import json
from search_agent import MultiSourceSearchAgent

def test_basic_search():
    """测试基本搜索功能"""
    agent = MultiSourceSearchAgent()
    query = "人工智能在医疗诊断中的应用"
    results = agent.search(query)
    
    # 打印结果以便调试
    print("搜索结果:", results)
    
    # 基本验证
    assert isinstance(results, dict)
    assert "query_info" in results
    assert "sources" in results

if __name__ == "__main__":
    test_basic_search()
