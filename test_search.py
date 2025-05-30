#!/usr/bin/env python
"""
测试搜索代理基本功能
"""

import asyncio
import json
from search_agent.core.agent import SearchAgent

async def test_search():
    """测试基本搜索功能"""
    print("初始化搜索代理...")
    agent = SearchAgent()
    
    print("执行搜索...")
    query = "人工智能在医疗领域的最新应用"
    result = await agent.search(
        query=query,
        output_format="json",
        include_raw_results=True,
        max_results_per_source=3,
        use_cache=True,
        timeout=60.0
    )
    
    print(f"搜索完成！结果包含 {len(result['search_results'])} 个来源")
    
    # 打印摘要
    print("\n摘要:")
    if "synthesis" in result and "summary" in result["synthesis"]:
        print(result["synthesis"]["summary"])
    else:
        print("未生成摘要")
    
    # 将完整结果保存到文件
    with open("search_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print("\n完整结果已保存到 search_result.json")

if __name__ == "__main__":
    asyncio.run(test_search())
