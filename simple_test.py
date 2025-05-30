#!/usr/bin/env python
"""
最小化测试脚本
"""

import asyncio
from search_agent.core.agent import SearchAgent

async def main():
    print("初始化搜索代理...")
    agent = SearchAgent()
    print("搜索代理初始化成功!")
    
    # 获取健康状态
    health = await agent.health_check()
    print(f"健康状态: {health['status']}")
    
    # 尝试一个简单的搜索
    result = await agent.quick_search("人工智能")
    print(f"快速搜索结果: {result[:100]}...")

if __name__ == "__main__":
    asyncio.run(main())
