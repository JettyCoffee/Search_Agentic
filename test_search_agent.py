#!/usr/bin/env python3
"""
测试搜索代理的简单脚本。
"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入搜索代理
from search_agent import SearchAgent

async def main():
    """运行搜索测试"""
    print("🔍 启动搜索代理测试")
    print("-" * 50)
    
    # 创建搜索代理
    # 注意：SearchAgent不接受配置参数
    agent = SearchAgent()
    
    # 执行搜索
    query = "人工智能在医疗诊断中的应用"
    print(f"搜索查询: {query}")
    
    try:
        # 执行搜索
        results = await agent.search(query)
        print("\n搜索结果:")
        print(results)
    except Exception as e:
        print(f"搜索过程中发生错误: {str(e)}")
    
    print("-" * 50)
    print("测试完成")

if __name__ == "__main__":
    asyncio.run(main())
