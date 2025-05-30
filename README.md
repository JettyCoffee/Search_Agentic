# 多源智能搜索代理 (Multi-Source Intelligent Search Agent)

## 项目简介

这是一个基于AI的智能搜索代理项目，能够同时利用多个搜索引擎和学术数据库进行信息检索。该代理使用LangChain和LangGraph框架，集成Google Gemini API，为用户提供全面、结构化的搜索结果。

## 核心特性

- 🔍 **多源搜索**: 集成Wikipedia、Google、Brave、Semantic Scholar、ArXiv、CORE等多个数据源
- 🧠 **智能查询优化**: 使用LLM分析用户意图并为不同平台优化查询
- 📚 **上下文增强**: 利用Wikipedia背景知识提升搜索精度
- 🔄 **并行处理**: 同时执行多个搜索任务，提高效率
- 📊 **结构化输出**: 按数据源分类的JSON格式结果
- 🛡️ **容错设计**: 强大的错误处理和降级策略

## 技术栈

- **AI框架**: LangChain, LangGraph
- **语言模型**: Google Gemini API
- **编程语言**: Python
- **数据格式**: JSON
- **API集成**: 多种搜索引擎和学术数据库

## 支持的数据源

### 通用搜索
- 🌐 **Google Custom Search**: 免费层级网页搜索
- 🦁 **Brave Search**: 隐私友好的搜索引擎

### 学术搜索
- 📖 **Wikipedia**: 背景知识和上下文信息
- 🎓 **Semantic Scholar**: 学术论文和引用数据
- 📄 **ArXiv**: 预印本科学论文
- 🔬 **CORE**: 开放获取学术文献

## 项目文档

- 📋 [详细设计文档](PROJECT_DESIGN.md): 完整的项目架构和实施计划
- 🏗️ [架构设计](PROJECT_DESIGN.md#系统架构设计): 系统组件和数据流
- 🔧 [工具设计](PROJECT_DESIGN.md#工具设计规范): API集成和工具接口
- 📊 [输出格式](PROJECT_DESIGN.md#输出格式规范): JSON结果结构

## 快速开始

### 先决条件

- Python 3.8+
- 有效的API密钥：
  - Google Custom Search API
  - Google Gemini API
  - Brave Search API（可选）
  - 其他学术API密钥

### 安装步骤

1. **克隆仓库**
   ```bash
   git clone <repository-url>
   cd Search_Agentic
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **配置环境**
   ```bash
   cp .env.example .env
   # 编辑 .env 文件，添加你的API密钥
   ```

4. **运行代理**
   ```bash
   python main.py
   ```

## 使用示例

### 基本查询
```python
from search_agent import MultiSourceSearchAgent

agent = MultiSourceSearchAgent()
results = agent.search("人工智能在医疗诊断中的应用")
print(results)
```

### 输出示例
```json
{
  "query_info": {
    "original_query": "人工智能在医疗诊断中的应用",
    "execution_timestamp": "2025-05-30T10:00:00Z",
    "total_execution_time": 2.5,
    "status": "success"
  },
  "sources": {
    "wikipedia": {
      "source_name": "Wikipedia",
      "query_used": "artificial intelligence medical diagnosis",
      "result_count": 3,
      "results": [...]
    },
    "semantic_scholar": {
      "source_name": "Semantic Scholar",
      "query_used": "AI medical diagnosis applications",
      "result_count": 10,
      "results": [...]
    }
  }
}
```

## 项目状态

🚧 **开发中** - 当前处于设计和规划阶段

### 开发计划

- [x] 项目设计和架构规划
- [ ] 基础框架搭建
- [ ] 核心搜索工具实现
- [ ] LLM集成和查询优化
- [ ] 测试和文档完善

## 贡献指南

欢迎贡献！请遵循以下步骤：

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 联系方式

- 项目作者: [Your Name]
- 邮箱: [your.email@example.com]
- 项目链接: [https://github.com/yourusername/Search_Agentic](https://github.com/yourusername/Search_Agentic)

## 致谢

- LangChain 和 LangGraph 社区
- 所有提供免费API的搜索引擎和数据库提供商
- 开源社区的支持和贡献

---

📖 **更多详细信息请查看 [项目设计文档](PROJECT_DESIGN.md)**