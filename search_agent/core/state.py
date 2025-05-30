"""State management for the Multi-Source Search Agent using LangGraph."""

from typing import Dict, List, Optional, Any, TypedDict
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class ExecutionStatus(str, Enum):
    """执行状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL_SUCCESS = "partial_success"


class SearchResult(BaseModel):
    """标准化搜索结果格式"""
    title: str
    url: str
    snippet: str
    source: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SourceResult(BaseModel):
    """单个数据源的搜索结果"""
    source_name: str
    query_used: str
    status: ExecutionStatus
    result_count: int
    results: List[SearchResult]
    execution_time: float
    error: Optional[str] = None


class AgentState(TypedDict):
    """代理状态定义，用于LangGraph状态管理"""
    
    # 基础信息
    original_query: str
    execution_timestamp: str
    
    # Wikipedia上下文
    wikipedia_summary: str
    wikipedia_error: Optional[str]
    
    # 优化查询
    refined_queries: Dict[str, str]  # {source_name: refined_query}
    
    # 原始搜索结果
    raw_search_results: Dict[str, List[Dict[str, Any]]]  # {source_name: raw_results}
    
    # 处理后的结果
    processed_results: Dict[str, SourceResult]  # {source_name: SourceResult}
    
    # 错误信息
    errors: Dict[str, str]  # {source_name: error_message}
    
    # 执行元数据
    execution_metadata: Dict[str, Any]
    
    # 最终状态
    final_status: ExecutionStatus
    total_execution_time: float


class StateManager:
    """状态管理器"""
    
    @staticmethod
    def create_initial_state(query: str) -> AgentState:
        """创建初始状态"""
        timestamp = datetime.utcnow().isoformat() + "Z"
        
        return AgentState(
            original_query=query,
            execution_timestamp=timestamp,
            wikipedia_summary="",
            wikipedia_error=None,
            refined_queries={},
            raw_search_results={},
            processed_results={},
            errors={},
            execution_metadata={
                "start_time": datetime.utcnow().timestamp(),
                "nodes_executed": [],
                "current_node": None
            },
            final_status=ExecutionStatus.PENDING,
            total_execution_time=0.0
        )
    
    @staticmethod
    def update_execution_metadata(state: AgentState, node_name: str, **kwargs) -> AgentState:
        """更新执行元数据"""
        state["execution_metadata"]["current_node"] = node_name
        if node_name not in state["execution_metadata"]["nodes_executed"]:
            state["execution_metadata"]["nodes_executed"].append(node_name)
        
        # 更新其他元数据
        for key, value in kwargs.items():
            state["execution_metadata"][key] = value
        
        return state
    
    @staticmethod
    def finalize_state(state: AgentState) -> AgentState:
        """最终化状态"""
        start_time = state["execution_metadata"]["start_time"]
        end_time = datetime.utcnow().timestamp()
        state["total_execution_time"] = end_time - start_time
        
        # 确定最终状态
        if state["errors"]:
            if state["processed_results"]:
                state["final_status"] = ExecutionStatus.PARTIAL_SUCCESS
            else:
                state["final_status"] = ExecutionStatus.FAILED
        else:
            state["final_status"] = ExecutionStatus.SUCCESS
        
        return state
    
    @staticmethod
    def get_successful_sources(state: AgentState) -> List[str]:
        """获取成功的数据源列表"""
        return [
            source for source, result in state["processed_results"].items()
            if result.status == ExecutionStatus.SUCCESS
        ]
    
    @staticmethod
    def get_failed_sources(state: AgentState) -> List[str]:
        """获取失败的数据源列表"""
        return [
            source for source, result in state["processed_results"].items()
            if result.status == ExecutionStatus.FAILED
        ]
