"""Configuration management for the Multi-Source Search Agent."""

import os
from typing import Optional, Dict, Any
from pydantic import BaseSettings, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class APIConfig(BaseSettings):
    """API配置设置"""
    
    # Google APIs
    google_api_key: str = Field(..., env="GOOGLE_API_KEY")
    google_cse_id: str = Field(..., env="GOOGLE_CSE_ID")
    
    # Brave Search
    brave_api_key: Optional[str] = Field(None, env="BRAVE_API_KEY")
    
    # Academic APIs
    semantic_scholar_api_key: Optional[str] = Field(None, env="SEMANTIC_SCHOLAR_API_KEY")
    arxiv_api_url: str = Field("http://export.arxiv.org/api/query", env="ARXIV_API_URL")
    core_api_key: Optional[str] = Field(None, env="CORE_API_KEY")


class AgentConfig(BaseSettings):
    """Agent配置设置"""
    
    # Search settings
    max_search_results: int = Field(10, env="MAX_SEARCH_RESULTS")
    search_timeout: int = Field(30, env="SEARCH_TIMEOUT")
    max_concurrent_searches: int = Field(5, env="MAX_CONCURRENT_SEARCHES")
    
    # Rate limiting
    default_rate_limit: int = Field(10, env="DEFAULT_RATE_LIMIT")
    retry_attempts: int = Field(3, env="RETRY_ATTEMPTS")
    retry_delay: float = Field(1.0, env="RETRY_DELAY")
    
    # Caching
    cache_enabled: bool = Field(True, env="CACHE_ENABLED")
    cache_ttl: int = Field(3600, env="CACHE_TTL")


class LoggingConfig(BaseSettings):
    """日志配置设置"""
    
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_format: str = Field("json", env="LOG_FORMAT")


class DevelopmentConfig(BaseSettings):
    """开发环境配置"""
    
    debug: bool = Field(False, env="DEBUG")


class Config:
    """主配置类，整合所有配置"""
    
    def __init__(self):
        self.api = APIConfig()
        self.agent = AgentConfig()
        self.logging = LoggingConfig()
        self.development = DevelopmentConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "api": self.api.dict(),
            "agent": self.agent.dict(),
            "logging": self.logging.dict(),
            "development": self.development.dict()
        }
    
    def validate_required_keys(self) -> bool:
        """验证必需的API密钥是否存在"""
        required_keys = [
            self.api.google_api_key,
            self.api.google_cse_id
        ]
        
        missing_keys = [key for key in required_keys if not key or key == "your_api_key_here"]
        
        if missing_keys:
            raise ValueError(f"Missing required API keys. Please check your .env file.")
        
        return True


# 全局配置实例
config = Config()
