"""Application configuration"""
import os
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    # App settings
    app_name: str = "Deep Deer"
    version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")

    # Server settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    reload: bool = Field(default=False, env="RELOAD")
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")

    # Database settings
    database_url: str = Field(
        default="sqlite+aiosqlite:///./deep_deer.db",
        env="DATABASE_URL"
    )

    # LLM settings
    llm_provider: str = Field(default="ollama", env="LLM_PROVIDER")  # ollama, huggingface_hub, huggingface_local

    # Ollama settings
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        env="OLLAMA_BASE_URL"
    )
    ollama_model: str = Field(
        default="gpt-oss:120b",
        env="OLLAMA_MODEL"
    )
    ollama_timeout: int = Field(default=300, env="OLLAMA_TIMEOUT")

    # HuggingFace settings
    huggingface_token: Optional[str] = Field(default=None, env="HUGGINGFACE_TOKEN")
    huggingface_model: str = Field(default="google/flan-t5-large", env="HUGGINGFACE_MODEL")
    huggingface_local_model: str = Field(default="gpt2", env="HUGGINGFACE_LOCAL_MODEL")

    # Vector store settings
    vector_store_path: str = Field(
        default="./vector_stores",
        env="VECTOR_STORE_PATH"
    )

    # Embedding settings
    embedding_provider: str = Field(default="ollama", env="EMBEDDING_PROVIDER")  # ollama or huggingface
    ollama_embedding_model: str = Field(
        default="jeffh/intfloat-multilingual-e5-large-instruct:f32",
        env="OLLAMA_EMBEDDING_MODEL"
    )
    embedding_model: str = Field(
        default="BAAI/bge-m3",
        env="EMBEDDING_MODEL"
    )

    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=100, env="CHUNK_OVERLAP")

    # Web search settings
    naver_client_id: Optional[str] = Field(default=None, env="NAVER_CLIENT_ID")
    naver_client_secret: Optional[str] = Field(default=None, env="NAVER_CLIENT_SECRET")
    search_api_key: Optional[str] = Field(default=None, env="SEARCH_API_KEY")
    max_search_results: int = Field(default=10, env="MAX_SEARCH_RESULTS")

    # Security settings
    secret_key: str = Field(
        default="your-secret-key-change-in-production",
        env="SECRET_KEY"
    )
    sensitive_fields: Optional[str] = Field(
        default="password,token,key,secret,credit_card,ssn,api_key",
        env="SENSITIVE_FIELDS"
    )

    @property
    def sensitive_fields_list(self) -> List[str]:
        """Get sensitive fields as a list"""
        if self.sensitive_fields:
            return [field.strip() for field in self.sensitive_fields.split(',')]
        return []

    # LangChain settings
    langchain_verbose: bool = Field(default=False, env="LANGCHAIN_VERBOSE")
    langchain_cache: bool = Field(default=True, env="LANGCHAIN_CACHE")

    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Create global settings instance
settings = Settings()