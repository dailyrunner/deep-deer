"""Embedding service with Ollama and HuggingFace support"""
import logging
from typing import List, Optional, Any
from abc import ABC, abstractmethod
from enum import Enum

from langchain_community.embeddings import OllamaEmbeddings, HuggingFaceEmbeddings
from langchain.embeddings.base import Embeddings

from app.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingProvider(str, Enum):
    """Available embedding providers"""
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"


class BaseEmbeddingProvider(ABC):
    """Base class for embedding providers"""

    @abstractmethod
    def get_embeddings(self) -> Embeddings:
        """Get LangChain embeddings instance"""
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        pass


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """Ollama embedding provider"""

    def __init__(
        self,
        model: str = "jeffh/intfloat-multilingual-e5-large-instruct:f32",
        base_url: str = None
    ):
        self.model = model
        self.base_url = base_url or settings.ollama_base_url
        self._embeddings = None

    def get_embeddings(self) -> Embeddings:
        """Get LangChain Ollama embeddings instance"""
        if self._embeddings is None:
            self._embeddings = OllamaEmbeddings(
                model=self.model,
                base_url=self.base_url
            )
            logger.info(f"Initialized Ollama embeddings with model: {self.model}")
        return self._embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        return self.get_embeddings().embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self.get_embeddings().embed_query(text)


class HuggingFaceEmbeddingProvider(BaseEmbeddingProvider):
    """HuggingFace embedding provider"""

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.model_name = model_name
        self._embeddings = None

    def get_embeddings(self) -> Embeddings:
        """Get LangChain HuggingFace embeddings instance"""
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info(f"Initialized HuggingFace embeddings with model: {self.model_name}")
        return self._embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        return self.get_embeddings().embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self.get_embeddings().embed_query(text)


class EmbeddingService:
    """Unified embedding service supporting multiple providers"""

    def __init__(self):
        self.provider: Optional[BaseEmbeddingProvider] = None
        self.provider_type: Optional[EmbeddingProvider] = None
        self.is_ready = False

    def initialize(
        self,
        provider_type: Optional[EmbeddingProvider] = None,
        **kwargs
    ):
        """Initialize embedding service with specified provider"""
        # Determine provider type
        if provider_type:
            self.provider_type = provider_type
        elif settings.embedding_provider == "ollama":
            self.provider_type = EmbeddingProvider.OLLAMA
        else:
            self.provider_type = EmbeddingProvider.HUGGINGFACE

        # Create provider instance
        if self.provider_type == EmbeddingProvider.OLLAMA:
            self.provider = OllamaEmbeddingProvider(
                model=kwargs.get("model", settings.ollama_embedding_model),
                base_url=kwargs.get("base_url", settings.ollama_base_url)
            )
        else:
            self.provider = HuggingFaceEmbeddingProvider(
                model_name=kwargs.get("model_name", settings.embedding_model)
            )

        self.is_ready = True
        logger.info(f"Embedding service initialized with provider: {self.provider_type.value}")

    def get_embeddings(self) -> Embeddings:
        """Get LangChain embeddings instance"""
        if not self.is_ready or not self.provider:
            raise RuntimeError("Embedding service is not initialized")
        return self.provider.get_embeddings()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        if not self.is_ready or not self.provider:
            raise RuntimeError("Embedding service is not initialized")
        return self.provider.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        if not self.is_ready or not self.provider:
            raise RuntimeError("Embedding service is not initialized")
        return self.provider.embed_query(text)

    def cleanup(self):
        """Cleanup resources"""
        self.provider = None
        self.is_ready = False


# Global instance
embedding_service = EmbeddingService()