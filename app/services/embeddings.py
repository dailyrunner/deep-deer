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
    """HuggingFace embedding provider with E5 instruction support"""

    def __init__(self, model_name: str = "BAAI/bge-m3", lazy_load: bool = False, token: str = None):
        self.model_name = model_name
        self.lazy_load = lazy_load
        self.token = token  # HuggingFace token for gated models
        self._embeddings = None

        # Check if this is an E5 instruct model
        self.is_e5_instruct = "e5" in model_name.lower() and "instruct" in model_name.lower()

        # E5-instruct specific instructions
        self.query_instruction = "query: " if self.is_e5_instruct else ""
        self.passage_instruction = "passage: " if self.is_e5_instruct else ""

    def get_embeddings(self) -> Embeddings:
        """Get LangChain HuggingFace embeddings instance"""
        if self._embeddings is None:
            # Set HuggingFace token if provided
            if self.token:
                import os
                os.environ["HF_TOKEN"] = self.token

            if self.lazy_load and self.is_e5_instruct:
                # Use lazy loading for E5 models
                self._embeddings = LazyE5InstructEmbeddings(
                    model_name=self.model_name,
                    device='cpu',
                    normalize=True,
                    token=self.token  # Pass token to lazy loader
                )
                logger.info(f"Initialized lazy E5 embeddings: {self.model_name}")
            else:
                # Regular immediate loading
                model_kwargs = {'device': 'cpu'}
                # Add token to model_kwargs for authentication
                if self.token:
                    model_kwargs['use_auth_token'] = self.token

                self._embeddings = HuggingFaceEmbeddings(
                    model_name=self.model_name,
                    model_kwargs=model_kwargs,
                    encode_kwargs={'normalize_embeddings': True}
                )
                logger.info(f"Initialized HuggingFace embeddings: {self.model_name}")
        return self._embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        if self.is_e5_instruct and not self.lazy_load:
            # Add passage instruction for E5 models (non-lazy)
            texts = [f"{self.passage_instruction}{text}" for text in texts]
        return self.get_embeddings().embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        if self.is_e5_instruct and not self.lazy_load:
            # Add query instruction for E5 models (non-lazy)
            text = f"{self.query_instruction}{text}"
        return self.get_embeddings().embed_query(text)


class LazyE5InstructEmbeddings(Embeddings):
    """Lazy-loaded E5 embeddings with instruction support"""

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large-instruct",
        device: str = "cpu",
        normalize: bool = True,
        token: str = None
    ):
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self.token = token  # HuggingFace token for gated models
        self._embedder: Optional[HuggingFaceEmbeddings] = None
        # E5-instruct specific instructions
        self.query_instruction = "query: "
        self.passage_instruction = "passage: "

    def _ensure(self) -> HuggingFaceEmbeddings:
        """Ensure embeddings are loaded"""
        if self._embedder is None:
            logger.info(f"Loading E5 model: {self.model_name}")

            model_kwargs = {'device': self.device}
            # Add token for authentication if provided
            if self.token:
                model_kwargs['use_auth_token'] = self.token

            self._embedder = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs=model_kwargs,
                encode_kwargs={'normalize_embeddings': self.normalize}
            )
            logger.info(f"E5 model loaded on {self.device}")
        return self._embedder

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents with passage instruction"""
        # Add passage instruction for E5
        instructed_texts = [f"{self.passage_instruction}{text}" for text in texts]
        return self._ensure().embed_documents(instructed_texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed query with query instruction"""
        # Add query instruction for E5
        instructed_text = f"{self.query_instruction}{text}"
        return self._ensure().embed_query(instructed_text)



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
            # HuggingFace provider (handles E5 instruct models automatically)
            model_name = kwargs.get("model_name", settings.embedding_model)
            lazy_load = kwargs.get("lazy_load", settings.enable_lazy_loading if hasattr(settings, 'enable_lazy_loading') else False)
            token = kwargs.get("token", settings.huggingface_token if hasattr(settings, 'huggingface_token') else None)

            self.provider = HuggingFaceEmbeddingProvider(
                model_name=model_name,
                lazy_load=lazy_load,
                token=token
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