"""LLM Provider abstraction for multiple model sources"""
import logging
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod
from enum import Enum

from langchain_community.llms import Ollama, HuggingFaceHub
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from app.core.config import settings

logger = logging.getLogger(__name__)


class ModelProvider(str, Enum):
    """Available model providers"""
    OLLAMA = "ollama"
    HUGGINGFACE_HUB = "huggingface_hub"
    HUGGINGFACE_LOCAL = "huggingface_local"


class BaseLLMProvider(ABC):
    """Base class for LLM providers"""

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the provider"""
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> str:
        """Generate text from prompt"""
        pass

    @abstractmethod
    async def cleanup(self):
        """Cleanup resources"""
        pass


class OllamaProvider(BaseLLMProvider):
    """Ollama LLM provider using LangChain"""

    def __init__(self, model_name: str = None, base_url: str = None):
        self.model_name = model_name or settings.ollama_model
        self.base_url = base_url or settings.ollama_base_url
        self.llm = None
        self.is_ready = False

    async def initialize(self) -> bool:
        """Initialize Ollama provider"""
        try:
            self.llm = Ollama(
                base_url=self.base_url,
                model=self.model_name,
                temperature=0.7,
                num_predict=2000,
                timeout=settings.ollama_timeout
            )

            # Test the model
            try:
                _ = self.llm.invoke("test")
                logger.info(f"Ollama provider initialized with model: {self.model_name}")
                self.is_ready = True
                return True
            except Exception as e:
                logger.warning(f"Ollama model test failed: {e}")
                # Continue anyway, model might need pulling
                self.is_ready = True
                return True

        except Exception as e:
            logger.error(f"Failed to initialize Ollama provider: {e}")
            self.is_ready = False
            return False

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> str:
        """Generate text using Ollama"""
        if not self.is_ready:
            raise RuntimeError("Ollama provider is not ready")

        # Update LLM parameters
        self.llm.temperature = temperature
        self.llm.num_predict = max_tokens

        try:
            response = self.llm.invoke(prompt)
            return response
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            raise

    async def cleanup(self):
        """Cleanup Ollama resources"""
        self.llm = None
        self.is_ready = False


class HuggingFaceHubProvider(BaseLLMProvider):
    """HuggingFace Hub API provider"""

    def __init__(self, model_name: str = None, api_token: str = None):
        self.model_name = model_name or "gpt2"
        self.api_token = api_token or settings.huggingface_token
        self.llm = None
        self.is_ready = False

    async def initialize(self) -> bool:
        """Initialize HuggingFace Hub provider"""
        try:
            if not self.api_token:
                logger.error("HuggingFace API token not provided")
                return False

            self.llm = HuggingFaceHub(
                repo_id=self.model_name,
                huggingfacehub_api_token=self.api_token,
                model_kwargs={
                    "temperature": 0.7,
                    "max_length": 2000
                }
            )

            logger.info(f"HuggingFace Hub provider initialized with model: {self.model_name}")
            self.is_ready = True
            return True

        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace Hub provider: {e}")
            self.is_ready = False
            return False

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> str:
        """Generate text using HuggingFace Hub"""
        if not self.is_ready:
            raise RuntimeError("HuggingFace Hub provider is not ready")

        # Update model kwargs
        self.llm.model_kwargs["temperature"] = temperature
        self.llm.model_kwargs["max_length"] = max_tokens

        try:
            response = self.llm.invoke(prompt)
            return response
        except Exception as e:
            logger.error(f"HuggingFace Hub generation error: {e}")
            raise

    async def cleanup(self):
        """Cleanup HuggingFace Hub resources"""
        self.llm = None
        self.is_ready = False


class HuggingFaceLocalProvider(BaseLLMProvider):
    """Local HuggingFace model provider"""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or "gpt2"
        self.llm = None
        self.tokenizer = None
        self.model = None
        self.is_ready = False

    async def initialize(self) -> bool:
        """Initialize local HuggingFace model"""
        try:
            logger.info(f"Loading local HuggingFace model: {self.model_name}")

            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

            # Create pipeline
            pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=2000,
                temperature=0.7,
                do_sample=True
            )

            # Create LangChain wrapper
            self.llm = HuggingFacePipeline(pipeline=pipe)

            logger.info(f"Local HuggingFace provider initialized with model: {self.model_name}")
            self.is_ready = True
            return True

        except Exception as e:
            logger.error(f"Failed to initialize local HuggingFace provider: {e}")
            self.is_ready = False
            return False

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> str:
        """Generate text using local HuggingFace model"""
        if not self.is_ready:
            raise RuntimeError("Local HuggingFace provider is not ready")

        try:
            # Update pipeline parameters
            self.llm.pipeline.model_kwargs["temperature"] = temperature
            self.llm.pipeline.model_kwargs["max_length"] = max_tokens

            response = self.llm.invoke(prompt)
            return response
        except Exception as e:
            logger.error(f"Local HuggingFace generation error: {e}")
            raise

    async def cleanup(self):
        """Cleanup local HuggingFace resources"""
        self.llm = None
        self.model = None
        self.tokenizer = None
        self.is_ready = False


class LLMProviderFactory:
    """Factory for creating LLM providers"""

    @staticmethod
    def create_provider(
        provider_type: ModelProvider,
        **kwargs
    ) -> BaseLLMProvider:
        """Create an LLM provider instance"""
        if provider_type == ModelProvider.OLLAMA:
            return OllamaProvider(**kwargs)
        elif provider_type == ModelProvider.HUGGINGFACE_HUB:
            return HuggingFaceHubProvider(**kwargs)
        elif provider_type == ModelProvider.HUGGINGFACE_LOCAL:
            return HuggingFaceLocalProvider(**kwargs)
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")


class LLMService:
    """Unified LLM service supporting multiple providers"""

    def __init__(self):
        self.providers: Dict[str, BaseLLMProvider] = {}
        self.default_provider: Optional[str] = None
        self.is_ready = False

    async def initialize(self, config: Optional[Dict[str, Any]] = None):
        """Initialize LLM service with configured providers"""
        config = config or {}

        # Initialize Ollama if configured
        if config.get("enable_ollama", True):
            ollama_provider = LLMProviderFactory.create_provider(
                ModelProvider.OLLAMA,
                model_name=config.get("ollama_model", settings.ollama_model),
                base_url=config.get("ollama_base_url", settings.ollama_base_url)
            )
            if await ollama_provider.initialize():
                self.providers["ollama"] = ollama_provider
                if not self.default_provider:
                    self.default_provider = "ollama"

        # Initialize HuggingFace Hub if configured
        if config.get("enable_huggingface_hub", False):
            hf_hub_provider = LLMProviderFactory.create_provider(
                ModelProvider.HUGGINGFACE_HUB,
                model_name=config.get("huggingface_model"),
                api_token=config.get("huggingface_token")
            )
            if await hf_hub_provider.initialize():
                self.providers["huggingface_hub"] = hf_hub_provider
                if not self.default_provider:
                    self.default_provider = "huggingface_hub"

        # Initialize local HuggingFace if configured
        if config.get("enable_huggingface_local", False):
            hf_local_provider = LLMProviderFactory.create_provider(
                ModelProvider.HUGGINGFACE_LOCAL,
                model_name=config.get("huggingface_local_model", "gpt2")
            )
            if await hf_local_provider.initialize():
                self.providers["huggingface_local"] = hf_local_provider
                if not self.default_provider:
                    self.default_provider = "huggingface_local"

        self.is_ready = len(self.providers) > 0
        if self.is_ready:
            logger.info(f"LLM service initialized with providers: {list(self.providers.keys())}")
            logger.info(f"Default provider: {self.default_provider}")
        else:
            logger.error("No LLM providers could be initialized")

    async def generate(
        self,
        prompt: str,
        provider: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> str:
        """Generate text using specified or default provider"""
        if not self.is_ready:
            raise RuntimeError("LLM service is not ready")

        provider_name = provider or self.default_provider
        if provider_name not in self.providers:
            raise ValueError(f"Provider '{provider_name}' not available")

        provider_instance = self.providers[provider_name]
        return await provider_instance.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

    async def generate_with_chain(
        self,
        prompt_template: str,
        input_variables: Dict[str, Any],
        provider: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate using LangChain with prompt template"""
        if not self.is_ready:
            raise RuntimeError("LLM service is not ready")

        provider_name = provider or self.default_provider
        if provider_name not in self.providers:
            raise ValueError(f"Provider '{provider_name}' not available")

        provider_instance = self.providers[provider_name]

        # Create prompt template
        prompt = PromptTemplate(
            input_variables=list(input_variables.keys()),
            template=prompt_template
        )

        # Create and run chain
        chain = LLMChain(llm=provider_instance.llm, prompt=prompt)
        result = await chain.ainvoke(input_variables)
        return result["text"]

    async def cleanup(self):
        """Cleanup all providers"""
        for provider in self.providers.values():
            await provider.cleanup()
        self.providers.clear()
        self.default_provider = None
        self.is_ready = False

    def list_providers(self) -> List[str]:
        """List available providers"""
        return list(self.providers.keys())

    def get_default_provider(self) -> Optional[str]:
        """Get default provider name"""
        return self.default_provider

    def set_default_provider(self, provider_name: str):
        """Set default provider"""
        if provider_name not in self.providers:
            raise ValueError(f"Provider '{provider_name}' not available")
        self.default_provider = provider_name


# Global instance
llm_service = LLMService()