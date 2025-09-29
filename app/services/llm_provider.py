"""LLM Provider abstraction for multiple model sources"""
import logging
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod
from enum import Enum
import torch

from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from app.core.config import settings

logger = logging.getLogger(__name__)


class ModelProvider(str, Enum):
    """Available model providers"""
    OLLAMA = "ollama"
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



class HuggingFaceLocalProvider(BaseLLMProvider):
    """Local HuggingFace model provider with support for various models including Qwen"""

    def __init__(self, model_name: str = None, token: str = None):
        self.model_name = model_name or "gpt2"
        self.token = token  # HuggingFace token for gated models
        self.llm = None
        self.tokenizer = None
        self.model = None
        self.is_ready = False

    async def initialize(self) -> bool:
        """Initialize local HuggingFace model"""
        try:
            import os
            from pathlib import Path

            # Get actual cache directory
            cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
            hub_cache = os.path.join(cache_dir, "hub")
            model_cache_name = f"models--{self.model_name.replace('/', '--')}"

            logger.info(f"Loading local HuggingFace model: {self.model_name}")
            logger.info(f"Model cache directory: {hub_cache}")

            # Check if model already cached
            model_cache_path = Path(hub_cache) / model_cache_name
            if model_cache_path.exists():
                logger.info(f"Model found in cache: {model_cache_path}")
            else:
                logger.info(f"Model not in cache, will download to: {model_cache_path}")

            # Set HuggingFace token if provided (for gated models)
            if self.token:
                os.environ["HF_TOKEN"] = self.token
                logger.info("HuggingFace token configured for gated model access")

            # Load tokenizer and model with trust_remote_code for models like Qwen
            logger.info(f"Loading tokenizer for {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                token=self.token  # Pass token for authentication
            )
            logger.info("Tokenizer loaded successfully")

            # Determine device and dtype
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            logger.info(f"Loading model {self.model_name} (this may take time on first run)...")
            logger.info(f"Using device: {device}, dtype: {dtype}")

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=dtype,
                device_map="auto" if torch.cuda.is_available() else None,
                token=self.token  # Pass token for authentication
            )
            logger.info("Model loaded successfully")

            if not torch.cuda.is_available():
                self.model = self.model.to("cpu")

            # Create pipeline with proper token IDs
            pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=2000,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id or self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
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
            self.llm.pipeline.model_kwargs["max_new_tokens"] = max_tokens

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

        # Initialize local HuggingFace if configured
        if config.get("enable_huggingface_local", False):
            hf_local_provider = LLMProviderFactory.create_provider(
                ModelProvider.HUGGINGFACE_LOCAL,
                model_name=config.get("huggingface_local_model", settings.huggingface_local_model),
                token=config.get("huggingface_token", settings.huggingface_token)
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
        if not hasattr(provider_instance, 'llm') or provider_instance.llm is None:
            raise RuntimeError(f"Provider {provider_name} does not have a LangChain-compatible LLM")

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