"""Ollama service for LLM interactions using LangChain"""
import logging
import json
from typing import Optional, Dict, Any, List
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from app.core.config import settings

logger = logging.getLogger(__name__)


class OllamaService:
    """Service for interacting with Ollama LLM via LangChain"""

    def __init__(self):
        self.llm: Optional[Ollama] = None
        self.is_ready = False

    async def initialize(self):
        """Initialize Ollama service"""
        try:
            # Initialize LangChain Ollama
            self.llm = Ollama(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
                temperature=0.7,
                num_predict=2000,
                timeout=settings.ollama_timeout
            )

            # Test the connection
            try:
                test_response = self.llm.invoke("Hello")
                logger.info(f"Ollama service initialized with model: {settings.ollama_model}")
                self.is_ready = True
            except Exception as e:
                logger.warning(f"Model test failed, attempting to continue: {e}")
                self.is_ready = True  # Continue anyway, model might need pulling

        except Exception as e:
            logger.error(f"Failed to initialize Ollama service: {e}")
            self.is_ready = False

    async def cleanup(self):
        """Cleanup resources"""
        self.llm = None
        self.is_ready = False

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        format: Optional[str] = None
    ) -> str:
        """Generate response from Ollama using LangChain"""
        if not self.is_ready:
            raise RuntimeError("Ollama service is not ready")

        try:
            # Configure Ollama with runtime parameters
            self.llm.temperature = temperature
            self.llm.num_predict = max_tokens

            # Combine system prompt and user prompt if needed
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"

            # Use LangChain to generate response
            response = self.llm.invoke(full_prompt)

            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    async def generate_sql(
        self,
        question: str,
        schema_info: str,
        examples: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Generate SQL from natural language question"""
        system_prompt = """You are an expert SQL query generator.
        Generate only valid SQL queries based on the provided schema.
        Return only the SQL query without any explanation or markdown formatting."""

        # Build prompt with schema and examples
        prompt = f"""Database Schema:
{schema_info}

"""
        if examples:
            prompt += "Examples:\n"
            for ex in examples[:3]:  # Use up to 3 examples
                prompt += f"Question: {ex['question']}\nSQL: {ex['sql']}\n\n"

        prompt += f"Question: {question}\nSQL:"

        response = await self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.1,  # Low temperature for accuracy
            max_tokens=500
        )

        # Clean up the response
        sql = response.strip()
        # Remove markdown code blocks if present
        if sql.startswith("```sql"):
            sql = sql[6:]
        if sql.startswith("```"):
            sql = sql[3:]
        if sql.endswith("```"):
            sql = sql[:-3]

        return sql.strip()

    async def generate_idea(
        self,
        context: str,
        requirements: str,
        temperature: float = 0.8
    ) -> str:
        """Generate business ideas based on context"""
        system_prompt = """You are a creative business strategist and innovation consultant.
        Generate innovative and practical business ideas based on the provided data and requirements.
        Focus on actionable insights and consider market trends, feasibility, and potential impact."""

        prompt = f"""Based on the following internal data and requirements, generate innovative business ideas:

Context Data:
{context}

Requirements:
{requirements}

Please provide:
1. A brief overview of the opportunity
2. Specific business idea or service concept
3. Target market and use cases
4. Implementation approach
5. Potential challenges and solutions
6. Expected impact and benefits"""

        return await self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=3000
        )

    async def extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract entities and keywords from text"""
        system_prompt = """You are an entity extraction expert.
        Extract key entities, concepts, and information from the text.
        Return the result as a JSON object."""

        prompt = f"""Extract entities and key information from the following text:

{text}

Return a JSON object with:
- entities: list of important entities (people, organizations, products)
- keywords: list of key concepts and topics
- metrics: any numerical data or metrics mentioned
- dates: any dates or time periods mentioned"""

        response = await self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.1,
            format="json"
        )

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON response: {response}")
            return {
                "entities": [],
                "keywords": [],
                "metrics": [],
                "dates": []
            }

    async def filter_sensitive_info(self, text: str) -> str:
        """Filter out sensitive information from text"""
        system_prompt = """You are a security expert responsible for data privacy.
        Remove or mask any sensitive information while preserving the general meaning."""

        sensitive_fields = ", ".join(settings.sensitive_fields_list)
        prompt = f"""Review and sanitize the following text by removing or masking sensitive information.

Sensitive fields to watch for: {sensitive_fields}
Also remove: personal identification numbers, private addresses, internal system details.

Original text:
{text}

Return the sanitized version that is safe for external use while maintaining the core information."""

        return await self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.1,
            max_tokens=2000
        )


# Global instance
ollama_service = OllamaService()