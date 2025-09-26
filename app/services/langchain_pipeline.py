"""LangChain-based integration pipeline"""
import logging
from typing import List, Dict, Any, Optional
from langchain_community.llms import Ollama
from app.services.embeddings import embedding_service
from langchain.chains import LLMChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

from app.core.config import settings
from app.services.vector_store import vector_manager
from app.services.nlq_processor import nlq_processor

logger = logging.getLogger(__name__)


class LangChainPipeline:
    """Main LangChain integration pipeline for Deep Deer"""

    def __init__(self):
        self.llm = None
        self.embeddings = None
        self.is_ready = False

    async def initialize(self):
        """Initialize LangChain components"""
        try:
            # Initialize Ollama LLM
            self.llm = Ollama(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
                temperature=0.7,
                num_predict=2000
            )

            # Initialize embeddings
            embedding_service.initialize()
            self.embeddings = embedding_service.get_embeddings()

            self.is_ready = True
            logger.info("LangChain pipeline initialized")
        except Exception as e:
            logger.error(f"Failed to initialize LangChain pipeline: {e}")
            self.is_ready = False

    def create_sql_chain(self):
        """Create a chain for SQL generation"""
        sql_prompt = PromptTemplate(
            input_variables=["schema", "question", "examples"],
            template="""You are an expert SQL query generator.

Database Schema:
{schema}

Examples:
{examples}

Question: {question}

Generate a valid SQL query to answer the question. Return only the SQL query without any explanation.

SQL Query:"""
        )

        return LLMChain(llm=self.llm, prompt=sql_prompt)

    def create_idea_generation_chain(self):
        """Create a chain for idea generation"""
        idea_prompt = PromptTemplate(
            input_variables=["context", "requirements", "market_trends"],
            template="""You are a creative business strategist and innovation consultant.

Internal Data Context:
{context}

Business Requirements:
{requirements}

Market Trends (if available):
{market_trends}

Based on the above information, generate innovative business ideas that:
1. Leverage the internal data and capabilities
2. Address market opportunities
3. Are feasible and actionable

Provide:
1. **Business Opportunity Overview**
2. **Specific Service/Product Concepts** (at least 3)
3. **Target Market and Use Cases**
4. **Implementation Approach**
5. **Competitive Advantages**
6. **Potential Challenges and Mitigation Strategies**
7. **Expected Impact and ROI**

Business Ideas:"""
        )

        return LLMChain(llm=self.llm, prompt=idea_prompt)

    def create_rag_chain(self, retriever):
        """Create a RAG chain for question answering"""
        # Create prompt template
        system_prompt = """You are an intelligent assistant helping users find insights from their data.
        Use the following context to answer the question. If you don't know the answer, say so.

Context:
{context}

Question: {input}

Answer:"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt)
        ])

        # Create document chain
        document_chain = create_stuff_documents_chain(self.llm, prompt)

        # Create retrieval chain
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        return retrieval_chain

    async def process_idea_request(
        self,
        user_request: str,
        use_web_search: bool = True
    ) -> Dict[str, Any]:
        """Process a complete idea generation request"""
        if not self.is_ready:
            raise RuntimeError("LangChain pipeline is not ready")

        results = {
            "request": user_request,
            "sql_data": None,
            "document_context": None,
            "web_insights": None,
            "generated_ideas": None,
            "error": None
        }

        try:
            import asyncio

            # Create tasks for parallel execution
            tasks = []

            # Task 1: Extract relevant data from database using NLQ
            logger.info("Starting parallel data collection...")

            async def get_sql_data():
                try:
                    nlq_result = await nlq_processor.process_query(user_request)
                    if nlq_result.success and nlq_result.data:
                        return {
                            "query": nlq_result.sql,
                            "data": nlq_result.data[:10],  # Limit to 10 rows
                            "tables": nlq_result.tables_used
                        }
                except Exception as e:
                    logger.error(f"Error in NLQ processing: {e}")
                return None

            # Task 2: Search relevant documents from vector store
            async def get_document_context():
                try:
                    doc_results = await vector_manager.search(
                        query=user_request,
                        k=5
                    )
                    if doc_results:
                        return [
                            {
                                "content": doc.page_content[:500],
                                "metadata": doc.metadata
                            }
                            for doc in doc_results
                        ]
                except Exception as e:
                    logger.error(f"Error in vector search: {e}")
                return None

            # Task 3: Web search for market trends (if enabled)
            async def get_web_insights():
                if not use_web_search:
                    return None
                try:
                    from app.services.web_search import web_searcher
                    # Extract keywords from user request first
                    keyword_prompt = PromptTemplate(
                        input_variables=["request"],
                        template="""Extract 3-5 key search terms from this business request for web search.
Focus on technology terms, industry concepts, and market trends.
Avoid specific company names or internal details.

Request: {request}

Keywords (comma-separated):"""
                    )

                    keyword_chain = LLMChain(llm=self.llm, prompt=keyword_prompt)
                    keyword_result = await keyword_chain.ainvoke({"request": user_request})
                    search_keywords = keyword_result["text"].strip()

                    logger.info(f"Extracted search keywords: {search_keywords}")

                    web_results = await web_searcher.search_and_summarize(
                        query=search_keywords,
                        context="",  # Will be updated with actual context later
                        max_results=5
                    )
                    return web_results
                except Exception as e:
                    logger.error(f"Error in web search: {e}")
                return None

            # Execute all tasks in parallel
            sql_data, document_context, web_insights = await asyncio.gather(
                get_sql_data(),
                get_document_context(),
                get_web_insights()
            )

            # Store results
            results["sql_data"] = sql_data
            results["document_context"] = document_context
            results["web_insights"] = web_insights

            # Step 4: Generate ideas using all collected data
            logger.info("Step 4: Generating business ideas")
            idea_chain = self.create_idea_generation_chain()

            context = self._prepare_context_for_ideas(results)
            market_trends = results.get("web_insights", {}).get("summary", "No market trends available")

            generated = await idea_chain.ainvoke({
                "context": context,
                "requirements": user_request,
                "market_trends": market_trends
            })

            results["generated_ideas"] = generated["text"]

        except Exception as e:
            logger.error(f"Error in idea generation pipeline: {e}")
            results["error"] = str(e)

        return results

    async def _create_safe_web_query(self, original_query: str, current_results: Dict) -> str:
        """Create a safe web search query by extracting key concepts and removing sensitive information"""
        # Extract general concepts without specific internal data
        filter_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="""Convert the following query into an optimized web search query.
Extract the main keywords and concepts for effective web search.
Remove any company-specific names, internal data, or sensitive information.
Focus on industry terms, technologies, and general business concepts.

Original Query: {query}
Internal Context Summary: {context}

Output a concise search query with 2-5 key terms that will return relevant market insights.
Web Search Query:"""
        )

        chain = LLMChain(llm=self.llm, prompt=filter_prompt)

        context_summary = self._summarize_internal_context(current_results)[:500]

        result = await chain.ainvoke({
            "query": original_query,
            "context": context_summary
        })

        return result["text"].strip()

    def _summarize_internal_context(self, results: Dict) -> str:
        """Summarize internal context from results"""
        context_parts = []

        if results.get("sql_data") and results["sql_data"]["data"]:
            context_parts.append(f"Database contains {len(results['sql_data']['data'])} relevant records")

        if results.get("document_context"):
            context_parts.append(f"Found {len(results['document_context'])} relevant documents")

        return ". ".join(context_parts) if context_parts else "No internal context available"

    def _prepare_context_for_ideas(self, results: Dict) -> str:
        """Prepare comprehensive context for idea generation"""
        context_parts = []

        # Add SQL data context
        if results.get("sql_data") and results["sql_data"]["data"]:
            context_parts.append("=== Database Insights ===")
            for row in results["sql_data"]["data"][:5]:  # Top 5 rows
                context_parts.append(str(row))

        # Add document context
        if results.get("document_context"):
            context_parts.append("\n=== Document Insights ===")
            for doc in results["document_context"][:3]:  # Top 3 documents
                context_parts.append(doc["content"])

        return "\n\n".join(context_parts) if context_parts else "Limited internal data available"

    async def answer_question(self, question: str, store_names: Optional[List[str]] = None) -> str:
        """Answer a question using RAG"""
        if not self.is_ready:
            raise RuntimeError("LangChain pipeline is not ready")

        # Get relevant documents
        docs = await vector_manager.search(
            query=question,
            store_names=store_names,
            k=5
        )

        if not docs:
            # Try to answer using SQL
            nlq_result = await nlq_processor.process_query(question)
            if nlq_result.success and nlq_result.data:
                return self._format_sql_answer(nlq_result)
            return "I couldn't find relevant information to answer your question."

        # Create a simple QA chain
        qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""Based on the following context, answer the question.
If the answer is not in the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer:"""
        )

        qa_chain = LLMChain(llm=self.llm, prompt=qa_prompt)

        # Prepare context from documents
        context = "\n\n".join([doc.page_content for doc in docs])

        result = await qa_chain.ainvoke({
            "context": context,
            "question": question
        })

        return result["text"]

    def _format_sql_answer(self, nlq_result) -> str:
        """Format SQL query results as an answer"""
        if not nlq_result.data:
            return "The query returned no results."

        # Format first few rows as answer
        answer_parts = [f"Found {len(nlq_result.data)} results:"]

        for row in nlq_result.data[:5]:  # Show first 5 rows
            row_str = ", ".join([f"{k}: {v}" for k, v in row.items()])
            answer_parts.append(f"- {row_str}")

        if len(nlq_result.data) > 5:
            answer_parts.append(f"... and {len(nlq_result.data) - 5} more results")

        return "\n".join(answer_parts)


# Global instance
langchain_pipeline = LangChainPipeline()