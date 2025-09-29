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
            from app.services.llm_provider import llm_service

            # Try to use llm_service if it's initialized
            if llm_service.is_ready and llm_service.default_provider:
                provider = llm_service.providers.get(llm_service.default_provider)

                # Use the provider's LLM if available
                if provider and hasattr(provider, 'llm') and provider.llm:
                    self.llm = provider.llm
                    logger.info(f"Using {llm_service.default_provider} provider for LangChain pipeline")
                else:
                    # Fallback to Ollama
                    self.llm = Ollama(
                        base_url=settings.ollama_base_url,
                        model=settings.ollama_model,
                        temperature=0.7,
                        num_predict=2000
                    )
                    logger.info("Using Ollama as fallback for LangChain pipeline")
            else:
                # Default to Ollama if llm_service is not ready
                self.llm = Ollama(
                    base_url=settings.ollama_base_url,
                    model=settings.ollama_model,
                    temperature=0.7,
                    num_predict=2000
                )
                logger.info("Using default Ollama for LangChain pipeline")

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
            input_variables=["sql_data", "document_data", "web_data", "requirements"],
            template="""당신은 데이터 기반 비즈니스 전략 컨설턴트입니다.
기업의 내부 데이터와 시장 트렌드를 종합하여 실행 가능한 비즈니스 아이디어를 제시해주세요.

**중요: 모든 응답은 한국어로 작성해주세요.**

## 입력 정보

**사용자 요청**: {requirements}

**📊 내부 DB 데이터 (NLQ 결과)**:
{sql_data}

**📁 사내 문서 검색 결과 (Vector DB)**:
{document_data}

**🌐 외부 시장 트렌드 (웹 검색)**:
{web_data}

## 요구사항

다음 구조로 **3개의 구체적인 비즈니스 아이디어**를 한국어로 제시하세요:

### 💡 아이디어 1: [아이디어명]

**📊 데이터 근거**
- DB 데이터 분석 결과: [SQL 결과에서 발견된 구체적 패턴]
- 사내 문서 인사이트: [문서에서 발견된 기존 역량/경험]
- 시장 트렌드 연결: [웹 검색 결과와의 연결점]

**🎯 타겟 고객 & 시장**
- 구체적인 타겟 고객군
- 시장 규모 및 성장 가능성

**⚙️ 구현 방안**
- 기술적 접근 방법
- 필요한 리소스 및 인프라
- 단계별 실행 계획 (3단계)

**💰 비즈니스 모델**
- 수익 구조
- 예상 ROI 및 타임라인

**🚧 리스크 & 완화방안**
- 주요 위험 요소 2가지
- 각각의 대응 전략

### 💡 아이디어 2: [아이디어명]
[동일한 구조로 반복]

### 💡 아이디어 3: [아이디어명]
[동일한 구조로 반복]

## 📈 종합 우선순위 및 권장사항
1. [아이디어명] - 실행 용이성: [높음/중간/낮음], 수익성: [높음/중상/중간/낮음]
2. [아이디어명] - 실행 용이성: [높음/중간/낮음], 수익성: [높음/중상/중간/낮음]
3. [아이디어명] - 실행 용이성: [높음/중간/낮음], 수익성: [높음/중상/중간/낮음]

**첫 번째 추진 권장**: [이유와 함께]

비즈니스 아이디어:"""
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
                        template="""기업 내부 데이터 기반 비즈니스 아이디어 발굴을 위한 시장 트렌드 검색 키워드를 추출해주세요.

사용자 요청: {request}

다음 관점에서 키워드를 추출하세요:
1. **기술 트렌드**: 관련 신기술, AI, 디지털 트랜스포메이션
2. **시장 동향**: 산업 동향, 경쟁사 분석, 소비자 트렌드
3. **비즈니스 모델**: 새로운 수익 모델, 서비스 모델
4. **규제/정책**: 관련 규제 변화, 정부 정책

중요:
- 가장 핵심적인 키워드 3-5개만 선택
- 각 키워드는 2-3단어 이내
- 회사명이나 구체적인 내부 정보는 제외

키워드 (영문, 쉼표로 구분, 최대 5개):"""
                    )

                    keyword_chain = LLMChain(llm=self.llm, prompt=keyword_prompt)
                    keyword_result = await keyword_chain.ainvoke({"request": user_request})
                    search_keywords = keyword_result["text"].strip()

                    # Limit keywords to prevent URI too long error
                    keywords_list = [k.strip() for k in search_keywords.split(',')][:5]  # Max 5 keywords
                    search_keywords = ', '.join(keywords_list)

                    # Further limit total length
                    if len(search_keywords) > 200:
                        # Take only first 3 keywords if still too long
                        keywords_list = keywords_list[:3]
                        search_keywords = ', '.join(keywords_list)

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

            # Prepare structured data for each source
            sql_data_formatted = self._format_sql_data(results.get("sql_data"))
            document_data_formatted = self._format_document_data(results.get("document_context"))
            web_data_formatted = self._format_web_data(results.get("web_insights"))

            generated = await idea_chain.ainvoke({
                "sql_data": sql_data_formatted,
                "document_data": document_data_formatted,
                "web_data": web_data_formatted,
                "requirements": user_request
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

    def _format_sql_data(self, sql_data: dict) -> str:
        """Format SQL data for idea generation prompt"""
        if not sql_data or not sql_data.get("data"):
            return "내부 DB 데이터가 없습니다."

        formatted = f"실행된 쿼리: {sql_data.get('query', 'N/A')}\n"
        formatted += f"사용된 테이블: {', '.join(sql_data.get('tables', []))}\n\n"
        formatted += "주요 데이터 패턴:\n"

        data = sql_data.get("data", [])[:5]  # 상위 5개 행만 표시
        for i, row in enumerate(data, 1):
            formatted += f"{i}. "
            row_items = []
            for key, value in row.items():
                row_items.append(f"{key}: {value}")
            formatted += ", ".join(row_items) + "\n"

        if len(sql_data.get("data", [])) > 5:
            formatted += f"... 총 {len(sql_data.get('data', []))}개 레코드 중 상위 5개만 표시"

        return formatted

    def _format_document_data(self, document_data: list) -> str:
        """Format document data for idea generation prompt"""
        if not document_data:
            return "검색된 사내 문서가 없습니다."

        formatted = "관련 사내 문서 요약:\n\n"
        for i, doc in enumerate(document_data[:3], 1):  # 상위 3개 문서만
            formatted += f"{i}. 문서 내용: {doc.get('content', '')[:200]}...\n"
            metadata = doc.get('metadata', {})
            if 'source' in metadata:
                formatted += f"   출처: {metadata['source']}\n"
            formatted += "\n"

        return formatted

    def _format_web_data(self, web_data: dict) -> str:
        """Format web search data for idea generation prompt"""
        if not web_data:
            return "외부 시장 트렌드 정보가 없습니다."

        formatted = f"검색 키워드: {web_data.get('query', 'N/A')}\n\n"

        # 검색 결과 요약
        results = web_data.get('results', [])
        if results:
            formatted += "주요 검색 결과:\n"
            for i, result in enumerate(results[:3], 1):  # 상위 3개 결과
                formatted += f"{i}. {result.get('title', '제목 없음')}\n"
                formatted += f"   출처: {result.get('source', '출처 불명')}\n"
                formatted += f"   내용: {result.get('snippet', '')[:150]}...\n\n"

        # AI 요약 정보
        if 'summary' in web_data:
            formatted += "시장 트렌드 요약:\n"
            formatted += web_data['summary'][:500] + "..."

        return formatted

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