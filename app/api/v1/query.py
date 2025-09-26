"""Query API endpoints for NLQ and RAG"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

from app.services.nlq_processor import nlq_processor
from app.services.langchain_pipeline import langchain_pipeline

router = APIRouter()


class NLQueryRequest(BaseModel):
    """Natural language query request"""
    question: str = Field(..., description="Natural language question")
    max_results: int = Field(default=100, description="Maximum number of results")


class NLQueryResponse(BaseModel):
    """Natural language query response"""
    success: bool
    question: str
    sql: Optional[str] = None
    data: Optional[List[Dict[str, Any]]] = None
    tables_used: Optional[List[str]] = None
    error: Optional[str] = None


@router.post("/nlq", response_model=NLQueryResponse)
async def natural_language_query(request: NLQueryRequest) -> NLQueryResponse:
    """Convert natural language to SQL and execute"""
    try:
        if not nlq_processor.is_ready:
            await nlq_processor.initialize()

        result = await nlq_processor.process_query(request.question)

        return NLQueryResponse(
            success=result.success,
            question=request.question,
            sql=result.sql,
            data=result.data[:request.max_results] if result.data else None,
            tables_used=result.tables_used,
            error=result.error
        )

    except Exception as e:
        return NLQueryResponse(
            success=False,
            question=request.question,
            error=str(e)
        )


class RAGQueryRequest(BaseModel):
    """RAG query request"""
    question: str = Field(..., description="Question to answer")
    store_names: Optional[List[str]] = Field(None, description="Vector stores to search")
    include_sql: bool = Field(default=True, description="Include SQL data in response")
    provider: Optional[str] = Field(None, description="LLM provider to use")


@router.post("/rag")
async def rag_query(request: RAGQueryRequest) -> Dict[str, Any]:
    """Answer questions using RAG (Retrieval-Augmented Generation)"""
    try:
        if not langchain_pipeline.is_ready:
            await langchain_pipeline.initialize()

        answer = await langchain_pipeline.answer_question(
            question=request.question,
            store_names=request.store_names
        )

        return {
            "success": True,
            "question": request.question,
            "answer": answer,
            "stores_searched": request.store_names
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/schema")
async def get_schema() -> Dict[str, Any]:
    """Get database schema information"""
    try:
        if not nlq_processor.is_ready:
            await nlq_processor.initialize()

        schema = await nlq_processor.get_schema_summary()
        return {
            "success": True,
            "schema": schema
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sql-explain")
async def explain_sql(sql: str) -> Dict[str, Any]:
    """Explain what a SQL query does in natural language"""
    try:
        from app.services.llm_provider import llm_service

        if not llm_service.is_ready:
            raise HTTPException(status_code=503, detail="LLM service is not ready")

        prompt_template = """Explain the following SQL query in simple terms:

SQL Query:
{sql}

Explanation:"""

        explanation = await llm_service.generate_with_chain(
            prompt_template=prompt_template,
            input_variables={"sql": sql},
            temperature=0.3
        )

        return {
            "success": True,
            "sql": sql,
            "explanation": explanation
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))