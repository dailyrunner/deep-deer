"""Idea generation API endpoints"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

from app.services.langchain_pipeline import langchain_pipeline
from app.services.llm_provider import llm_service

router = APIRouter()


class IdeaRequest(BaseModel):
    """Request model for idea generation"""
    request: str = Field(..., description="User's request for idea generation")
    use_web_search: bool = Field(default=True, description="Whether to use web search for market trends")
    provider: Optional[str] = Field(default=None, description="LLM provider to use")
    max_results: int = Field(default=5, description="Maximum number of results to return")
    include_sql_data: bool = Field(default=True, description="Include database analysis")
    include_documents: bool = Field(default=True, description="Include document search")


class IdeaResponse(BaseModel):
    """Response model for idea generation"""
    success: bool
    request: str
    ideas: Optional[str] = None
    sql_context: Optional[Dict[str, Any]] = None
    document_context: Optional[List[Dict[str, Any]]] = None
    web_insights: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@router.post("/generate", response_model=IdeaResponse)
async def generate_idea(request: IdeaRequest) -> IdeaResponse:
    """Generate business ideas based on internal and external data"""
    try:
        if not langchain_pipeline.is_ready:
            await langchain_pipeline.initialize()

        # Process the idea request
        result = await langchain_pipeline.process_idea_request(
            user_request=request.request,
            use_web_search=request.use_web_search
        )

        # Format response
        response = IdeaResponse(
            success=result.get("error") is None,
            request=request.request,
            ideas=result.get("generated_ideas"),
            sql_context=result.get("sql_data"),
            document_context=result.get("document_context"),
            web_insights=result.get("web_insights"),
            error=result.get("error")
        )

        return response

    except Exception as e:
        return IdeaResponse(
            success=False,
            request=request.request,
            error=str(e)
        )


class SimpleIdeaRequest(BaseModel):
    """Simple request for quick idea generation"""
    prompt: str = Field(..., description="Idea generation prompt")
    context: Optional[str] = Field(None, description="Additional context")
    provider: Optional[str] = Field(None, description="LLM provider to use")


@router.post("/generate-simple")
async def generate_simple_idea(request: SimpleIdeaRequest) -> Dict[str, Any]:
    """Generate ideas with simple prompt (no data integration)"""
    try:
        if not llm_service.is_ready:
            raise HTTPException(status_code=503, detail="LLM service is not ready")

        prompt_template = """You are a creative business strategist.

Context: {context}

Request: {prompt}

Generate innovative business ideas that are practical and actionable.

Ideas:"""

        idea_text = await llm_service.generate_with_chain(
            prompt_template=prompt_template,
            input_variables={
                "context": request.context or "General business context",
                "prompt": request.prompt
            },
            provider=request.provider
        )

        return {
            "success": True,
            "prompt": request.prompt,
            "ideas": idea_text,
            "provider": request.provider or llm_service.get_default_provider()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-opportunity")
async def analyze_opportunity(
    topic: str,
    market_data: Optional[str] = None,
    internal_data: Optional[str] = None,
    provider: Optional[str] = None
) -> Dict[str, Any]:
    """Analyze a business opportunity"""
    try:
        if not llm_service.is_ready:
            raise HTTPException(status_code=503, detail="LLM service is not ready")

        prompt_template = """Analyze the following business opportunity:

Topic: {topic}

Market Data:
{market_data}

Internal Data:
{internal_data}

Provide:
1. Opportunity Assessment
2. Market Potential
3. Required Resources
4. Risk Analysis
5. Recommended Next Steps

Analysis:"""

        analysis = await llm_service.generate_with_chain(
            prompt_template=prompt_template,
            input_variables={
                "topic": topic,
                "market_data": market_data or "No market data provided",
                "internal_data": internal_data or "No internal data provided"
            },
            provider=provider,
            temperature=0.7
        )

        return {
            "success": True,
            "topic": topic,
            "analysis": analysis,
            "provider": provider or llm_service.get_default_provider()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))