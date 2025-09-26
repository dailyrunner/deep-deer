"""Health check endpoints"""
from fastapi import APIRouter
from typing import Dict, Any

from app.services.vector_store import vector_manager
from app.services.nlq_processor import nlq_processor
from app.services.llm_provider import llm_service
from app.services.web_search import web_searcher
from app.services.langchain_pipeline import langchain_pipeline

router = APIRouter()


@router.get("/")
async def health_check() -> Dict[str, Any]:
    """Basic health check"""
    return {
        "status": "healthy",
        "services": {
            "vector_store": vector_manager.is_ready,
            "nlq_processor": nlq_processor.is_ready,
            "llm_service": llm_service.is_ready,
            "web_search": web_searcher.is_ready,
            "langchain": langchain_pipeline.is_ready
        }
    }


@router.get("/detailed")
async def detailed_health() -> Dict[str, Any]:
    """Detailed health check with service information"""
    return {
        "status": "healthy",
        "services": {
            "vector_store": {
                "ready": vector_manager.is_ready,
                "stores": vector_manager.list_stores() if vector_manager.is_ready else []
            },
            "nlq_processor": {
                "ready": nlq_processor.is_ready,
                "schema_tables": len(nlq_processor.schema_cache) if nlq_processor.is_ready else 0
            },
            "llm_service": {
                "ready": llm_service.is_ready,
                "providers": llm_service.list_providers() if llm_service.is_ready else [],
                "default": llm_service.get_default_provider() if llm_service.is_ready else None
            },
            "web_search": {
                "ready": web_searcher.is_ready
            },
            "langchain": {
                "ready": langchain_pipeline.is_ready
            }
        }
    }