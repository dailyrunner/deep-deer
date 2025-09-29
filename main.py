"""Deep Deer - Enterprise Data-Based Idea Generation Platform"""
import os
import logging
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.v1.router import api_router
from app.core.config import settings
from app.services.vector_store import vector_manager
from app.services.nlq_processor import nlq_processor
from app.services.llm_provider import llm_service
from app.services.ollama_service import ollama_service
from app.services.web_search import web_searcher
from app.services.langchain_pipeline import langchain_pipeline
from app.core.database import init_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    logger.info("Starting Deep Deer application...")

    # Initialize database
    await init_db()

    # Initialize services
    await vector_manager.initialize()
    await nlq_processor.initialize()
    await ollama_service.initialize()  # Initialize ollama_service

    # Initialize LLM service with configured providers
    llm_config = {
        "enable_ollama": settings.llm_provider == "ollama" or (not settings.enable_huggingface_local),
        "enable_huggingface_local": settings.llm_provider == "huggingface_local" or settings.enable_huggingface_local,
        "ollama_model": settings.ollama_model,
        "ollama_base_url": settings.ollama_base_url,
        "huggingface_token": settings.huggingface_token,
        "huggingface_local_model": settings.huggingface_local_model
    }
    await llm_service.initialize(llm_config)

    await web_searcher.initialize()
    await langchain_pipeline.initialize()

    logger.info("All services initialized successfully")

    yield

    # Cleanup
    logger.info("Shutting down Deep Deer application...")
    await vector_manager.cleanup()
    await ollama_service.cleanup()
    await llm_service.cleanup()
    await web_searcher.cleanup()


# Create FastAPI app
app = FastAPI(
    title="Deep Deer",
    description="AI-powered enterprise data analysis and idea generation platform",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(api_router, prefix="/api/v1")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Deep Deer",
        "version": "1.0.0",
        "status": "running",
        "description": "Enterprise Data-Based Idea Generation Platform"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "services": {
            "vector_store": vector_manager.is_ready,
            "nlq": nlq_processor.is_ready,
            "llm": llm_service.is_ready,
            "web_search": web_searcher.is_ready,
            "langchain": langchain_pipeline.is_ready
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level="info"
    )