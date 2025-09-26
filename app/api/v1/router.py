"""Main API v1 router"""
from fastapi import APIRouter
from app.api.v1 import idea, query, vector, health

api_router = APIRouter()

# Include sub-routers
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(idea.router, prefix="/idea", tags=["idea"])
api_router.include_router(query.router, prefix="/query", tags=["query"])
api_router.include_router(vector.router, prefix="/vector", tags=["vector"])