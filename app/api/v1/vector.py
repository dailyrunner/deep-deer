"""Vector store management API endpoints"""
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import tempfile
import shutil
from pathlib import Path
import logging

from app.services.vector_store import vector_manager

router = APIRouter()
logger = logging.getLogger(__name__)


class IndexTableRequest(BaseModel):
    """Request to index a database table"""
    table_name: str = Field(..., description="Name of the table to index")


@router.post("/index-table")
async def index_table(request: IndexTableRequest) -> Dict[str, Any]:
    """Index a database table into vector store"""
    try:
        if not vector_manager.is_ready:
            await vector_manager.initialize()

        success = await vector_manager.index_database_table(request.table_name)

        if success:
            return {
                "success": True,
                "message": f"Table {request.table_name} indexed successfully",
                "store_name": f"db_{request.table_name}"
            }
        else:
            raise HTTPException(status_code=400, detail=f"Failed to index table {request.table_name}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/index-document")
async def index_document(
    file: UploadFile = File(...),
    store_name: str = "documents"
) -> Dict[str, Any]:
    """Upload and index a document"""
    try:
        if not vector_manager.is_ready:
            await vector_manager.initialize()

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name

        try:
            # Index the document
            success = await vector_manager.index_document(tmp_path, store_name)

            if success:
                return {
                    "success": True,
                    "message": f"Document {file.filename} indexed successfully",
                    "store_name": store_name
                }
            else:
                raise HTTPException(status_code=400, detail=f"Failed to index document {file.filename}")

        finally:
            # Clean up temporary file
            Path(tmp_path).unlink(missing_ok=True)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class IndexFolderRequest(BaseModel):
    """Request to index all files in a folder"""
    folder_path: str = Field(..., description="Path to the folder to index")
    store_name: str = Field(default="documents", description="Name of the vector store")
    recursive: bool = Field(default=True, description="Process subfolders recursively")
    extensions: Optional[List[str]] = Field(
        default=None,
        description="File extensions to process (e.g., ['.pdf', '.txt', '.docx']). If None, all supported formats are processed"
    )
    exclude_patterns: Optional[List[str]] = Field(
        default=None,
        description="Patterns to exclude (e.g., ['*.tmp', '__pycache__/*'])"
    )


@router.post("/index-folder")
async def index_folder(request: IndexFolderRequest) -> Dict[str, Any]:
    """Index all documents in a folder"""
    try:
        if not vector_manager.is_ready:
            await vector_manager.initialize()

        folder_path = Path(request.folder_path)

        # Validate folder exists
        if not folder_path.exists():
            raise HTTPException(status_code=404, detail=f"Folder not found: {request.folder_path}")

        if not folder_path.is_dir():
            raise HTTPException(status_code=400, detail=f"Path is not a directory: {request.folder_path}")

        # Index the folder
        result = await vector_manager.index_folder(
            folder_path=str(folder_path),
            store_name=request.store_name,
            recursive=request.recursive,
            extensions=request.extensions,
            exclude_patterns=request.exclude_patterns
        )

        return {
            "success": True,
            "message": f"Folder indexed successfully",
            "store_name": request.store_name,
            "stats": result
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error indexing folder: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class SearchRequest(BaseModel):
    """Vector search request"""
    query: str = Field(..., description="Search query")
    store_names: Optional[List[str]] = Field(None, description="Stores to search")
    k: int = Field(default=5, description="Number of results")
    score_threshold: Optional[float] = Field(None, description="Minimum similarity score")


@router.post("/search")
async def vector_search(request: SearchRequest) -> Dict[str, Any]:
    """Search across vector stores"""
    try:
        if not vector_manager.is_ready:
            await vector_manager.initialize()

        if request.score_threshold is not None:
            results = await vector_manager.search_with_score(
                query=request.query,
                store_names=request.store_names,
                k=request.k,
                score_threshold=request.score_threshold
            )
            return {
                "success": True,
                "query": request.query,
                "results": [
                    {
                        "content": doc.page_content[:500],
                        "metadata": doc.metadata,
                        "score": score
                    }
                    for doc, score in results
                ]
            }
        else:
            results = await vector_manager.search(
                query=request.query,
                store_names=request.store_names,
                k=request.k
            )
            return {
                "success": True,
                "query": request.query,
                "results": [
                    {
                        "content": doc.page_content[:500],
                        "metadata": doc.metadata
                    }
                    for doc in results
                ]
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stores")
async def list_stores() -> Dict[str, Any]:
    """List all vector stores"""
    try:
        if not vector_manager.is_ready:
            await vector_manager.initialize()

        stores = vector_manager.list_stores()
        store_info = []

        for store_name in stores:
            info = vector_manager.get_store_info(store_name)
            if info:
                store_info.append(info)

        return {
            "success": True,
            "stores": store_info
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/stores/{store_name}")
async def delete_store(store_name: str) -> Dict[str, Any]:
    """Delete a vector store"""
    try:
        if not vector_manager.is_ready:
            await vector_manager.initialize()

        success = await vector_manager.delete_store(store_name)

        if success:
            return {
                "success": True,
                "message": f"Store {store_name} deleted successfully"
            }
        else:
            raise HTTPException(status_code=400, detail=f"Failed to delete store {store_name}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))