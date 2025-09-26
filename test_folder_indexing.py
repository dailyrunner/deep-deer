#!/usr/bin/env python3
"""Test script for folder indexing functionality"""
import asyncio
import aiohttp
import json
from pathlib import Path

BASE_URL = "http://localhost:8000"


async def test_folder_indexing():
    """Test the folder indexing API endpoint"""
    async with aiohttp.ClientSession() as session:
        # Test 1: Index a folder with documents
        print("\n=== Test 1: Index a folder ===")

        # You can change this path to any folder you want to index
        # Example: index the project's own documentation
        folder_path = str(Path.cwd())  # Current directory

        payload = {
            "folder_path": folder_path,
            "store_name": "project_docs",
            "recursive": True,
            "extensions": [".py", ".md", ".txt", ".json", ".yaml"],  # Only index code and docs
            "exclude_patterns": ["*/__pycache__/*", "*.pyc", ".git/*", ".venv/*", "*.db"]
        }

        print(f"Indexing folder: {folder_path}")
        print(f"Extensions: {payload['extensions']}")
        print(f"Exclude patterns: {payload['exclude_patterns']}")

        try:
            async with session.post(
                f"{BASE_URL}/api/v1/vector/index-folder",
                json=payload
            ) as response:
                result = await response.json()

                if response.status == 200:
                    print("\n✅ Folder indexed successfully!")
                    print(f"Store name: {result.get('store_name')}")

                    stats = result.get("stats", {})
                    print("\n📊 Statistics:")
                    print(f"  Total files found: {stats.get('total_files', 0)}")
                    print(f"  Files indexed: {stats.get('indexed_files', 0)}")
                    print(f"  Files skipped: {stats.get('skipped_files', 0)}")
                    print(f"  Files failed: {stats.get('failed_files', 0)}")
                    print(f"  Total chunks created: {stats.get('total_chunks', 0)}")

                    if stats.get("errors"):
                        print("\n⚠️ Errors:")
                        for error in stats["errors"][:5]:  # Show first 5 errors
                            print(f"  - {error['file']}: {error['error']}")
                else:
                    print(f"\n❌ Failed to index folder: {result.get('detail', 'Unknown error')}")

        except Exception as e:
            print(f"\n❌ Error: {e}")

        # Test 2: Search in the indexed documents
        print("\n\n=== Test 2: Search in indexed documents ===")

        search_query = "vector store"  # Change this to search for different content

        search_payload = {
            "query": search_query,
            "store_names": ["project_docs"],
            "k": 5
        }

        print(f"Searching for: '{search_query}'")

        try:
            async with session.post(
                f"{BASE_URL}/api/v1/vector/search",
                json=search_payload
            ) as response:
                result = await response.json()

                if response.status == 200 and result.get("success"):
                    results = result.get("results", [])
                    print(f"\n📝 Found {len(results)} results:")

                    for i, res in enumerate(results, 1):
                        print(f"\n{i}. {res['metadata'].get('file_name', 'Unknown file')}")
                        print(f"   Type: {res['metadata'].get('file_type', 'unknown')}")
                        print(f"   Content preview: {res['content'][:200]}...")
                else:
                    print(f"\n❌ Search failed: {result.get('detail', 'Unknown error')}")

        except Exception as e:
            print(f"\n❌ Error: {e}")


async def test_specific_folder():
    """Test indexing a specific folder"""
    async with aiohttp.ClientSession() as session:
        print("\n=== Index Specific Folder Example ===")

        # Example: Index a specific documentation folder
        folder_path = "/Users/danielim/CS/projects/deep-deer"  # Change this path

        payload = {
            "folder_path": folder_path,
            "store_name": "deep_deer_docs",
            "recursive": False,  # Only files in the root folder
            "extensions": [".md", ".txt"],  # Only markdown and text files
        }

        print(f"Indexing folder: {folder_path}")
        print(f"Recursive: {payload['recursive']}")
        print(f"Extensions: {payload['extensions']}")

        try:
            async with session.post(
                f"{BASE_URL}/api/v1/vector/index-folder",
                json=payload
            ) as response:
                result = await response.json()

                if response.status == 200:
                    print("\n✅ Folder indexed successfully!")
                    stats = result.get("stats", {})
                    print(f"Indexed {stats.get('indexed_files', 0)} files")
                    print(f"Created {stats.get('total_chunks', 0)} chunks")
                else:
                    print(f"\n❌ Failed: {result.get('detail', 'Unknown error')}")

        except Exception as e:
            print(f"\n❌ Error: {e}")


async def main():
    """Run all tests"""
    print("🚀 Testing Folder Indexing API")
    print("=" * 50)

    # Make sure the server is running
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{BASE_URL}/health") as response:
                if response.status != 200:
                    print("❌ Server is not running. Please start the server first.")
                    return
    except:
        print("❌ Cannot connect to server. Please start the server with: uv run python main.py")
        return

    # Run tests
    await test_folder_indexing()

    # Uncomment to test a specific folder
    # await test_specific_folder()


if __name__ == "__main__":
    asyncio.run(main())
