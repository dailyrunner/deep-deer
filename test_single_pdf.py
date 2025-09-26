#!/usr/bin/env python3
"""Test single PDF embedding"""
import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_pdf_embedding():
    """Test embedding a single PDF file"""

    # 1. Index the PDF file
    print("📄 Testing PDF Embedding...")
    print("=" * 50)

    payload = {
        "folder_path": "/Users/danielim/CS/projects/deep-deer/docs",
        "store_name": "test_pdf",
        "recursive": False,
        "extensions": [".pdf"]
    }

    print(f"📁 Folder: {payload['folder_path']}")
    print(f"📦 Store name: {payload['store_name']}")
    print(f"📋 Extensions: {payload['extensions']}")
    print("\n🚀 Starting indexing...")

    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/vector/index-folder",
            json=payload,
            timeout=60  # 60 seconds timeout
        )

        if response.status_code == 200:
            result = response.json()
            print("\n✅ Indexing successful!")

            stats = result.get("stats", {})
            print("\n📊 Statistics:")
            print(f"  Total files: {stats.get('total_files', 0)}")
            print(f"  Indexed: {stats.get('indexed_files', 0)}")
            print(f"  Failed: {stats.get('failed_files', 0)}")
            print(f"  Chunks created: {stats.get('total_chunks', 0)}")

            if stats.get("errors"):
                print("\n⚠️ Errors:")
                for error in stats["errors"]:
                    print(f"  {error}")
        else:
            print(f"\n❌ Error: {response.status_code}")
            print(response.text)

    except requests.Timeout:
        print("\n⏱️ Request timed out (>60s)")
    except Exception as e:
        print(f"\n❌ Error: {e}")

    # 2. Search in the indexed content
    print("\n\n🔍 Testing Search...")
    print("=" * 50)

    search_query = "SKALA"  # Search for SKALA in the PDF

    search_payload = {
        "query": search_query,
        "store_names": ["test_pdf"],
        "k": 3
    }

    print(f"🔎 Searching for: '{search_query}'")

    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/vector/search",
            json=search_payload,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                results = result.get("results", [])
                print(f"\n📝 Found {len(results)} results:")

                for i, res in enumerate(results, 1):
                    print(f"\n{i}. Content preview:")
                    print(f"   {res['content'][:200]}...")
                    print(f"   Source: {res['metadata'].get('source', 'Unknown')}")
            else:
                print("\n❌ Search failed")
        else:
            print(f"\n❌ Error: {response.status_code}")

    except Exception as e:
        print(f"\n❌ Search error: {e}")

if __name__ == "__main__":
    # Check if server is running
    try:
        health = requests.get(f"{BASE_URL}/health", timeout=5)
        if health.status_code == 200:
            print("✅ Server is running\n")
            test_pdf_embedding()
        else:
            print("❌ Server health check failed")
    except:
        print("❌ Cannot connect to server. Start with: uv run python main.py")