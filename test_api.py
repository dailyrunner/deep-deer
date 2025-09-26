#!/usr/bin/env python
"""Test script for Deep Deer API"""
import asyncio
import aiohttp
import json
from typing import Any, Dict


class DeepDeerAPITester:
    """Test client for Deep Deer API"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def test_health(self):
        """Test health check endpoint"""
        print("\n📋 Testing Health Check...")
        async with self.session.get(f"{self.base_url}/health") as resp:
            data = await resp.json()
            print(f"Status: {resp.status}")
            print(f"Response: {json.dumps(data, indent=2)}")
            return resp.status == 200

    async def test_detailed_health(self):
        """Test detailed health check"""
        print("\n📋 Testing Detailed Health Check...")
        async with self.session.get(f"{self.base_url}/api/v1/health/detailed") as resp:
            data = await resp.json()
            print(f"Status: {resp.status}")
            print(f"Services Status:")
            for service, info in data.get("services", {}).items():
                if isinstance(info, dict):
                    print(f"  - {service}: {info.get('ready', False)}")
                else:
                    print(f"  - {service}: {info}")
            return resp.status == 200

    async def test_schema(self):
        """Test database schema endpoint"""
        print("\n🗂️ Testing Database Schema...")
        async with self.session.get(f"{self.base_url}/api/v1/query/schema") as resp:
            if resp.status == 200:
                data = await resp.json()
                schema = data.get("schema", {})
                print(f"Tables found: {schema.get('tables_count', 0)}")
                for table in schema.get("tables", []):
                    print(f"  - {table['name']}: {table['columns_count']} columns")
            else:
                print(f"Failed to get schema: {resp.status}")
            return resp.status == 200

    async def test_nlq(self, question: str = "Show all customers"):
        """Test natural language query"""
        print(f"\n🔍 Testing NLQ: '{question}'...")
        payload = {"question": question, "max_results": 5}

        async with self.session.post(
            f"{self.base_url}/api/v1/query/nlq",
            json=payload
        ) as resp:
            data = await resp.json()
            if data.get("success"):
                print(f"Generated SQL: {data.get('sql')}")
                if data.get("data"):
                    print(f"Results ({len(data['data'])} rows):")
                    for row in data["data"][:3]:
                        print(f"  {row}")
            else:
                print(f"Error: {data.get('error')}")
            return data.get("success", False)

    async def test_simple_idea(self, prompt: str = None):
        """Test simple idea generation"""
        prompt = prompt or "새로운 AI 기반 고객 서비스 아이디어"
        print(f"\n💡 Testing Simple Idea Generation: '{prompt}'...")

        payload = {
            "prompt": prompt,
            "context": "우리는 B2B SaaS 기업입니다"
        }

        async with self.session.post(
            f"{self.base_url}/api/v1/idea/generate-simple",
            json=payload
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                print(f"Provider: {data.get('provider')}")
                print(f"Ideas: {data.get('ideas', '')[:500]}...")
                return True
            else:
                print(f"Failed: {resp.status}")
                return False

    async def test_vector_stores(self):
        """Test vector store listing"""
        print("\n📦 Testing Vector Stores...")
        async with self.session.get(f"{self.base_url}/api/v1/vector/stores") as resp:
            data = await resp.json()
            if data.get("success"):
                stores = data.get("stores", [])
                print(f"Vector stores found: {len(stores)}")
                for store in stores:
                    print(f"  - {store.get('name')}: {store.get('type')}")
            return resp.status == 200

    async def test_index_table(self, table_name: str = "customers"):
        """Test table indexing"""
        print(f"\n🗃️ Testing Table Indexing: '{table_name}'...")
        payload = {"table_name": table_name}

        async with self.session.post(
            f"{self.base_url}/api/v1/vector/index-table",
            json=payload
        ) as resp:
            data = await resp.json()
            if resp.status == 200:
                print(f"Success: {data.get('message')}")
                print(f"Store created: {data.get('store_name')}")
                return True
            else:
                print(f"Failed: {data}")
                return False

    async def test_vector_search(self, query: str = "customer information"):
        """Test vector search"""
        print(f"\n🔎 Testing Vector Search: '{query}'...")
        payload = {"query": query, "k": 3}

        async with self.session.post(
            f"{self.base_url}/api/v1/vector/search",
            json=payload
        ) as resp:
            data = await resp.json()
            if data.get("success"):
                results = data.get("results", [])
                print(f"Found {len(results)} results:")
                for i, result in enumerate(results, 1):
                    print(f"  {i}. {result['content'][:100]}...")
            return data.get("success", False)

    async def test_rag_query(self, question: str = "What products do we have?"):
        """Test RAG query"""
        print(f"\n🤖 Testing RAG Query: '{question}'...")
        payload = {"question": question}

        async with self.session.post(
            f"{self.base_url}/api/v1/query/rag",
            json=payload
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                print(f"Answer: {data.get('answer', 'No answer')}")
                return True
            else:
                print(f"Failed: {resp.status}")
                return False

    async def run_all_tests(self):
        """Run all tests"""
        print("=" * 60)
        print("🦌 Deep Deer API Test Suite")
        print("=" * 60)

        tests = [
            ("Health Check", self.test_health()),
            ("Detailed Health", self.test_detailed_health()),
            ("Database Schema", self.test_schema()),
            # ("NLQ Query", self.test_nlq()),
            # ("Simple Idea Generation", self.test_simple_idea()),
            ("Vector Stores", self.test_vector_stores()),
            # ("Index Table", self.test_index_table()),
            # ("Vector Search", self.test_vector_search()),
            # ("RAG Query", self.test_rag_query()),
        ]

        results = []
        for name, test_coro in tests:
            try:
                result = await test_coro
                results.append((name, result))
            except Exception as e:
                print(f"\n❌ Error in {name}: {e}")
                results.append((name, False))

        print("\n" + "=" * 60)
        print("📊 Test Results Summary")
        print("=" * 60)
        for name, passed in results:
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{status}: {name}")

        total = len(results)
        passed = sum(1 for _, p in results if p)
        print(f"\nTotal: {passed}/{total} tests passed")


async def main():
    """Main test function"""
    import sys

    # Check if server is running
    print("🔍 Checking if server is running...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8000") as resp:
                if resp.status != 200:
                    print("❌ Server is not responding properly")
                    sys.exit(1)
    except aiohttp.ClientConnectorError:
        print("❌ Cannot connect to server at http://localhost:8000")
        print("Please start the server with: ./run.sh or uv run python main.py")
        sys.exit(1)

    # Run tests
    async with DeepDeerAPITester() as tester:
        await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())