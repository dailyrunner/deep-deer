#!/usr/bin/env python3
"""Test web search functionality"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_web_search():
    """Test the web search through idea generation API"""

    print("🌐 Testing Web Search Functionality")
    print("=" * 50)

    # Test 1: Simple idea generation with web search
    payload = {
        "request": "AI 기반 고객 서비스 자동화 솔루션",
        "use_web_search": True
    }

    print("📋 Request:")
    print(f"  Topic: {payload['request']}")
    print(f"  Web Search: {payload['use_web_search']}")
    print("\n⏳ Generating ideas with web search...")

    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/idea/generate",
            json=payload,
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()

            print("\n✅ Response received!")

            # Check if web insights were included
            if "web_insights" in result:
                web_data = result["web_insights"]
                print("\n🌐 Web Search Results:")

                if web_data and isinstance(web_data, dict):
                    # Check search results
                    if "results" in web_data:
                        results = web_data["results"]
                        print(f"  Found {len(results)} web results")

                        for i, res in enumerate(results[:3], 1):
                            print(f"\n  {i}. {res.get('title', 'No title')}")
                            print(f"     URL: {res.get('url', 'No URL')}")
                            print(f"     Source: {res.get('source', 'Unknown')}")

                    # Check if results were filtered
                    if web_data.get("filtered"):
                        print("\n  ⚠️ Some results were filtered for sensitive information")

                    # Show summary if available
                    if "summary" in web_data:
                        print(f"\n  📝 Web Insights Summary:")
                        print(f"     {web_data['summary'][:200]}...")
                else:
                    print("  No web search data available")

            # Show generated ideas
            if "ideas" in result:
                print("\n💡 Generated Ideas (first 500 chars):")
                print(result["ideas"][:500] + "...")

        else:
            print(f"\n❌ Error: {response.status_code}")
            print(response.text)

    except requests.Timeout:
        print("\n⏱️ Request timed out")
    except Exception as e:
        print(f"\n❌ Error: {e}")

def test_direct_web_search():
    """Test web search directly through internal testing"""
    print("\n\n🔍 Direct Web Search Test")
    print("=" * 50)

    # This would need access to the web_searcher directly
    # For now, we'll test through the API

    payload = {
        "request": "latest AI trends 2024",
        "use_web_search": True
    }

    print(f"🔎 Searching for: {payload['request']}")

    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/idea/generate-simple",
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            print("\n✅ Simple generation with web search completed")

            if "idea" in result:
                print(f"\n💭 Result preview:")
                print(result["idea"][:300] + "...")
        else:
            print(f"\n❌ Error: {response.status_code}")

    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    # Check server health
    try:
        health = requests.get(f"{BASE_URL}/health", timeout=5)
        if health.status_code == 200:
            print("✅ Server is running\n")
            test_web_search()
            test_direct_web_search()
        else:
            print("❌ Server health check failed")
    except:
        print("❌ Cannot connect to server")