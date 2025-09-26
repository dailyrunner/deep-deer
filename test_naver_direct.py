#!/usr/bin/env python3
"""Direct test of Naver Search API"""
import requests
import json

# Naver API credentials
NAVER_CLIENT_ID = "gbqzUVViEiF6WXhuq3gZ"
NAVER_CLIENT_SECRET = "y0YXaa5unU"

def test_naver_api():
    """Direct test of Naver API"""
    print("🔍 Direct Naver API Test")
    print("=" * 50)

    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
    }

    # Test query
    query = "AI 머신러닝"

    # Test different endpoints
    endpoints = [
        ("webkr", "https://openapi.naver.com/v1/search/webkr"),
        ("blog", "https://openapi.naver.com/v1/search/blog"),
        ("news", "https://openapi.naver.com/v1/search/news"),
        ("shop", "https://openapi.naver.com/v1/search/shop"),
    ]

    for endpoint_name, url in endpoints:
        print(f"\n📋 Testing {endpoint_name} endpoint:")
        print(f"   URL: {url}")

        params = {
            "query": query,
            "display": 5,
            "start": 1,
            "sort": "sim"
        }

        try:
            response = requests.get(url, headers=headers, params=params)
            print(f"   Status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                total = data.get("total", 0)
                items = data.get("items", [])
                print(f"   ✅ Success! Total: {total}, Items: {len(items)}")

                if items:
                    item = items[0]
                    title = item.get("title", "").replace("<b>", "").replace("</b>", "")
                    print(f"   First result: {title[:60]}...")
            else:
                print(f"   ❌ Error: {response.status_code}")
                print(f"   Response: {response.text[:200]}")

        except Exception as e:
            print(f"   ❌ Exception: {e}")

if __name__ == "__main__":
    test_naver_api()