#!/usr/bin/env python3
"""Test Naver Search API functionality"""
import asyncio
import logging
from app.services.web_search import web_searcher
from app.services.ollama_service import ollama_service
from app.services.langchain_pipeline import langchain_pipeline

logging.basicConfig(level=logging.INFO)

async def test_naver_search():
    """Test Naver search functionality"""
    print("🔍 Testing Naver Search API")
    print("=" * 50)

    # Initialize services
    await ollama_service.initialize()
    await web_searcher.initialize()
    await langchain_pipeline.initialize()

    # Test queries
    test_queries = [
        "AI 고객 서비스 자동화",
        "머신러닝 추천 시스템",
        "빅데이터 분석 플랫폼",
        "chatbot development frameworks"
    ]

    for query in test_queries:
        print(f"\n📝 Query: {query}")
        print("-" * 40)

        try:
            # Direct search test
            results = await web_searcher.search(query, max_results=5)

            if results:
                print(f"✅ Found {len(results)} results:")
                for i, result in enumerate(results[:3], 1):
                    print(f"\n  {i}. {result.title[:80]}...")
                    print(f"     URL: {result.url}")
                    print(f"     Source: {result.source}")
                    print(f"     Snippet: {result.snippet[:100]}...")
                    if result.metadata:
                        print(f"     Type: {result.metadata.get('search_type', 'unknown')}")
            else:
                print("❌ No results found")

        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n" + "=" * 50)
    print("🎯 Testing with keyword extraction and search")
    print("=" * 50)

    # Test with keyword extraction
    complex_query = "우리 회사의 고객 데이터를 활용해서 개인화된 마케팅 캠페인을 자동화하는 AI 시스템을 구축하고 싶습니다"
    print(f"\n📝 Complex Query: {complex_query}")

    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain

    keyword_prompt = PromptTemplate(
        input_variables=["request"],
        template="""Extract 3-5 key search terms from this business request for web search.
Focus on technology terms, industry concepts, and market trends.
Avoid specific company names or internal details.

Request: {request}

Keywords (comma-separated):"""
    )

    keyword_chain = LLMChain(llm=langchain_pipeline.llm, prompt=keyword_prompt)
    result = await keyword_chain.ainvoke({"request": complex_query})
    keywords = result["text"].strip()

    print(f"✨ Extracted keywords: {keywords}")

    # Search with extracted keywords
    results = await web_searcher.search(keywords, max_results=5)
    if results:
        print(f"\n📊 Found {len(results)} results for extracted keywords:")
        for i, result in enumerate(results[:2], 1):
            print(f"\n  {i}. {result.title[:80]}...")
            print(f"     Source: {result.source}")
    else:
        print("❌ No results found for extracted keywords")

    # Cleanup
    await ollama_service.cleanup()
    await web_searcher.cleanup()

if __name__ == "__main__":
    asyncio.run(test_naver_search())