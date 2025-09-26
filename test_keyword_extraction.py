#!/usr/bin/env python3
"""Test keyword extraction and web search functionality"""
import asyncio
from app.services.langchain_pipeline import langchain_pipeline
from app.services.web_search import web_searcher
from app.services.ollama_service import ollama_service

async def test_keyword_extraction():
    """Test that keyword extraction works properly"""

    print("🔬 Testing Keyword Extraction and Web Search")
    print("=" * 50)

    # Initialize services
    await ollama_service.initialize()
    await web_searcher.initialize()
    await langchain_pipeline.initialize()

    # Test queries
    test_queries = [
        "AI 기반 고객 서비스 자동화 솔루션",
        "Build a recommendation system for e-commerce platform",
        "데이터 분석을 활용한 마케팅 전략 수립"
    ]

    for query in test_queries:
        print(f"\n📝 Query: {query}")
        print("-" * 40)

        # Extract keywords using LangChain pipeline
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

        try:
            result = await keyword_chain.ainvoke({"request": query})
            keywords = result["text"].strip()
            print(f"✅ Extracted keywords: {keywords}")

            # Try web search with extracted keywords
            print(f"\n🔍 Searching with: {keywords}")
            search_results = await web_searcher.search(keywords, max_results=3)

            if search_results:
                print(f"📊 Found {len(search_results)} results:")
                for i, res in enumerate(search_results[:2], 1):
                    print(f"\n  {i}. {res.title}")
                    print(f"     Source: {res.source}")
                    if res.metadata and res.metadata.get("filtered"):
                        print("     ⚠️ Content was filtered")
            else:
                print("❌ No search results found")

        except Exception as e:
            print(f"❌ Error: {e}")

    # Cleanup
    await ollama_service.cleanup()
    await web_searcher.cleanup()

if __name__ == "__main__":
    asyncio.run(test_keyword_extraction())