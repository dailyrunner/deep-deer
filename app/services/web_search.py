"""Web search service for external data retrieval"""
import logging
import re
from typing import List, Dict, Any, Optional
import aiohttp
import asyncio
from dataclasses import dataclass

from app.core.config import settings
from app.services.ollama_service import ollama_service

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Web search result"""
    title: str
    url: str
    snippet: str
    source: str = "web"
    metadata: Optional[Dict[str, Any]] = None


class WebSearcher:
    """Web search service with sensitive info filtering"""

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.is_ready = False

    async def initialize(self):
        """Initialize web searcher"""
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
            self.is_ready = True
            logger.info("Web searcher initialized")
        except Exception as e:
            logger.error(f"Failed to initialize web searcher: {e}")
            self.is_ready = False

    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
            self.session = None
        self.is_ready = False

    async def search(
        self,
        query: str,
        max_results: int = 10,
        filter_sensitive: bool = True
    ) -> List[SearchResult]:
        """Perform web search"""
        if not self.is_ready:
            logger.error("Web searcher is not ready")
            return []

        results = []

        # Try Naver API first if credentials are available
        if settings.naver_client_id and settings.naver_client_secret:
            try:
                results = await self._search_naver(query, max_results)
            except Exception as e:
                logger.error(f"Naver search failed: {e}")

        # Fallback to DuckDuckGo if Naver fails or no results
        if not results:
            try:
                results = await self._search_duckduckgo(query, max_results)
            except Exception as e:
                logger.error(f"DuckDuckGo search failed: {e}")

        # If we have an API key, try other search engines
        if not results and settings.search_api_key:
            try:
                # You can add Google Custom Search, Bing, etc. here
                pass
            except Exception as e:
                logger.error(f"Alternative search failed: {e}")

        # Filter sensitive information if requested
        if filter_sensitive and results:
            results = await self._filter_sensitive_results(results)

        return results

    async def _search_naver(self, query: str, max_results: int) -> List[SearchResult]:
        """Search using Naver Search API"""
        results = []

        try:
            import re

            headers = {
                "X-Naver-Client-Id": settings.naver_client_id,
                "X-Naver-Client-Secret": settings.naver_client_secret,
            }

            def strip_tags(s: str) -> str:
                """Remove HTML tags from string"""
                return re.sub(r"</?b>", "", s or "")

            # Try different search endpoints with proper error handling
            search_endpoints = [
                {"type": "news", "url": "https://openapi.naver.com/v1/search/news.json"},
                {"type": "blog", "url": "https://openapi.naver.com/v1/search/blog.json"},
                {"type": "webkr", "url": "https://openapi.naver.com/v1/search/webkr.json"}
            ]

            want = max_results
            display = min(20, want)  # Max 20 per request
            start = 1

            for endpoint in search_endpoints:
                if len(results) >= want:
                    break

                try:
                    # Calculate how many more results we need
                    remaining = want - len(results)
                    current_display = min(display, remaining)

                    params = {
                        "query": query,
                        "display": current_display,
                        "start": start,
                        "sort": "sim"  # Sort by relevance
                    }

                    async with self.session.get(endpoint["url"], headers=headers, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            items = data.get("items", [])

                            logger.info(f"Naver {endpoint['type']} search: {len(items)} items retrieved")

                            for item in items:
                                if len(results) >= want:
                                    break

                                # Clean HTML tags
                                clean_title = strip_tags(item.get("title", ""))
                                clean_description = strip_tags(item.get("description", ""))

                                results.append(SearchResult(
                                    title=clean_title,
                                    url=item.get("link", ""),
                                    snippet=clean_description,
                                    source=f"naver_{endpoint['type']}",
                                    metadata={
                                        "search_type": endpoint['type'],
                                        "pub_date": item.get("pubDate", ""),
                                        "original_title": item.get("title", "")
                                    }
                                ))

                        elif response.status == 401:
                            logger.error(f"Naver API authentication failed for {endpoint['type']}")
                            break  # Don't try other endpoints if auth fails
                        else:
                            logger.warning(f"Naver {endpoint['type']} search returned status {response.status}")

                except Exception as e:
                    logger.error(f"Error in Naver {endpoint['type']} search: {e}")
                    continue  # Try next endpoint

            logger.info(f"Found {len(results)} total results from Naver for query: {query}")

        except Exception as e:
            logger.error(f"Error in Naver search: {e}")

        return results

    async def _search_duckduckgo(self, query: str, max_results: int) -> List[SearchResult]:
        """Search using DuckDuckGo Search API"""
        results = []

        try:
            # Use the synchronous DDGS (newer version doesn't have AsyncDDGS)
            from duckduckgo_search import DDGS

            # Run in thread pool to avoid blocking
            import asyncio
            from functools import partial

            def sync_search():
                with DDGS() as ddgs:
                    search_results = list(ddgs.text(query, max_results=max_results))
                    return search_results

            # Run synchronous function in executor
            loop = asyncio.get_event_loop()
            search_results = await loop.run_in_executor(None, sync_search)

            for result in search_results:
                results.append(SearchResult(
                    title=result.get("title", ""),
                    url=result.get("href", "") or result.get("link", ""),
                    snippet=result.get("body", "") or result.get("snippet", ""),
                    source="duckduckgo"
                ))

            logger.info(f"Found {len(results)} results for query: {query}")

        except Exception as e:
            logger.error(f"Error in DuckDuckGo search: {e}")

        # Fallback to using a simple web search simulation
        if not results:
            results = self._simulate_search_results(query, max_results)

        return results

    def _simulate_search_results(self, query: str, max_results: int) -> List[SearchResult]:
        """Simulate search results for testing/demo purposes"""
        simulated = []

        # Generate some realistic-looking results based on the query
        keywords = query.lower().split()

        templates = [
            {
                "title": f"Understanding {query}: A Comprehensive Guide",
                "snippet": f"Learn everything about {query} with our detailed guide covering best practices and implementation strategies.",
                "url": f"https://example.com/guides/{'-'.join(keywords)}"
            },
            {
                "title": f"Top 10 {query} Solutions for 2024",
                "snippet": f"Discover the latest trends and solutions related to {query} that are transforming businesses today.",
                "url": f"https://techblog.example.com/{'-'.join(keywords)}-solutions"
            },
            {
                "title": f"{query} Best Practices and Case Studies",
                "snippet": f"Real-world examples and case studies showing successful implementation of {query} in various industries.",
                "url": f"https://casestudies.example.com/{'-'.join(keywords)}"
            },
            {
                "title": f"How to Implement {query} in Your Organization",
                "snippet": f"Step-by-step guide to implementing {query} with practical tips and common pitfalls to avoid.",
                "url": f"https://enterprise.example.com/implement-{'-'.join(keywords)}"
            },
            {
                "title": f"{query}: Market Analysis and Trends",
                "snippet": f"In-depth market analysis of {query} including growth projections and competitive landscape.",
                "url": f"https://marketresearch.example.com/{'-'.join(keywords)}-analysis"
            }
        ]

        for i, template in enumerate(templates[:max_results]):
            simulated.append(SearchResult(
                title=template["title"],
                url=template["url"],
                snippet=template["snippet"],
                source="simulated",
                metadata={"relevance_score": 1.0 - (i * 0.1)}
            ))

        return simulated

    async def _filter_sensitive_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Filter out sensitive information from search results"""
        filtered_results = []

        for result in results:
            try:
                # Check if the content contains sensitive information
                content_to_check = f"{result.title} {result.snippet}"

                if await self._contains_sensitive_info(content_to_check):
                    # Filter the content
                    filtered_title = await ollama_service.filter_sensitive_info(result.title)
                    filtered_snippet = await ollama_service.filter_sensitive_info(result.snippet)

                    filtered_results.append(SearchResult(
                        title=filtered_title,
                        url=result.url,
                        snippet=filtered_snippet,
                        source=result.source,
                        metadata={
                            **(result.metadata or {}),
                            "filtered": True
                        }
                    ))
                else:
                    filtered_results.append(result)

            except Exception as e:
                logger.error(f"Error filtering result: {e}")
                # If filtering fails, exclude the result to be safe
                continue

        return filtered_results

    async def _contains_sensitive_info(self, text: str) -> bool:
        """Check if text contains sensitive information"""
        text_lower = text.lower()

        # Check for sensitive patterns
        sensitive_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
            r'\b\d{16}\b',  # Credit card pattern
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b(?:password|pwd|token|api[_\s]?key|secret)\s*[:=]\s*\S+',  # Credentials
        ]

        for pattern in sensitive_patterns:
            if re.search(pattern, text_lower):
                return True

        # Check for sensitive keywords
        for keyword in settings.sensitive_fields_list:
            if keyword.lower() in text_lower:
                # More sophisticated check: keyword with value
                if re.search(f'{keyword.lower()}\\s*[:=]\\s*\\S+', text_lower):
                    return True

        return False

    async def search_and_summarize(
        self,
        query: str,
        context: str = "",
        max_results: int = 5
    ) -> Dict[str, Any]:
        """Search web and provide summarized insights"""
        # Perform search
        search_results = await self.search(query, max_results=max_results)

        if not search_results:
            return {
                "query": query,
                "results": [],
                "summary": "No search results found.",
                "insights": []
            }

        # Prepare content for summarization
        search_content = "\n\n".join([
            f"Title: {r.title}\nURL: {r.url}\nSnippet: {r.snippet}"
            for r in search_results
        ])

        # Generate summary using Ollama
        summary_prompt = f"""Based on the following search results and context, provide a comprehensive summary:

Context (Internal Data):
{context[:1500] if context else "No internal context provided"}

Search Results:
{search_content}

Query: {query}

Please provide:
1. A brief summary of the findings
2. Key insights relevant to the query
3. How this relates to the internal context (if provided)
4. Potential opportunities or recommendations"""

        summary = await ollama_service.generate(
            prompt=summary_prompt,
            temperature=0.7,
            max_tokens=1000
        )

        # Extract key insights
        insights = self._extract_insights(summary)

        return {
            "query": query,
            "results": [
                {
                    "title": r.title,
                    "url": r.url,
                    "snippet": r.snippet,
                    "source": r.source
                }
                for r in search_results
            ],
            "summary": summary,
            "insights": insights,
            "filtered": any(
                r.metadata and r.metadata.get("filtered", False)
                for r in search_results
            )
        }

    def _extract_insights(self, text: str) -> List[str]:
        """Extract key insights from text"""
        insights = []

        # Look for numbered points or bullet points
        patterns = [
            r'^\d+\.\s+(.+)$',  # Numbered list
            r'^[-•]\s+(.+)$',   # Bullet points
            r'^(?:Key insight|Important|Note):\s+(.+)$',  # Key phrases
        ]

        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            for pattern in patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    insight = match.group(1).strip()
                    if len(insight) > 20:  # Filter out very short insights
                        insights.append(insight)
                    break

        # Limit to top 5 insights
        return insights[:5]


# Global instance
web_searcher = WebSearcher()