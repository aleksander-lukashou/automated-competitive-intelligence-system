"""
Search Agent

This module implements the web search agent for retrieving competitive intelligence data.
"""

import asyncio
import json
import logging
import re
import os
from typing import List, Dict, Any, Optional
import aiohttp
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from acis.config.settings import settings
from acis.utils.helpers import sanitize_query, extract_date
from acis.utils.data_storage import store_search_results, store_insight, log_activity

# Configure logging
logger = logging.getLogger(__name__)


class SearchAgent:
    """Agent for performing web searches and extracting competitive intelligence."""

    def __init__(self):
        """Initialize the search agent."""
        # Try to initialize LLM with OpenAI API
        try:
            openai_api_key = settings.openai_api_key
            if not openai_api_key or openai_api_key == "sk-your-openai-api-key-here":
                logger.warning("OpenAI API key not properly configured. Using mock mode.")
                self.use_mock = True
            else:
                # Initialize LLM with updated ChatOpenAI implementation
                self.llm = ChatOpenAI(
                    temperature=0.2,
                    model_name=settings.llm_model,
                    openai_api_key=openai_api_key
                )
                self.use_mock = False
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI client: {e}. Using mock mode.")
            self.use_mock = True
            
        # Direct check of environment variables
        logger.info(f"Env GOOGLE_CX direct: {os.environ.get('GOOGLE_CX', 'Not found in env')}")
        logger.info(f"Settings google_search_engine_id: {settings.google_search_engine_id}")
            
        # Define search prompt
        self.search_prompt = PromptTemplate(
            input_variables=["query"],
            template="""
            Generate an effective search query to find competitive intelligence about the following:
            
            {query}
            
            Your search query should focus on recent developments, product launches, financial performance,
            or strategic moves. Make the query specific and targeted.
            
            Search query:
            """
        )
        
        if not self.use_mock:
            self.search_chain = LLMChain(llm=self.llm, prompt=self.search_prompt)
        
        # Define summarization prompt
        self.summarization_prompt = PromptTemplate(
            input_variables=["content"],
            template="""
            Extract key competitive intelligence information from the following content.
            Focus on business strategy, product launches, financial performance, and market positioning.
            Provide a concise summary of the most important points:
            
            {content}
            
            Summary:
            """
        )
        
        if not self.use_mock:
            self.summarization_chain = LLMChain(llm=self.llm, prompt=self.summarization_prompt)
        
        # Define sentiment analysis prompt
        self.sentiment_prompt = PromptTemplate(
            input_variables=["summary"],
            template="""
            Analyze the sentiment of the following business news summary.
            Return a score between -1 (extremely negative) and 1 (extremely positive).
            Only return the numerical score.
            
            {summary}
            
            Sentiment score:
            """
        )
        
        if not self.use_mock:
            self.sentiment_chain = LLMChain(llm=self.llm, prompt=self.sentiment_prompt)
        
        # Define insight extraction prompt
        self.insight_prompt = PromptTemplate(
            input_variables=["content"],
            template="""
            Extract key competitive intelligence insights from the following content.
            Each insight should be a specific, actionable piece of intelligence about a company.
            Format your response as a list of insights in this format:
            
            Company: [Company Name]
            Insight: [Concise insight statement]
            
            Content:
            {content}
            
            Insights:
            """
        )
        
        if not self.use_mock:
            self.insight_chain = LLMChain(llm=self.llm, prompt=self.insight_prompt)

    async def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Perform a web search for competitive intelligence.
        
        Args:
            query: The search query
            max_results: Maximum number of results to return
            
        Returns:
            List of search results with metadata
        """
        # Log the search activity
        log_activity("Search", f"Query: {query}")
        
        # Step 1: Optimize search query using LLM
        optimized_query = await self._optimize_query(query)
        
        # Step 2: Perform search using external API
        raw_results = await self._perform_search(optimized_query, max_results)
        
        # Step 3: Process and enrich results
        enriched_results = await self._process_results(raw_results)
        
        # Step 4: Store the search results
        store_search_results(query, enriched_results)
        
        # Step 5: Extract and store insights
        await self._extract_insights(enriched_results)
        
        return enriched_results

    async def _optimize_query(self, query: str) -> str:
        """
        Optimize the search query using LLM.
        
        Args:
            query: The original query
            
        Returns:
            Optimized search query
        """
        if self.use_mock:
            # In mock mode, just return the original query
            return query
        
        try:
            # Use LLM to optimize query
            response = await asyncio.to_thread(
                self.search_chain.run,
                query=query
            )
            
            # Clean and return optimized query
            optimized_query = sanitize_query(response.strip())
            logger.info(f"Optimized query: {optimized_query}")
            
            return optimized_query
        except Exception as e:
            logger.error(f"Error optimizing query: {e}")
            return query

    async def _perform_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """
        Perform search using Google Custom Search API.
        
        Args:
            query: The search query
            max_results: Maximum number of results
            
        Returns:
            Raw search results
        """
        # Google Custom Search API URL
        search_url = "https://www.googleapis.com/customsearch/v1"
        
        # Define hardcoded values for critical settings
        hardcoded_cx = "721e1b1a4c4a84218"  # Your Google Custom Search Engine ID
        
        # Get values from various sources
        # 1. Try settings from the settings object
        # 2. Try environment variables directly
        # 3. Try hardcoded values as last resort
        google_api_key = settings.google_api_key
        google_cx = settings.google_search_engine_id or os.environ.get('GOOGLE_CX', '') or hardcoded_cx
        
        # Debug logging
        logger.info(f"Search query: {query}")
        logger.info(f"Using Google API Key: {google_api_key[:5]}... (length: {len(google_api_key) if google_api_key else 0})")
        logger.info(f"Using Search Engine ID: {google_cx} (length: {len(google_cx) if google_cx else 0})")
        
        # Check if API key is configured
        if not google_api_key:
            logger.error("Google API key not configured")
            return self._get_mock_results(query, max_results)
        
        # Check if Search Engine ID is configured
        if not google_cx:
            logger.error("Google Search Engine ID not configured")
            return self._get_mock_results(query, max_results)
        
        # Prepare parameters
        params = {
            "q": query,
            "key": google_api_key,
            "cx": google_cx,
            "num": min(max_results, 10)  # API limits to 10 results per request
        }
        
        results = []
        
        async with aiohttp.ClientSession() as session:
            try:
                # Make the API request
                logger.info(f"Making request to Google API: {search_url} with params: q={query}, cx={google_cx[:5]}...")
                async with session.get(search_url, params=params) as response:
                    response_text = await response.text()
                    
                    if response.status != 200:
                        logger.error(f"Error from Google API (status {response.status}): {response_text}")
                        return self._get_mock_results(query, max_results)
                    
                    # Try to parse the response as JSON
                    try:
                        data = json.loads(response_text)
                        logger.info(f"Response received from Google API: {len(response_text)} bytes")
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing Google API response as JSON: {str(e)}")
                        logger.error(f"Response was: {response_text[:200]}...")
                        return self._get_mock_results(query, max_results)
                    
                    # Process search results
                    if "items" in data:
                        logger.info(f"Found {len(data['items'])} search results")
                        for item in data["items"]:
                            # Extract relevant information
                            result = {
                                "source": item.get("displayLink", "Unknown"),
                                "headline": item.get("title", ""),
                                "url": item.get("link", ""),
                                "content": item.get("snippet", "")
                            }
                            results.append(result)
                    else:
                        logger.warning(f"No items found in Google search response. Response keys: {list(data.keys())}")
                        if "error" in data:
                            logger.error(f"Google API error: {data['error'].get('message', 'Unknown error')}")
                
                return results[:max_results]
                
            except Exception as e:
                logger.error(f"Error performing search: {str(e)}")
                return self._get_mock_results(query, max_results)
    
    def _get_mock_results(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """
        Generate mock search results when real search fails.
        
        Args:
            query: The search query
            max_results: Maximum number of results
        
        Returns:
            Mock search results
        """
        logger.warning("Using mock search results due to API error or missing configuration")
        
        # Mock results for fallback
        mock_results = [
            {
                "source": "TechCrunch",
                "headline": f"New developments in {query.split()[0]} industry",
                "url": f"https://techcrunch.com/article/{query.replace(' ', '-')}",
                "content": f"Recent news about {query} indicates significant market movement. Companies are investing in new technologies and expanding their product offerings."
            },
            {
                "source": "Forbes",
                "headline": f"Financial analysis of {query.split()[0]}",
                "url": f"https://forbes.com/article/{query.replace(' ', '-')}",
                "content": f"Financial experts suggest that {query} market is growing at 15% annually. Key players are reporting strong quarterly results."
            }
        ]
        
        # Limit results
        return mock_results[:max_results]

    async def _process_results(self, raw_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process and enrich search results.
        
        Args:
            raw_results: Raw search results
            
        Returns:
            Processed and enriched results
        """
        enriched_results = []
        
        for result in raw_results:
            try:
                # Extract content
                content = result.get("content", "")
                
                if self.use_mock:
                    # In mock mode, generate fake summaries and sentiment
                    summary = f"Summary of {result.get('headline', '')}: {content[:100]}..."
                    sentiment = 0.2 if "positive" in content.lower() else -0.2 if "negative" in content.lower() else 0.0
                else:
                    # Generate summary
                    summary = await asyncio.to_thread(
                        self.summarization_chain.run,
                        content=content
                    )
                    
                    # Analyze sentiment
                    sentiment_str = await asyncio.to_thread(
                        self.sentiment_chain.run,
                        summary=summary
                    )
                    
                    try:
                        sentiment = float(sentiment_str.strip())
                    except ValueError:
                        sentiment = 0.0
                
                # Extract date if available
                date = extract_date(result.get("headline", ""), result.get("content", ""))
                
                # Create enriched result
                enriched_result = {
                    "source": result.get("source", "Unknown"),
                    "headline": result.get("headline", ""),
                    "url": result.get("url", ""),
                    "date": date,
                    "summary": summary.strip() if not self.use_mock else summary,
                    "sentiment": sentiment
                }
                
                enriched_results.append(enriched_result)
            except Exception as e:
                logger.error(f"Error processing result: {e}")
                enriched_results.append(result)
        
        return enriched_results
    
    async def _extract_insights(self, results: List[Dict[str, Any]]) -> None:
        """
        Extract and store insights from search results.
        
        Args:
            results: Processed search results
        """
        if not results:
            return
        
        # Combine all summaries for insight extraction
        combined_content = "\n\n".join([
            f"Source: {r.get('source', 'Unknown')}\n"
            f"Headline: {r.get('headline', '')}\n"
            f"Summary: {r.get('summary', '')}"
            for r in results
        ])
        
        if self.use_mock:
            # In mock mode, generate some mock insights
            companies = ["Tesla", "Apple", "Google", "Microsoft"]
            insights = [
                {"competitor": "Tesla", "content": "Launching new electric vehicle model next quarter", "sentiment": 0.7},
                {"competitor": "Apple", "content": "Experiencing supply chain challenges in Asian markets", "sentiment": -0.3},
                {"competitor": "Google", "content": "Increased investment in AI research and development", "sentiment": 0.5}
            ]
            
            # Store mock insights
            for insight in insights:
                store_insight(
                    competitor=insight["competitor"],
                    content=insight["content"],
                    sentiment=insight["sentiment"],
                    source="Search (Mock)"
                )
                
                # Log activity
                log_activity("Insight", f"New insight about {insight['competitor']}: {insight['content']}")
                
            return
        
        try:
            # Extract insights
            insights_text = await asyncio.to_thread(
                self.insight_chain.run,
                content=combined_content
            )
            
            # Parse insights
            insight_blocks = re.split(r'\n\s*\n', insights_text)
            
            for block in insight_blocks:
                if not block.strip():
                    continue
                
                company_match = re.search(r'Company:\s*(.+?)(?:\n|$)', block)
                insight_match = re.search(r'Insight:\s*(.+?)(?:\n|$)', block)
                
                if company_match and insight_match:
                    company = company_match.group(1).strip()
                    insight_content = insight_match.group(1).strip()
                    
                    # Calculate sentiment for this insight
                    sentiment_str = await asyncio.to_thread(
                        self.sentiment_chain.run,
                        summary=insight_content
                    )
                    
                    try:
                        sentiment = float(sentiment_str.strip())
                    except ValueError:
                        sentiment = 0.0
                    
                    # Store the insight
                    store_insight(
                        competitor=company,
                        content=insight_content,
                        sentiment=sentiment,
                        source="Search"
                    )
                    
                    # Log activity
                    log_activity("Insight", f"New insight about {company}: {insight_content[:50]}...")
        
        except Exception as e:
            logger.error(f"Error extracting insights: {e}") 