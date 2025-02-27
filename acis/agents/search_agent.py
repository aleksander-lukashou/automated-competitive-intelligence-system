"""
Search Agent

This module implements the web search agent for retrieving competitive intelligence data.
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional
import aiohttp
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from acis.config.settings import settings
from acis.utils.helpers import sanitize_query, extract_date

# Configure logging
logger = logging.getLogger(__name__)


class SearchAgent:
    """Agent for performing web searches and extracting competitive intelligence."""

    def __init__(self):
        """Initialize the search agent."""
        self.llm = OpenAI(
            temperature=0.2,
            model_name=settings.llm_model,
            openai_api_key=settings.openai_api_key
        )
        
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
        
        self.sentiment_chain = LLMChain(llm=self.llm, prompt=self.sentiment_prompt)

    async def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Perform a web search for competitive intelligence.
        
        Args:
            query: The search query
            max_results: Maximum number of results to return
            
        Returns:
            List of search results with metadata
        """
        # Step 1: Optimize search query using LLM
        optimized_query = await self._optimize_query(query)
        
        # Step 2: Perform search using external API
        raw_results = await self._perform_search(optimized_query, max_results)
        
        # Step 3: Process and enrich results
        enriched_results = await self._process_results(raw_results)
        
        return enriched_results

    async def _optimize_query(self, query: str) -> str:
        """
        Optimize the search query using LLM.
        
        Args:
            query: The original query
            
        Returns:
            Optimized search query
        """
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
        Perform search using external API.
        
        Args:
            query: The search query
            max_results: Maximum number of results
            
        Returns:
            Raw search results
        """
        # Placeholder for actual search API integration
        # In a real implementation, this would connect to Google/Bing/etc.
        
        # Mock implementation for architecture demonstration
        async with aiohttp.ClientSession() as session:
            try:
                # Simulating API call
                await asyncio.sleep(1)
                
                # Mock results
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
            except Exception as e:
                logger.error(f"Error performing search: {e}")
                return []

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
                    "summary": summary.strip(),
                    "sentiment": sentiment
                }
                
                enriched_results.append(enriched_result)
            except Exception as e:
                logger.error(f"Error processing result: {e}")
                enriched_results.append(result)
        
        return enriched_results 