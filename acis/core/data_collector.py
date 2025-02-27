"""
Data Collector

This module provides functionality for collecting data from various sources
like news websites, company websites, social media, and industry reports.
"""

import asyncio
import logging
import aiohttp
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
from bs4 import BeautifulSoup
import re

from acis.config.settings import settings
from acis.utils.helpers import sanitize_query, extract_date

# Configure logging
logger = logging.getLogger(__name__)


class DataCollector:
    """Collect data from various external sources."""

    def __init__(self):
        """Initialize the data collector."""
        self.session = None
        self.rate_limits = {
            "news": 5,  # Requests per second
            "social": 2,
            "financial": 1
        }
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    async def __aenter__(self):
        """Context manager entry point."""
        self.session = aiohttp.ClientSession(headers=self.headers)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point."""
        if self.session:
            await self.session.close()
            self.session = None

    async def collect_news(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Collect news articles related to a query.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of news articles with metadata
        """
        # In a real implementation, this would connect to news APIs
        # For this example, we'll simulate results
        
        if not self.session:
            self.session = aiohttp.ClientSession(headers=self.headers)
        
        try:
            # Simulate API request delay
            await asyncio.sleep(1)
            
            # Generate mock news data
            news_sources = [
                "TechCrunch", "Wall Street Journal", "Bloomberg", 
                "CNBC", "Reuters", "Financial Times"
            ]
            
            articles = []
            for i in range(min(max_results, 20)):
                # Create mock article
                article = {
                    "source": news_sources[i % len(news_sources)],
                    "headline": f"New developments in {query.split()[0]} industry - Article {i+1}",
                    "url": f"https://example.com/news/{query.replace(' ', '-')}/{i+1}",
                    "date": (datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - 
                            asyncio.timedelta(days=i % 30)).isoformat(),
                    "content": f"This is a simulated article about {query}. It contains information that would be relevant for competitive intelligence purposes."
                }
                articles.append(article)
            
            return articles[:max_results]
            
        except Exception as e:
            logger.error(f"Error collecting news: {e}")
            return []

    async def collect_financial_data(self, company: str, quarters: int = 4) -> Dict[str, Any]:
        """
        Collect financial data for a company.
        
        Args:
            company: Company name or ticker symbol
            quarters: Number of quarters of data to retrieve
            
        Returns:
            Financial data for the company
        """
        # In a real implementation, this would connect to financial data APIs
        # For this example, we'll simulate results
        
        if not self.session:
            self.session = aiohttp.ClientSession(headers=self.headers)
        
        try:
            # Simulate API request delay
            await asyncio.sleep(1.5)
            
            # Generate mock financial data
            metrics = [
                "Revenue", "Net Income", "Gross Margin", "Operating Margin",
                "EPS", "R&D Expenses", "Marketing Expenses", "Cash Flow"
            ]
            
            financial_data = {
                "company": company,
                "ticker": company[:4].upper(),
                "currency": "USD",
                "metrics": {}
            }
            
            # Current quarter (simulated)
            current_quarter = {
                "year": datetime.now().year,
                "quarter": ((datetime.now().month - 1) // 3) + 1
            }
            
            for metric in metrics:
                financial_data["metrics"][metric] = []
                
                # Base value for this metric
                base_value = 100 + (hash(metric + company) % 900)
                growth_factor = 1.0 + ((hash(metric + company) % 30) / 100)
                
                for i in range(quarters):
                    # Calculate quarter and year (going backward from current)
                    q = current_quarter["quarter"] - i
                    y = current_quarter["year"]
                    
                    while q <= 0:
                        q += 4
                        y -= 1
                    
                    # Generate value with some growth pattern and randomness
                    quarter_value = base_value * (growth_factor ** (quarters - i - 1))
                    quarter_value *= 1.0 + ((hash(metric + company + str(q) + str(y)) % 20) - 10) / 100
                    
                    financial_data["metrics"][metric].append({
                        "year": y,
                        "quarter": q,
                        "value": round(quarter_value, 2)
                    })
            
            return financial_data
            
        except Exception as e:
            logger.error(f"Error collecting financial data: {e}")
            return {
                "company": company,
                "metrics": {}
            }

    async def collect_social_media(self, company: str, days: int = 7) -> Dict[str, Any]:
        """
        Collect social media mentions for a company.
        
        Args:
            company: Company name
            days: Number of days of data to retrieve
            
        Returns:
            Social media data for the company
        """
        # In a real implementation, this would connect to social media APIs
        # For this example, we'll simulate results
        
        if not self.session:
            self.session = aiohttp.ClientSession(headers=self.headers)
        
        try:
            # Simulate API request delay
            await asyncio.sleep(0.8)
            
            # Generate mock social media data
            platforms = ["Twitter", "Reddit", "LinkedIn", "Facebook"]
            
            social_data = {
                "company": company,
                "period": f"Last {days} days",
                "platforms": {}
            }
            
            for platform in platforms:
                # Generate random counts with a slight trend
                base_mentions = 100 + (hash(platform + company) % 900)
                sentiment_base = (hash(platform + company) % 100) / 100
                
                daily_data = []
                for i in range(days):
                    # Calculate date
                    date = (datetime.now() - asyncio.timedelta(days=i)).strftime("%Y-%m-%d")
                    
                    # Generate mentions with some randomness and weekly pattern
                    day_factor = 1.0 - (i % 7) * 0.05
                    mentions = int(base_mentions * day_factor * (1.0 + ((hash(date + platform) % 40) - 20) / 100))
                    
                    # Generate sentiment
                    sentiment = sentiment_base + ((hash(date + platform) % 40) - 20) / 100
                    sentiment = max(-1.0, min(1.0, sentiment))  # Clamp to [-1, 1]
                    
                    daily_data.append({
                        "date": date,
                        "mentions": mentions,
                        "sentiment": round(sentiment, 2),
                        "popular_topics": [
                            f"{company} product",
                            f"{company} news",
                            f"{company} customer service"
                        ][:1 + (hash(date + platform) % 3)]
                    })
                
                social_data["platforms"][platform] = {
                    "total_mentions": sum(d["mentions"] for d in daily_data),
                    "average_sentiment": round(sum(d["sentiment"] for d in daily_data) / len(daily_data), 2),
                    "daily_data": daily_data
                }
            
            return social_data
            
        except Exception as e:
            logger.error(f"Error collecting social media data: {e}")
            return {
                "company": company,
                "platforms": {}
            }

    async def collect_product_data(self, company: str, competitor: bool = False) -> List[Dict[str, Any]]:
        """
        Collect product data for a company.
        
        Args:
            company: Company name
            competitor: Whether this is for a competitor
            
        Returns:
            List of product data
        """
        # In a real implementation, this would scrape websites or use product APIs
        # For this example, we'll simulate results
        
        if not self.session:
            self.session = aiohttp.ClientSession(headers=self.headers)
        
        try:
            # Simulate website scraping delay
            await asyncio.sleep(1.2)
            
            # Generate mock product data
            product_types = ["Software", "Hardware", "Service", "Subscription"]
            
            products = []
            num_products = 3 + (hash(company) % 5)
            
            for i in range(num_products):
                product_type = product_types[hash(company + str(i)) % len(product_types)]
                
                # Generate product details
                product = {
                    "name": f"{company} {product_type} {i+1}",
                    "type": product_type,
                    "description": f"A {product_type.lower()} product by {company} that provides value to customers.",
                    "launch_date": (datetime.now() - asyncio.timedelta(days=(hash(company + str(i)) % 1000))).strftime("%Y-%m-%d"),
                    "pricing": {
                        "amount": 10 * (1 + (hash(company + str(i)) % 100)),
                        "currency": "USD",
                        "model": "one-time" if product_type in ["Hardware"] else "subscription"
                    },
                    "features": [
                        f"Feature {j+1}" for j in range(3 + (hash(company + str(i)) % 5))
                    ],
                    "market_share": round(5 + (hash(company + str(i)) % 20), 1) if competitor else None
                }
                
                products.append(product)
            
            return products
            
        except Exception as e:
            logger.error(f"Error collecting product data: {e}")
            return []

    async def collect_web_content(self, url: str) -> Dict[str, Any]:
        """
        Collect and parse content from a web page.
        
        Args:
            url: URL to scrape
            
        Returns:
            Parsed content from the web page
        """
        if not self.session:
            self.session = aiohttp.ClientSession(headers=self.headers)
        
        try:
            # Make the request
            async with self.session.get(url, timeout=10) as response:
                if response.status != 200:
                    logger.error(f"Error fetching {url}: Status {response.status}")
                    return {"url": url, "error": f"Status {response.status}"}
                
                # Get the content
                html = await response.text()
                
                # Parse with BeautifulSoup
                soup = BeautifulSoup(html, 'lxml')
                
                # Extract useful content
                title = soup.title.text.strip() if soup.title else ""
                
                # Extract main content
                content = ""
                main_tags = soup.find_all(['article', 'main', 'div', 'section'], class_=re.compile('(content|article|main|body)'))
                if main_tags:
                    main_tag = max(main_tags, key=lambda x: len(str(x)))
                    # Remove scripts, styles, and nav elements
                    for tag in main_tag.find_all(['script', 'style', 'nav', 'header', 'footer']):
                        tag.decompose()
                    content = main_tag.get_text(separator=' ', strip=True)
                
                # If no content found, use the body
                if not content and soup.body:
                    content = soup.body.get_text(separator=' ', strip=True)
                
                # Extract date
                date = None
                date_tags = soup.find_all(['time', 'span', 'div', 'p'], class_=re.compile('(date|time|published)'))
                if date_tags:
                    date_text = date_tags[0].get_text(strip=True)
                    date = extract_date(date_text, "")
                
                # Extract structured data
                structured_data = []
                for script in soup.find_all('script', type='application/ld+json'):
                    try:
                        data = json.loads(script.string)
                        structured_data.append(data)
                    except:
                        pass
                
                return {
                    "url": url,
                    "title": title,
                    "content": content[:5000],  # Truncate long content
                    "date": date,
                    "structured_data": structured_data
                }
                
        except Exception as e:
            logger.error(f"Error collecting web content from {url}: {e}")
            return {"url": url, "error": str(e)}

    async def collect_all_competitor_data(self, competitor: str) -> Dict[str, Any]:
        """
        Collect all available data for a competitor.
        
        Args:
            competitor: Competitor name
            
        Returns:
            Comprehensive data about the competitor
        """
        if not self.session:
            self.session = aiohttp.ClientSession(headers=self.headers)
        
        # Prepare tasks for concurrent execution
        tasks = [
            self.collect_news(f"{competitor} news", max_results=15),
            self.collect_financial_data(competitor, quarters=8),
            self.collect_social_media(competitor, days=30),
            self.collect_product_data(competitor, competitor=True)
        ]
        
        # Execute tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        competitor_data = {
            "name": competitor,
            "last_updated": datetime.now().isoformat()
        }
        
        if isinstance(results[0], list):
            competitor_data["news"] = results[0]
        else:
            competitor_data["news"] = []
            logger.error(f"Error collecting news for {competitor}: {results[0]}")
        
        if isinstance(results[1], dict):
            competitor_data["financial"] = results[1]
        else:
            competitor_data["financial"] = {"metrics": {}}
            logger.error(f"Error collecting financial data for {competitor}: {results[1]}")
        
        if isinstance(results[2], dict):
            competitor_data["social"] = results[2]
        else:
            competitor_data["social"] = {"platforms": {}}
            logger.error(f"Error collecting social media data for {competitor}: {results[2]}")
        
        if isinstance(results[3], list):
            competitor_data["products"] = results[3]
        else:
            competitor_data["products"] = []
            logger.error(f"Error collecting product data for {competitor}: {results[3]}")
        
        return competitor_data 