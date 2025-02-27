"""
Data Processor

This module implements data processing pipelines for competitive intelligence data.
It handles extraction, transformation, and loading of data for analysis.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
import re
import pandas as pd
import numpy as np

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from acis.config.settings import settings
from acis.core.data_collector import DataCollector
from acis.utils.helpers import sentiment_to_label, calculate_growth_rate

# Configure logging
logger = logging.getLogger(__name__)


class DataProcessor:
    """Process and transform raw data into structured format for analysis."""

    def __init__(self):
        """Initialize the data processor."""
        self.llm = OpenAI(
            temperature=0.1,
            model_name=settings.llm_model,
            openai_api_key=settings.openai_api_key
        )
        
        # Define text summarization prompt
        self.summary_prompt = PromptTemplate(
            input_variables=["text", "max_words"],
            template="""
            Summarize the following text in {max_words} words or less, 
            focusing on the most relevant information for competitive intelligence:
            
            {text}
            
            Concise summary:
            """
        )
        
        self.summary_chain = LLMChain(llm=self.llm, prompt=self.summary_prompt)
        
        # Define entity extraction prompt
        self.entity_prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            Extract the following entities from the text below:
            - Companies
            - Products
            - Technologies
            - People (executives, key figures)
            - Locations
            - Dates and time references
            
            For each entity, provide the type and the exact text.
            Format your response as a JSON array of objects with "type" and "text" fields.
            
            Text:
            {text}
            
            Extracted entities (JSON array):
            """
        )
        
        self.entity_chain = LLMChain(llm=self.llm, prompt=self.entity_prompt)
        
        # Define sentiment analysis prompt
        self.sentiment_prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            Analyze the sentiment of the following text in the context of business and competitive intelligence.
            Return a numerical score between -1 (extremely negative) and 1 (extremely positive).
            
            Text:
            {text}
            
            Sentiment score (just the number between -1 and 1):
            """
        )
        
        self.sentiment_chain = LLMChain(llm=self.llm, prompt=self.sentiment_prompt)

    async def process_news_data(self, news_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process raw news data.
        
        Args:
            news_data: List of news articles
            
        Returns:
            Processed news data with summaries, sentiment, and entities
        """
        processed_news = []
        
        for article in news_data:
            try:
                # Extract the content
                content = article.get("content", "")
                
                if not content:
                    processed_news.append(article)
                    continue
                
                # Process in parallel
                summary_task = self.summarize_text(content, max_words=100)
                entities_task = self.extract_entities(content)
                sentiment_task = self.analyze_sentiment(content)
                
                # Await results
                summary, entities, sentiment = await asyncio.gather(
                    summary_task, entities_task, sentiment_task
                )
                
                # Create processed article
                processed_article = article.copy()
                processed_article["summary"] = summary
                processed_article["entities"] = entities
                processed_article["sentiment"] = sentiment
                processed_article["sentiment_label"] = sentiment_to_label(sentiment)
                
                processed_news.append(processed_article)
                
            except Exception as e:
                logger.error(f"Error processing news article: {e}")
                processed_news.append(article)
        
        return processed_news

    async def process_financial_data(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process raw financial data.
        
        Args:
            financial_data: Raw financial data for a company
            
        Returns:
            Processed financial data with calculations and insights
        """
        try:
            processed_data = financial_data.copy()
            
            # Add calculated metrics dictionary
            processed_data["calculated_metrics"] = {}
            
            # Extract metrics
            metrics = financial_data.get("metrics", {})
            
            # Calculate growth rates for each metric
            for metric_name, values in metrics.items():
                if len(values) < 2:
                    continue
                
                # Sort by year and quarter in descending order
                sorted_values = sorted(
                    values, 
                    key=lambda x: (x["year"], x["quarter"]), 
                    reverse=True
                )
                
                # Calculate quarter-over-quarter growth rates
                growth_rates = []
                for i in range(len(sorted_values) - 1):
                    current = sorted_values[i]
                    previous = sorted_values[i + 1]
                    
                    growth = calculate_growth_rate(
                        current.get("value", 0), 
                        previous.get("value", 0)
                    )
                    
                    growth_rates.append({
                        "from_year": previous["year"],
                        "from_quarter": previous["quarter"],
                        "to_year": current["year"],
                        "to_quarter": current["quarter"],
                        "growth_rate": round(growth, 2)
                    })
                
                # Add growth rates to calculated metrics
                processed_data["calculated_metrics"][f"{metric_name}_growth"] = growth_rates
                
                # Calculate year-over-year growth if available
                if len(sorted_values) >= 4:
                    yoy_growth_rates = []
                    for i in range(len(sorted_values) - 4):
                        current = sorted_values[i]
                        previous = sorted_values[i + 4]
                        
                        growth = calculate_growth_rate(
                            current.get("value", 0), 
                            previous.get("value", 0)
                        )
                        
                        yoy_growth_rates.append({
                            "from_year": previous["year"],
                            "from_quarter": previous["quarter"],
                            "to_year": current["year"],
                            "to_quarter": current["quarter"],
                            "growth_rate": round(growth, 2)
                        })
                    
                    # Add YoY growth rates
                    processed_data["calculated_metrics"][f"{metric_name}_yoy_growth"] = yoy_growth_rates
            
            # Calculate compound annual growth rate if enough data
            for metric_name, values in metrics.items():
                if len(values) >= 4:
                    sorted_values = sorted(
                        values, 
                        key=lambda x: (x["year"], x["quarter"]), 
                        reverse=True
                    )
                    
                    first = sorted_values[-1]
                    last = sorted_values[0]
                    
                    periods = len(sorted_values) - 1
                    
                    if first.get("value", 0) > 0 and periods > 0:
                        cagr = (
                            (last.get("value", 0) / first.get("value", 0)) ** (1 / periods) - 1
                        ) * 100
                        
                        processed_data["calculated_metrics"][f"{metric_name}_cagr"] = round(cagr, 2)
            
            # Add summary statistics
            for metric_name, values in metrics.items():
                if not values:
                    continue
                
                metric_values = [v.get("value", 0) for v in values]
                
                processed_data["calculated_metrics"][f"{metric_name}_stats"] = {
                    "mean": round(np.mean(metric_values), 2),
                    "median": round(np.median(metric_values), 2),
                    "min": round(min(metric_values), 2),
                    "max": round(max(metric_values), 2),
                    "std_dev": round(np.std(metric_values), 2),
                }
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing financial data: {e}")
            return financial_data

    async def process_social_media_data(self, social_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process raw social media data.
        
        Args:
            social_data: Raw social media data
            
        Returns:
            Processed social media data with trends and insights
        """
        try:
            processed_data = social_data.copy()
            
            # Add trends dictionary
            processed_data["trends"] = {}
            
            # Process each platform
            for platform, data in social_data.get("platforms", {}).items():
                daily_data = data.get("daily_data", [])
                
                if not daily_data:
                    continue
                
                # Sort by date
                sorted_data = sorted(daily_data, key=lambda x: x.get("date", ""), reverse=True)
                
                # Calculate mention trends (7-day moving average if enough data)
                if len(sorted_data) >= 7:
                    mention_trends = []
                    sentiment_trends = []
                    
                    for i in range(len(sorted_data) - 6):
                        window = sorted_data[i:i+7]
                        avg_mentions = sum(d.get("mentions", 0) for d in window) / 7
                        avg_sentiment = sum(d.get("sentiment", 0) for d in window) / 7
                        
                        mention_trends.append({
                            "date": sorted_data[i]["date"],
                            "value": round(avg_mentions, 1)
                        })
                        
                        sentiment_trends.append({
                            "date": sorted_data[i]["date"],
                            "value": round(avg_sentiment, 2)
                        })
                    
                    # Store trends
                    if platform not in processed_data["trends"]:
                        processed_data["trends"][platform] = {}
                    
                    processed_data["trends"][platform]["mention_trend"] = mention_trends
                    processed_data["trends"][platform]["sentiment_trend"] = sentiment_trends
                
                # Extract popular topics
                topics = {}
                for day in sorted_data:
                    for topic in day.get("popular_topics", []):
                        topics[topic] = topics.get(topic, 0) + 1
                
                # Sort topics by frequency
                sorted_topics = sorted(
                    [{"topic": k, "count": v} for k, v in topics.items()],
                    key=lambda x: x["count"],
                    reverse=True
                )
                
                # Store popular topics
                if platform not in processed_data["trends"]:
                    processed_data["trends"][platform] = {}
                    
                processed_data["trends"][platform]["popular_topics"] = sorted_topics
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing social media data: {e}")
            return social_data

    async def process_product_data(self, product_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process raw product data.
        
        Args:
            product_data: List of product data
            
        Returns:
            Processed product data with categorization and insights
        """
        try:
            # Convert to DataFrame for easier processing
            df = pd.DataFrame(product_data)
            
            # Init results dictionary
            results = {
                "products": product_data,
                "categories": {},
                "pricing": {},
                "features": [],
                "timeline": []
            }
            
            # Categorize by product type
            if "type" in df.columns:
                for product_type in df["type"].unique():
                    type_products = df[df["type"] == product_type]
                    results["categories"][product_type] = {
                        "count": len(type_products),
                        "products": type_products["name"].tolist()
                    }
            
            # Analyze pricing
            if "pricing" in df.columns:
                pricing_models = {}
                price_ranges = {
                    "low": [],
                    "medium": [],
                    "high": []
                }
                
                for _, product in df.iterrows():
                    if isinstance(product.get("pricing"), dict):
                        model = product["pricing"].get("model", "unknown")
                        pricing_models[model] = pricing_models.get(model, 0) + 1
                        
                        # Categorize price
                        amount = product["pricing"].get("amount", 0)
                        if amount < 50:
                            price_ranges["low"].append(product["name"])
                        elif amount < 200:
                            price_ranges["medium"].append(product["name"])
                        else:
                            price_ranges["high"].append(product["name"])
                
                results["pricing"]["models"] = [
                    {"model": k, "count": v} for k, v in pricing_models.items()
                ]
                results["pricing"]["ranges"] = price_ranges
            
            # Extract all features
            if "features" in df.columns:
                all_features = []
                for _, product in df.iterrows():
                    if isinstance(product.get("features"), list):
                        for feature in product["features"]:
                            all_features.append({
                                "feature": feature,
                                "product": product["name"]
                            })
                
                results["features"] = all_features
            
            # Create product timeline
            if "launch_date" in df.columns:
                timeline = []
                for _, product in df.iterrows():
                    if product.get("launch_date"):
                        timeline.append({
                            "date": product["launch_date"],
                            "product": product["name"],
                            "event": "Launch",
                            "type": product.get("type", "Unknown")
                        })
                
                # Sort by date
                timeline.sort(key=lambda x: x["date"])
                results["timeline"] = timeline
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing product data: {e}")
            return {"products": product_data, "error": str(e)}

    async def summarize_text(self, text: str, max_words: int = 100) -> str:
        """
        Summarize text using LLM.
        
        Args:
            text: Text to summarize
            max_words: Maximum words in summary
            
        Returns:
            Summarized text
        """
        try:
            # Limit input text to avoid token issues
            if len(text) > 10000:
                text = text[:10000]
            
            # Generate summary using LLM
            summary = await asyncio.to_thread(
                self.summary_chain.run,
                text=text,
                max_words=max_words
            )
            
            return summary.strip()
            
        except Exception as e:
            logger.error(f"Error summarizing text: {e}")
            return text[:max_words * 5] + "..."  # Fallback to simple truncation

    async def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """
        Extract entities from text using LLM.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            List of entities with type and text
        """
        try:
            # Limit input text to avoid token issues
            if len(text) > 10000:
                text = text[:10000]
            
            # Extract entities using LLM
            result = await asyncio.to_thread(
                self.entity_chain.run,
                text=text
            )
            
            # Parse JSON result
            try:
                entities = json.loads(result.strip())
                return entities
            except json.JSONDecodeError:
                logger.error(f"Failed to parse entity JSON: {result}")
                return []
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []

    async def analyze_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of text using LLM.
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment score (-1 to 1)
        """
        try:
            # Limit input text to avoid token issues
            if len(text) > 5000:
                text = text[:5000]
            
            # Analyze sentiment using LLM
            result = await asyncio.to_thread(
                self.sentiment_chain.run,
                text=text
            )
            
            # Parse result as float
            try:
                sentiment = float(result.strip())
                return max(-1.0, min(1.0, sentiment))  # Clamp to [-1, 1]
            except ValueError:
                logger.error(f"Failed to parse sentiment as float: {result}")
                return 0.0
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return 0.0

    async def process_all_competitor_data(self, competitor_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process all competitor data.
        
        Args:
            competitor_data: Raw competitor data
            
        Returns:
            Processed competitor data
        """
        # Create tasks for parallel processing
        news_task = None
        financial_task = None
        social_task = None
        product_task = None
        
        if "news" in competitor_data and competitor_data["news"]:
            news_task = self.process_news_data(competitor_data["news"])
        
        if "financial" in competitor_data and competitor_data["financial"]:
            financial_task = self.process_financial_data(competitor_data["financial"])
        
        if "social" in competitor_data and competitor_data["social"]:
            social_task = self.process_social_media_data(competitor_data["social"])
        
        if "products" in competitor_data and competitor_data["products"]:
            product_task = self.process_product_data(competitor_data["products"])
        
        # Create processed data dictionary
        processed_data = competitor_data.copy()
        processed_data["processed"] = True
        processed_data["processed_at"] = datetime.now().isoformat()
        
        # Process data in parallel
        tasks = []
        if news_task:
            tasks.append(news_task)
        if financial_task:
            tasks.append(financial_task)
        if social_task:
            tasks.append(social_task)
        if product_task:
            tasks.append(product_task)
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            result_index = 0
            if news_task:
                if isinstance(results[result_index], list):
                    processed_data["news"] = results[result_index]
                result_index += 1
            
            if financial_task:
                if isinstance(results[result_index], dict):
                    processed_data["financial"] = results[result_index]
                result_index += 1
            
            if social_task:
                if isinstance(results[result_index], dict):
                    processed_data["social"] = results[result_index]
                result_index += 1
            
            if product_task:
                if isinstance(results[result_index], dict):
                    processed_data["products"] = results[result_index]
                result_index += 1
        
        return processed_data 