"""
Tests for ACIS core components.

This module contains unit and integration tests for the core components,
including data collector and processor.
"""

import pytest
import asyncio
import json
from unittest.mock import patch, MagicMock, AsyncMock

from acis.core.data_collector import DataCollector
from acis.core.processor import DataProcessor
from acis.core.analyzer import CompetitorAnalyzer


class TestDataCollector:
    """Tests for the DataCollector class."""

    @pytest.fixture
    def collector(self):
        """Create a DataCollector instance for testing."""
        with patch('aiohttp.ClientSession') as mock_session:
            # Configure the mock session
            mock_session.return_value.__aenter__.return_value = mock_session.return_value
            mock_session.return_value.get.return_value.__aenter__.return_value.status = 200
            mock_session.return_value.get.return_value.__aenter__.return_value.text = AsyncMock(
                return_value="<html><body>Test content</body></html>"
            )
            
            collector = DataCollector()
            collector.session = mock_session.return_value
            
            yield collector
    
    @pytest.mark.asyncio
    async def test_collect_news(self, collector):
        """Test news collection."""
        news = await collector.collect_news("test company", 5)
        
        assert isinstance(news, list)
        assert len(news) <= 5
        
        if news:
            assert "source" in news[0]
            assert "headline" in news[0]
            assert "url" in news[0]
    
    @pytest.mark.asyncio
    async def test_collect_financial_data(self, collector):
        """Test financial data collection."""
        data = await collector.collect_financial_data("test company", 4)
        
        assert isinstance(data, dict)
        assert "company" in data
        assert "metrics" in data
        
        # Check if we have metrics data
        if data["metrics"]:
            # Get the first metric
            first_metric = next(iter(data["metrics"].values()))
            assert isinstance(first_metric, list)
            assert len(first_metric) <= 4  # Should have at most 4 quarters
            
            if first_metric:
                assert "year" in first_metric[0]
                assert "quarter" in first_metric[0]
                assert "value" in first_metric[0]
    
    @pytest.mark.asyncio
    async def test_collect_social_media(self, collector):
        """Test social media data collection."""
        data = await collector.collect_social_media("test company", 7)
        
        assert isinstance(data, dict)
        assert "company" in data
        assert "platforms" in data
        
        # Check if we have platform data
        if data["platforms"]:
            # Get the first platform
            first_platform = next(iter(data["platforms"].values()))
            assert "total_mentions" in first_platform
            assert "average_sentiment" in first_platform
            assert "daily_data" in first_platform
            
            if first_platform["daily_data"]:
                assert "date" in first_platform["daily_data"][0]
                assert "mentions" in first_platform["daily_data"][0]
                assert "sentiment" in first_platform["daily_data"][0]
    
    @pytest.mark.asyncio
    async def test_collect_product_data(self, collector):
        """Test product data collection."""
        products = await collector.collect_product_data("test company")
        
        assert isinstance(products, list)
        
        if products:
            assert "name" in products[0]
            assert "type" in products[0]
            assert "description" in products[0]
            assert "features" in products[0]
    
    @pytest.mark.asyncio
    async def test_collect_web_content(self, collector):
        """Test web content collection."""
        content = await collector.collect_web_content("https://example.com")
        
        assert isinstance(content, dict)
        assert "url" in content
        assert "title" in content or "content" in content
    
    @pytest.mark.asyncio
    async def test_collect_all_competitor_data(self, collector):
        """Test collecting all competitor data."""
        # Mock the individual collection methods
        collector.collect_news = AsyncMock(return_value=[{"headline": "Test News"}])
        collector.collect_financial_data = AsyncMock(return_value={"metrics": {"Revenue": []}})
        collector.collect_social_media = AsyncMock(return_value={"platforms": {}})
        collector.collect_product_data = AsyncMock(return_value=[{"name": "Test Product"}])
        
        data = await collector.collect_all_competitor_data("test company")
        
        assert isinstance(data, dict)
        assert "name" in data
        assert "last_updated" in data
        assert "news" in data
        assert "financial" in data
        assert "social" in data
        assert "products" in data


class TestDataProcessor:
    """Tests for the DataProcessor class."""

    @pytest.fixture
    def processor(self):
        """Create a DataProcessor instance for testing."""
        with patch('langchain.llms.OpenAI') as mock_openai:
            # Configure the mock
            mock_chain = MagicMock()
            mock_chain.run.return_value = "Test summary"
            mock_openai.return_value.run = mock_chain.run
            
            processor = DataProcessor()
            
            # Replace the chains with mocks
            processor.summary_chain = mock_chain
            processor.entity_chain = MagicMock()
            processor.entity_chain.run.return_value = '[{"type": "company", "text": "Test Company"}]'
            processor.sentiment_chain = MagicMock()
            processor.sentiment_chain.run.return_value = "0.5"
            
            yield processor
    
    @pytest.mark.asyncio
    async def test_summarize_text(self, processor):
        """Test text summarization."""
        summary = await processor.summarize_text("This is a test text to summarize. It contains information about a company.")
        
        assert isinstance(summary, str)
        assert len(summary) > 0
    
    @pytest.mark.asyncio
    async def test_extract_entities(self, processor):
        """Test entity extraction."""
        entities = await processor.extract_entities("Test Company is launching a new product called Test Product.")
        
        assert isinstance(entities, list)
        assert len(entities) > 0
        assert "type" in entities[0]
        assert "text" in entities[0]
    
    @pytest.mark.asyncio
    async def test_analyze_sentiment(self, processor):
        """Test sentiment analysis."""
        sentiment = await processor.analyze_sentiment("This is a very positive development for the company.")
        
        assert isinstance(sentiment, float)
        assert -1.0 <= sentiment <= 1.0
    
    @pytest.mark.asyncio
    async def test_process_news_data(self, processor):
        """Test news data processing."""
        test_news = [
            {
                "source": "Test Source",
                "headline": "Test Headline",
                "url": "https://example.com",
                "content": "Test content for processing."
            }
        ]
        
        processed = await processor.process_news_data(test_news)
        
        assert isinstance(processed, list)
        assert len(processed) == 1
        assert "summary" in processed[0]
        assert "entities" in processed[0]
        assert "sentiment" in processed[0]
    
    @pytest.mark.asyncio
    async def test_process_financial_data(self, processor):
        """Test financial data processing."""
        test_financial = {
            "company": "Test Company",
            "metrics": {
                "Revenue": [
                    {"year": 2023, "quarter": 1, "value": 100},
                    {"year": 2022, "quarter": 4, "value": 90},
                    {"year": 2022, "quarter": 3, "value": 80},
                    {"year": 2022, "quarter": 2, "value": 70}
                ]
            }
        }
        
        processed = await processor.process_financial_data(test_financial)
        
        assert isinstance(processed, dict)
        assert "calculated_metrics" in processed
        assert "Revenue_growth" in processed["calculated_metrics"]
    
    @pytest.mark.asyncio
    async def test_process_social_media_data(self, processor):
        """Test social media data processing."""
        test_social = {
            "company": "Test Company",
            "platforms": {
                "Twitter": {
                    "total_mentions": 1000,
                    "average_sentiment": 0.5,
                    "daily_data": [
                        {"date": "2023-01-01", "mentions": 100, "sentiment": 0.5, "popular_topics": ["product"]},
                        {"date": "2023-01-02", "mentions": 110, "sentiment": 0.6, "popular_topics": ["news"]},
                        {"date": "2023-01-03", "mentions": 90, "sentiment": 0.4, "popular_topics": ["service"]},
                        {"date": "2023-01-04", "mentions": 95, "sentiment": 0.5, "popular_topics": ["product"]},
                        {"date": "2023-01-05", "mentions": 105, "sentiment": 0.6, "popular_topics": ["product"]},
                        {"date": "2023-01-06", "mentions": 115, "sentiment": 0.7, "popular_topics": ["news"]},
                        {"date": "2023-01-07", "mentions": 120, "sentiment": 0.6, "popular_topics": ["product"]}
                    ]
                }
            }
        }
        
        processed = await processor.process_social_media_data(test_social)
        
        assert isinstance(processed, dict)
        assert "trends" in processed
        assert "Twitter" in processed["trends"]
        assert "popular_topics" in processed["trends"]["Twitter"]
    
    @pytest.mark.asyncio
    async def test_process_product_data(self, processor):
        """Test product data processing."""
        test_products = [
            {
                "name": "Product 1",
                "type": "Software",
                "description": "A software product",
                "launch_date": "2023-01-01",
                "pricing": {"amount": 100, "currency": "USD", "model": "subscription"},
                "features": ["Feature 1", "Feature 2"]
            },
            {
                "name": "Product 2",
                "type": "Hardware",
                "description": "A hardware product",
                "launch_date": "2022-06-01",
                "pricing": {"amount": 200, "currency": "USD", "model": "one-time"},
                "features": ["Feature 3", "Feature 4"]
            }
        ]
        
        processed = await processor.process_product_data(test_products)
        
        assert isinstance(processed, dict)
        assert "products" in processed
        assert "categories" in processed
        assert "timeline" in processed
        assert "features" in processed
    
    @pytest.mark.asyncio
    async def test_process_all_competitor_data(self, processor):
        """Test processing all competitor data."""
        # Create simplified test data
        test_data = {
            "name": "Test Company",
            "news": [{"headline": "Test", "content": "Test content"}],
            "financial": {"metrics": {"Revenue": [{"year": 2023, "quarter": 1, "value": 100}]}},
            "social": {"platforms": {"Twitter": {"daily_data": []}}},
            "products": [{"name": "Test Product", "type": "Software"}]
        }
        
        # Mock the individual processing methods
        processor.process_news_data = AsyncMock(return_value=[{"headline": "Test", "summary": "Test summary"}])
        processor.process_financial_data = AsyncMock(return_value={"calculated_metrics": {}})
        processor.process_social_media_data = AsyncMock(return_value={"trends": {}})
        processor.process_product_data = AsyncMock(return_value={"categories": {}})
        
        processed = await processor.process_all_competitor_data(test_data)
        
        assert isinstance(processed, dict)
        assert "processed" in processed
        assert processed["processed"] is True
        assert "processed_at" in processed


class TestCompetitorAnalyzer:
    """Tests for the CompetitorAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create a CompetitorAnalyzer instance for testing."""
        with patch('langchain.llms.OpenAI') as mock_openai:
            # Configure the mock
            mock_chain = MagicMock()
            mock_chain.run.return_value = '{"summary": "Test summary", "sections": []}'
            mock_openai.return_value.run = mock_chain.run
            
            analyzer = CompetitorAnalyzer()
            
            # Replace the chains with mocks
            analyzer.report_chain = mock_chain
            analyzer.trend_chain = mock_chain
            
            # Mock the agents
            analyzer.search_agent = MagicMock()
            analyzer.search_agent.search = AsyncMock(return_value=[])
            analyzer.mindmap_agent = MagicMock()
            
            yield analyzer
    
    @pytest.mark.asyncio
    async def test_generate_report(self, analyzer):
        """Test report generation."""
        report = await analyzer.generate_report("Test Company", "3 months", "full")
        
        assert isinstance(report, dict)
        assert "summary" in report
        assert "sections" in report
    
    @pytest.mark.asyncio
    async def test_detect_trends(self, analyzer):
        """Test trend detection."""
        # Set up the mock to return JSON
        analyzer.trend_chain.run.return_value = '{"trends": [{"category": "Market", "description": "Test trend"}]}'
        
        trends = await analyzer.detect_trends("Test Company")
        
        assert isinstance(trends, dict)
        assert "trends" in trends
        assert len(trends["trends"]) > 0
    
    @pytest.mark.asyncio
    async def test_compare_competitors(self, analyzer):
        """Test competitor comparison."""
        comparison = await analyzer.compare_competitors(["Company 1", "Company 2"])
        
        assert isinstance(comparison, dict)
        assert "competitors" in comparison
        assert "categories" in comparison 