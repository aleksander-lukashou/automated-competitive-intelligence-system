"""
Competitor Models

This module defines data models for storing competitor information.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class Product(BaseModel):
    """Product data model."""
    
    name: str = Field(..., description="Product name")
    category: str = Field(..., description="Product category")
    launch_date: Optional[str] = Field(None, description="Product launch date")
    description: str = Field(..., description="Product description")
    features: List[str] = Field(default_factory=list, description="Product features")
    pricing: Optional[Dict[str, Any]] = Field(None, description="Pricing information")
    market_share: Optional[float] = Field(None, description="Market share percentage")
    reviews: Optional[Dict[str, Any]] = Field(None, description="Review information")


class FinancialMetric(BaseModel):
    """Financial metric data model."""
    
    name: str = Field(..., description="Metric name")
    value: float = Field(..., description="Metric value")
    unit: str = Field(..., description="Unit of measurement")
    date: str = Field(..., description="Date of measurement")
    change: Optional[float] = Field(None, description="Change from previous period")
    change_percentage: Optional[float] = Field(None, description="Percentage change")


class MarketPosition(BaseModel):
    """Market position data model."""
    
    segment: str = Field(..., description="Market segment")
    position: int = Field(..., description="Position in segment")
    market_share: float = Field(..., description="Market share percentage")
    competitors: List[str] = Field(default_factory=list, description="Direct competitors in segment")
    strengths: List[str] = Field(default_factory=list, description="Competitive strengths")
    weaknesses: List[str] = Field(default_factory=list, description="Competitive weaknesses")


class Strategy(BaseModel):
    """Strategy data model."""
    
    category: str = Field(..., description="Strategy category")
    description: str = Field(..., description="Strategy description")
    timeline: Optional[str] = Field(None, description="Implementation timeline")
    impact: Optional[str] = Field(None, description="Expected business impact")
    evidence: List[str] = Field(default_factory=list, description="Evidence supporting strategy assessment")


class NewsItem(BaseModel):
    """News item data model."""
    
    headline: str = Field(..., description="News headline")
    source: str = Field(..., description="News source")
    date: str = Field(..., description="Publication date")
    url: str = Field(..., description="URL to the news item")
    summary: str = Field(..., description="Summary of the news item")
    sentiment: float = Field(..., description="Sentiment score (-1 to 1)")
    categories: List[str] = Field(default_factory=list, description="Content categories")
    entities: List[str] = Field(default_factory=list, description="Named entities mentioned")


class Competitor(BaseModel):
    """Competitor data model."""
    
    name: str = Field(..., description="Competitor name")
    description: str = Field(..., description="Competitor description")
    industry: str = Field(..., description="Primary industry")
    founded: Optional[str] = Field(None, description="Founding date")
    headquarters: Optional[str] = Field(None, description="Headquarters location")
    employees: Optional[int] = Field(None, description="Number of employees")
    revenue: Optional[Dict[str, Any]] = Field(None, description="Revenue information")
    products: List[Product] = Field(default_factory=list, description="Products offered")
    financials: List[FinancialMetric] = Field(default_factory=list, description="Financial metrics")
    market_positions: List[MarketPosition] = Field(default_factory=list, description="Market positions")
    strategies: List[Strategy] = Field(default_factory=list, description="Business strategies")
    news: List[NewsItem] = Field(default_factory=list, description="Recent news")
    last_updated: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Last updated timestamp") 