"""
Report Models

This module defines data models for competitor intelligence reports.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class Insight(BaseModel):
    """Insight data model."""
    
    text: str = Field(..., description="Insight text")
    category: str = Field(..., description="Insight category")
    importance: int = Field(1, description="Importance level (1-5)")
    source: Optional[str] = Field(None, description="Source of the insight")
    confidence: float = Field(1.0, description="Confidence level (0-1)")


class ReportSection(BaseModel):
    """Report section data model."""
    
    title: str = Field(..., description="Section title")
    content: str = Field(..., description="Section content")
    insights: List[Insight] = Field(default_factory=list, description="Key insights from this section")
    charts: List[Dict[str, Any]] = Field(default_factory=list, description="Charts and visualizations")
    references: List[str] = Field(default_factory=list, description="References and sources")


class Trend(BaseModel):
    """Trend data model."""
    
    category: str = Field(..., description="Trend category")
    description: str = Field(..., description="Trend description")
    evidence: List[str] = Field(default_factory=list, description="Evidence supporting the trend")
    impact: str = Field(..., description="Potential business impact")
    timeframe: Optional[str] = Field(None, description="Expected timeframe")
    confidence: float = Field(1.0, description="Confidence level (0-1)")


class Recommendation(BaseModel):
    """Recommendation data model."""
    
    action: str = Field(..., description="Recommended action")
    rationale: str = Field(..., description="Rationale for recommendation")
    priority: int = Field(1, description="Priority level (1-5)")
    difficulty: int = Field(1, description="Implementation difficulty (1-5)")
    impact: int = Field(1, description="Expected impact (1-5)")
    timeframe: str = Field(..., description="Implementation timeframe")


class Report(BaseModel):
    """Report data model."""
    
    report_id: str = Field(..., description="Unique report identifier")
    title: str = Field(..., description="Report title")
    competitor: str = Field(..., description="Competitor covered in the report")
    date_generated: str = Field(..., description="Date the report was generated")
    time_period: str = Field(..., description="Time period covered by the report")
    report_type: str = Field(..., description="Type of report")
    summary: str = Field(..., description="Executive summary")
    sections: List[ReportSection] = Field(default_factory=list, description="Report sections")
    trends: List[Trend] = Field(default_factory=list, description="Identified trends")
    recommendations: List[Recommendation] = Field(default_factory=list, description="Strategic recommendations")
    charts: List[Dict[str, Any]] = Field(default_factory=list, description="Summary charts and visualizations")
    references: List[str] = Field(default_factory=list, description="References and sources")
    version: int = Field(1, description="Report version")
    status: str = Field("draft", description="Report status (draft, final, archived)")
    created_by: str = Field("ACIS System", description="Report creator")
    last_updated: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Last updated timestamp") 