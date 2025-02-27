"""
API Schemas

This module defines the request and response schemas for the ACIS API.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


# Search API Schemas
class SearchRequest(BaseModel):
    """Search request schema."""
    
    query: str = Field(..., description="The search query")
    max_results: int = Field(10, description="Maximum number of results to return")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "Tesla new product launch",
                "max_results": 10
            }
        }


class SearchResult(BaseModel):
    """Individual search result schema."""
    
    source: str = Field(..., description="Source of the information")
    headline: str = Field(..., description="Headline or title of the result")
    url: str = Field(..., description="URL to the source article")
    date: Optional[str] = Field(None, description="Publication date")
    summary: Optional[str] = Field(None, description="Summary of the content")
    sentiment: Optional[float] = Field(None, description="Sentiment score (-1 to 1)")


class SearchResponse(BaseModel):
    """Search response schema."""
    
    status: str = Field(..., description="Status of the request")
    results: List[SearchResult] = Field(default_factory=list, description="List of search results")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "results": [
                    {
                        "source": "TechCrunch",
                        "headline": "Tesla announces new Model Y variant",
                        "url": "https://techcrunch.com/article123",
                        "date": "2023-05-15",
                        "summary": "Tesla has announced a new lower-cost variant of its Model Y vehicle...",
                        "sentiment": 0.75
                    }
                ]
            }
        }


# Mind Map API Schemas
class MindMapRequest(BaseModel):
    """Mind map request schema."""
    
    entity: str = Field(..., description="The main entity to create a mind map for")
    relations: List[str] = Field(default_factory=list, description="Relations to include in the mind map")
    
    class Config:
        schema_extra = {
            "example": {
                "entity": "Tesla",
                "relations": ["Product Launch", "Revenue Impact"]
            }
        }


class MindMapNode(BaseModel):
    """Mind map node schema."""
    
    id: str = Field(..., description="Unique identifier for the node")
    label: str = Field(..., description="Label for the node")
    type: str = Field(..., description="Type of node (entity, product, event, etc.)")
    data: Dict[str, Any] = Field(default_factory=dict, description="Additional data for the node")


class MindMapEdge(BaseModel):
    """Mind map edge schema."""
    
    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    label: str = Field(..., description="Label for the edge")
    type: str = Field(..., description="Type of relationship")


class MindMapResponse(BaseModel):
    """Mind map response schema."""
    
    status: str = Field(..., description="Status of the request")
    nodes: List[MindMapNode] = Field(default_factory=list, description="Nodes in the mind map")
    edges: List[MindMapEdge] = Field(default_factory=list, description="Edges in the mind map")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "nodes": [
                    {
                        "id": "1",
                        "label": "Tesla",
                        "type": "company",
                        "data": {}
                    },
                    {
                        "id": "2",
                        "label": "Model Y",
                        "type": "product",
                        "data": {"launch_date": "2023-05-15"}
                    }
                ],
                "edges": [
                    {
                        "source": "1",
                        "target": "2",
                        "label": "launched",
                        "type": "product_launch"
                    }
                ]
            }
        }


# Report API Schemas
class ReportRequest(BaseModel):
    """Report generation request schema."""
    
    competitor: str = Field(..., description="Competitor to generate report for")
    time_period: str = Field("1 month", description="Time period to cover in the report")
    report_type: str = Field("full", description="Type of report (full, summary, financial, product)")
    
    class Config:
        schema_extra = {
            "example": {
                "competitor": "Tesla",
                "time_period": "3 months",
                "report_type": "full"
            }
        }


class ReportSection(BaseModel):
    """Report section schema."""
    
    title: str = Field(..., description="Section title")
    content: str = Field(..., description="Section content")
    insights: List[str] = Field(default_factory=list, description="Key insights from this section")


class ReportResponse(BaseModel):
    """Report response schema."""
    
    status: str = Field(..., description="Status of the request")
    report_id: str = Field(..., description="Unique identifier for the report")
    title: str = Field(..., description="Report title")
    competitor: str = Field(..., description="Competitor covered in the report")
    date_generated: str = Field(..., description="Date the report was generated")
    sections: List[ReportSection] = Field(default_factory=list, description="Report sections")
    summary: str = Field(..., description="Executive summary")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "report_id": "r123456",
                "title": "Tesla Competitive Analysis - Q2 2023",
                "competitor": "Tesla",
                "date_generated": "2023-07-01",
                "sections": [
                    {
                        "title": "Product Launches",
                        "content": "Tesla has launched several new products in Q2 2023...",
                        "insights": ["New Model Y variant targets mid-market", "Price reduction strategy"]
                    }
                ],
                "summary": "Tesla has shown aggressive market expansion in Q2 2023..."
            }
        } 