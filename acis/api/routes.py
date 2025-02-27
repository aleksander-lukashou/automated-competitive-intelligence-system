"""
API Routes

This module defines the API routes for the ACIS system.
"""

from fastapi import APIRouter, Depends, HTTPException, Header
from typing import Optional
import datetime
import uuid

from acis.api.schemas import (
    SearchRequest, SearchResponse, SearchResult,
    MindMapRequest, MindMapResponse, MindMapNode, MindMapEdge,
    ReportRequest, ReportResponse, ReportSection
)
from acis.agents.search_agent import SearchAgent
from acis.agents.mindmap_agent import MindMapAgent
from acis.core.analyzer import CompetitorAnalyzer
from acis.config.settings import settings
from acis.utils.data_storage import log_activity

# Create router
router = APIRouter()

# API key dependency
async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Verify API key if required."""
    # Temporarily disable API key verification for development
    return True
    
    # Original code (commented out)
    # if settings.api_key_required:
    #     if not x_api_key or x_api_key != settings.api_key:
    #         raise HTTPException(status_code=401, detail="Invalid API key")
    # return True


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Perform a web search for competitive intelligence.
    
    - **query**: The search query
    - **max_results**: Maximum number of results to return
    """
    # Log API call
    log_activity("API", f"Search request: {request.query}")
    
    # Initialize search agent
    search_agent = SearchAgent()
    
    try:
        # Perform search
        results = await search_agent.search(request.query, request.max_results)
        
        # Format results
        search_results = []
        for result in results:
            search_results.append(
                SearchResult(
                    source=result.get("source", "Unknown"),
                    headline=result.get("headline", ""),
                    url=result.get("url", ""),
                    date=result.get("date"),
                    summary=result.get("summary"),
                    sentiment=result.get("sentiment")
                )
            )
        
        # Log success
        log_activity("API", f"Search successful: {len(search_results)} results")
        
        return SearchResponse(
            status="success",
            results=search_results
        )
    except Exception as e:
        # Log error
        log_activity("API", f"Search error: {str(e)}")
        
        # Raise HTTP exception
        raise HTTPException(
            status_code=500,
            detail=f"Error performing search: {str(e)}"
        )


@router.post("/mindmap", response_model=MindMapResponse, tags=["Mind Map"])
async def create_mindmap(request: MindMapRequest, api_key_valid: bool = Depends(verify_api_key)):
    """
    Generate a mind map for a competitor.
    
    - **entity**: The main entity to create a mind map for
    - **relations**: Relations to include in the mind map
    """
    # Initialize mind map agent
    mindmap_agent = MindMapAgent()
    
    try:
        # Generate mind map
        graph = await mindmap_agent.generate_mindmap(request.entity, request.relations)
        
        # Format response
        nodes = []
        edges = []
        
        for node in graph.get("nodes", []):
            nodes.append(
                MindMapNode(
                    id=node.get("id"),
                    label=node.get("label"),
                    type=node.get("type"),
                    data=node.get("data", {})
                )
            )
        
        for edge in graph.get("edges", []):
            edges.append(
                MindMapEdge(
                    source=edge.get("source"),
                    target=edge.get("target"),
                    label=edge.get("label"),
                    type=edge.get("type")
                )
            )
        
        return MindMapResponse(
            status="success",
            nodes=nodes,
            edges=edges
        )
    except Exception as e:
        # Handle errors
        return MindMapResponse(
            status="error",
            nodes=[],
            edges=[]
        )


@router.post("/report", response_model=ReportResponse, tags=["Reports"])
async def generate_report(request: ReportRequest, api_key_valid: bool = Depends(verify_api_key)):
    """
    Generate a competitor analysis report.
    
    - **competitor**: Competitor to generate report for
    - **time_period**: Time period to cover in the report
    - **report_type**: Type of report (full, summary, financial, product)
    """
    # Initialize competitor analyzer
    analyzer = CompetitorAnalyzer()
    
    try:
        # Generate report
        report_data = await analyzer.generate_report(
            request.competitor,
            request.time_period,
            request.report_type
        )
        
        # Format response
        sections = []
        for section in report_data.get("sections", []):
            sections.append(
                ReportSection(
                    title=section.get("title"),
                    content=section.get("content"),
                    insights=section.get("insights", [])
                )
            )
        
        return ReportResponse(
            status="success",
            report_id=str(uuid.uuid4()),
            title=f"{request.competitor} Competitive Analysis - {datetime.datetime.now().strftime('%B %Y')}",
            competitor=request.competitor,
            date_generated=datetime.datetime.now().strftime("%Y-%m-%d"),
            sections=sections,
            summary=report_data.get("summary", "No summary available")
        )
    except Exception as e:
        # Handle errors
        return ReportResponse(
            status="error",
            report_id=str(uuid.uuid4()),
            title=f"{request.competitor} Competitive Analysis - Error",
            competitor=request.competitor,
            date_generated=datetime.datetime.now().strftime("%Y-%m-%d"),
            sections=[],
            summary=f"Error generating report: {str(e)}"
        ) 