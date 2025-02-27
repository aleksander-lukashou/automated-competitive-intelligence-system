"""
Competitor Analyzer

This module implements analysis functionality for competitive intelligence data.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from acis.agents.search_agent import SearchAgent
from acis.agents.mindmap_agent import MindMapAgent
from acis.config.settings import settings

# Configure logging
logger = logging.getLogger(__name__)


class CompetitorAnalyzer:
    """Analyzer for competitive intelligence data."""

    def __init__(self):
        """Initialize the competitor analyzer."""
        self.llm = OpenAI(
            temperature=0.2,
            model_name=settings.llm_model,
            openai_api_key=settings.openai_api_key
        )
        
        # Initialize agents
        self.search_agent = SearchAgent()
        self.mindmap_agent = MindMapAgent()
        
        # Define report generation prompt
        self.report_prompt = PromptTemplate(
            input_variables=["competitor", "search_results", "time_period", "report_type"],
            template="""
            Generate a comprehensive competitive intelligence report for {competitor} covering the past {time_period}.
            
            Use the following search results as input:
            {search_results}
            
            Report type: {report_type}
            
            The report should have the following sections:
            1. Executive Summary
            2. Key Developments
            3. Product Analysis
            4. Market Position
            5. Financial Performance
            6. Strategic Recommendations
            
            Format your response as a JSON object with the following structure:
            {{
              "summary": "Executive summary text",
              "sections": [
                {{
                  "title": "Section title",
                  "content": "Section content",
                  "insights": ["Key insight 1", "Key insight 2"]
                }}
              ]
            }}
            
            Only provide the JSON object, no other text.
            """
        )
        
        self.report_chain = LLMChain(llm=self.llm, prompt=self.report_prompt)
        
        # Define trend detection prompt
        self.trend_prompt = PromptTemplate(
            input_variables=["competitor", "search_results"],
            template="""
            Analyze the following competitive intelligence data for {competitor} and identify key trends:
            
            {search_results}
            
            Identify:
            1. Market trends
            2. Product development trends
            3. Strategic shifts
            4. Financial patterns
            
            Format your response as a JSON object with the following structure:
            {{
              "trends": [
                {{
                  "category": "Trend category",
                  "description": "Trend description",
                  "evidence": ["Evidence 1", "Evidence 2"],
                  "impact": "Potential business impact"
                }}
              ]
            }}
            
            Only provide the JSON object, no other text.
            """
        )
        
        self.trend_chain = LLMChain(llm=self.llm, prompt=self.trend_prompt)

    async def generate_report(self, competitor: str, time_period: str, report_type: str) -> Dict[str, Any]:
        """
        Generate a competitor analysis report.
        
        Args:
            competitor: The competitor to analyze
            time_period: Time period to cover (e.g., "3 months")
            report_type: Type of report (full, summary, financial, product)
            
        Returns:
            Report data as a dictionary
        """
        try:
            # Step 1: Gather data using search agent
            search_queries = [
                f"{competitor} financial performance",
                f"{competitor} product launch",
                f"{competitor} market strategy",
                f"{competitor} competitive positioning"
            ]
            
            all_results = []
            for query in search_queries:
                results = await self.search_agent.search(query, max_results=5)
                all_results.extend(results)
            
            # Format search results for the prompt
            search_results_text = json.dumps(all_results, indent=2)
            
            # Step 2: Generate report using LLM
            response = await asyncio.to_thread(
                self.report_chain.run,
                competitor=competitor,
                search_results=search_results_text,
                time_period=time_period,
                report_type=report_type
            )
            
            # Parse JSON response
            try:
                report_data = json.loads(response)
                logger.info(f"Generated report for {competitor}")
                return report_data
            except json.JSONDecodeError:
                logger.error(f"Failed to parse report JSON: {response}")
                # Return a basic report structure
                return {
                    "summary": f"Error generating report for {competitor}.",
                    "sections": [
                        {
                            "title": "Error",
                            "content": "There was an error generating the report. Please try again.",
                            "insights": []
                        }
                    ]
                }
                
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return {
                "summary": f"Error generating report for {competitor}.",
                "sections": [
                    {
                        "title": "Error",
                        "content": f"There was an error generating the report: {str(e)}",
                        "insights": []
                    }
                ]
            }

    async def detect_trends(self, competitor: str) -> Dict[str, Any]:
        """
        Detect trends in competitive intelligence data.
        
        Args:
            competitor: The competitor to analyze
            
        Returns:
            Trend data as a dictionary
        """
        try:
            # Gather data using search agent
            search_queries = [
                f"{competitor} trends",
                f"{competitor} industry analysis",
                f"{competitor} market direction"
            ]
            
            all_results = []
            for query in search_queries:
                results = await self.search_agent.search(query, max_results=5)
                all_results.extend(results)
            
            # Format search results for the prompt
            search_results_text = json.dumps(all_results, indent=2)
            
            # Generate trend analysis using LLM
            response = await asyncio.to_thread(
                self.trend_chain.run,
                competitor=competitor,
                search_results=search_results_text
            )
            
            # Parse JSON response
            try:
                trend_data = json.loads(response)
                logger.info(f"Detected trends for {competitor}")
                return trend_data
            except json.JSONDecodeError:
                logger.error(f"Failed to parse trend JSON: {response}")
                return {
                    "trends": [
                        {
                            "category": "Error",
                            "description": "Error detecting trends",
                            "evidence": [],
                            "impact": "Unknown"
                        }
                    ]
                }
                
        except Exception as e:
            logger.error(f"Error detecting trends: {e}")
            return {
                "trends": [
                    {
                        "category": "Error",
                        "description": f"Error detecting trends: {str(e)}",
                        "evidence": [],
                        "impact": "Unknown"
                    }
                ]
            }

    async def compare_competitors(self, competitors: List[str]) -> Dict[str, Any]:
        """
        Compare multiple competitors.
        
        Args:
            competitors: List of competitors to compare
            
        Returns:
            Comparison data as a dictionary
        """
        # This is a placeholder for future implementation
        # Would integrate data from multiple competitor analyses
        comparison = {
            "competitors": competitors,
            "comparison_date": datetime.now().strftime("%Y-%m-%d"),
            "categories": [
                {
                    "name": "Market Share",
                    "data": {}
                },
                {
                    "name": "Product Innovation",
                    "data": {}
                },
                {
                    "name": "Financial Performance",
                    "data": {}
                }
            ]
        }
        
        # Add placeholder data for each competitor
        for category in comparison["categories"]:
            for competitor in competitors:
                category["data"][competitor] = {
                    "score": 0,
                    "notes": f"Data for {competitor} not yet available"
                }
        
        return comparison 