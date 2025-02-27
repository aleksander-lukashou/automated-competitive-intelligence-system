"""
ACIS Dashboard

This module implements the Streamlit dashboard for the ACIS system.
"""

import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

from acis.config.settings import settings
from acis.dashboard.visualizations import create_mindmap, create_trend_chart
from acis.utils.data_storage import get_recent_activities, get_latest_insights, log_activity


# API URL
API_BASE_URL = f"http://{settings.host}:{settings.port}/api/v1"

# API Headers with API key
API_HEADERS = {"X-API-Key": settings.api_key}


def main():
    """Main dashboard function."""
    # Configure the page
    st.set_page_config(
        page_title="ACIS - Automated Competitive Intelligence System",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar
    with st.sidebar:
        st.title("📊 ACIS")
        st.subheader("Automated Competitive Intelligence System")
        
        # Navigation
        page = st.radio(
            "Navigate",
            ["Home", "Search", "Mind Maps", "Reports", "Trends", "Settings"]
        )
        
        st.divider()
        
        # Quick search
        st.subheader("Quick Search")
        quick_search = st.text_input("Search competitors")
        if st.button("Go"):
            st.session_state.quick_search = quick_search
            st.session_state.page = "Search"
    
    # Main content
    if page == "Home":
        render_home_page()
    elif page == "Search":
        render_search_page()
    elif page == "Mind Maps":
        render_mindmaps_page()
    elif page == "Reports":
        render_reports_page()
    elif page == "Trends":
        render_trends_page()
    elif page == "Settings":
        render_settings_page()
    
    # Footer
    st.divider()
    st.caption("Automated Competitive Intelligence System (ACIS) | v1.0.0")


def render_home_page():
    """Render the home page."""
    st.title("Automated Competitive Intelligence System")
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Companies Tracked", "15")
    with col2:
        st.metric("Reports Generated", "42")
    with col3:
        st.metric("Insights Detected", "156")
    with col4:
        st.metric("Data Sources", "24")
    
    # Recent activity
    st.subheader("Recent Activity")
    
    # Get real activity data
    activity_data = get_recent_activities(limit=5)
    
    if activity_data:
        activity_df = pd.DataFrame(activity_data)
        st.dataframe(activity_df, hide_index=True)
    else:
        st.info("No recent activity recorded yet. Start by using the search functionality.")
    
    # Quick actions
    st.subheader("Quick Actions")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Generate New Report"):
            log_activity("Navigation", "Navigated to Reports page")
            st.session_state.page = "Reports"
    with col2:
        if st.button("Create Mind Map"):
            log_activity("Navigation", "Navigated to Mind Maps page")
            st.session_state.page = "Mind Maps"
    with col3:
        if st.button("View Trends"):
            log_activity("Navigation", "Navigated to Trends page")
            st.session_state.page = "Trends"
    
    # Latest insights
    st.subheader("Latest Insights")
    
    # Get real insights
    insights = get_latest_insights(limit=5)
    
    if insights:
        for insight in insights:
            sentiment = insight.get("sentiment", 0)
            sentiment_color = "green" if sentiment > 0.1 else "red" if sentiment < -0.1 else "gray"
            
            with st.container():
                st.markdown(f"**{insight.get('competitor', 'Unknown')}**: "
                          f"{insight.get('content', '')}")
                st.caption(f"Source: {insight.get('source', 'Unknown')} | "
                         f"Sentiment: <span style='color:{sentiment_color}'>{sentiment:.2f}</span>",
                         unsafe_allow_html=True)
                st.divider()
    else:
        st.info("No insights detected yet. Try searching for competitive intelligence using the Search page.")


def render_search_page():
    """Render the search page."""
    st.title("Competitive Intelligence Search")
    
    # Search form
    with st.form("search_form"):
        query = st.text_input(
            "Search query",
            value=st.session_state.get("quick_search", "")
        )
        max_results = st.slider("Maximum results", 5, 50, 10)
        
        submitted = st.form_submit_button("Search")
        
        if submitted and query:
            st.session_state.search_query = query
            st.session_state.max_results = max_results
            st.session_state.search_submitted = True
            
            # Log search activity
            log_activity("UI Search", f"Query: {query}")
    
    # Display search results
    if st.session_state.get("search_submitted", False):
        with st.spinner("Searching..."):
            try:
                # Call the backend API with API key in headers
                response = requests.post(
                    f"{API_BASE_URL}/search",
                    json={"query": st.session_state.search_query, "max_results": st.session_state.max_results},
                    headers=API_HEADERS
                )
                
                if response.status_code == 200:
                    results = response.json().get("results", [])
                    
                    # Log successful search results
                    log_activity("Search Results", f"Found {len(results)} results for query: {st.session_state.search_query}")
                else:
                    st.error(f"Error: API returned status code {response.status_code}")
                    st.json(response.json())
                    results = []
                    
                    # Log error
                    log_activity("Search Error", f"API error {response.status_code} for query: {st.session_state.search_query}")
            except Exception as e:
                st.error(f"Error connecting to API: {str(e)}")
                results = []
                
                # Log connection error
                log_activity("Connection Error", f"Error connecting to search API: {str(e)}")
            
            # Display results
            st.subheader(f"Search Results for '{st.session_state.search_query}'")
            
            if not results:
                st.info("No results found. Try a different search query.")
            
            for result in results:
                with st.expander(result["headline"]):
                    st.markdown(f"**Source:** {result['source']}")
                    st.markdown(f"**Date:** {result.get('date', 'Unknown')}")
                    st.markdown(f"**URL:** [{result['url']}]({result['url']})")
                    st.markdown("**Summary:**")
                    st.write(result.get("summary", "No summary available."))
                    
                    # Display sentiment
                    sentiment = result.get("sentiment", 0)
                    sentiment_color = "green" if sentiment > 0 else "red" if sentiment < 0 else "gray"
                    st.markdown(f"**Sentiment:** <span style='color:{sentiment_color}'>{sentiment:.2f}</span>", unsafe_allow_html=True)
                    
                    # Add a button to open the URL
                    if st.button(f"Visit Source: {result['source']}", key=f"btn_visit_{result['source']}"):
                        # Log the click
                        log_activity("Source Visit", f"Clicked on source: {result['source']} ({result['url']})")


def render_mindmaps_page():
    """Render the mind maps page."""
    st.title("Competitor Mind Maps")
    
    # Mind map form
    with st.form("mindmap_form"):
        entity = st.text_input("Company/Entity", value="Tesla")
        relations = st.multiselect(
            "Relations to include",
            ["Product Launch", "Revenue Impact", "Market Strategy", "Competitors", "Technology"],
            ["Product Launch", "Revenue Impact"]
        )
        
        submitted = st.form_submit_button("Generate Mind Map")
        
        if submitted and entity and relations:
            st.session_state.mindmap_entity = entity
            st.session_state.mindmap_relations = relations
            st.session_state.mindmap_submitted = True
    
    # Display mind map
    if st.session_state.get("mindmap_submitted", False):
        with st.spinner("Generating mind map..."):
            # This would normally call the API with API key
            try:
                response = requests.post(
                    f"{API_BASE_URL}/mindmap",
                    json={"entity": st.session_state.mindmap_entity, "relations": st.session_state.mindmap_relations},
                    headers=API_HEADERS
                )
                if response.status_code == 200:
                    mind_map_data = response.json()
                    # Process the real mind map data
                    # For now, we'll still use the mock visualization
                else:
                    st.error(f"Error: API returned status code {response.status_code}")
                    st.json(response.json())
            except Exception as e:
                st.error(f"Error connecting to API: {str(e)}")
            
            # Create a simple mind map using Plotly
            mind_map_html = create_mindmap(
                st.session_state.mindmap_entity,
                st.session_state.mindmap_relations
            )
            
            st.subheader(f"Mind Map for {st.session_state.mindmap_entity}")
            st.components.v1.html(mind_map_html, height=600)
            
            # Download option
            st.download_button(
                "Download Mind Map",
                data=mind_map_html,
                file_name=f"{st.session_state.mindmap_entity.lower().replace(' ', '_')}_mind_map.html",
                mime="text/html"
            )


def render_reports_page():
    """Render the reports page."""
    st.title("Competitor Intelligence Reports")
    
    # Report generation form
    with st.form("report_form"):
        competitor = st.text_input("Competitor", value="Tesla")
        time_period = st.select_slider(
            "Time Period",
            options=["1 week", "1 month", "3 months", "6 months", "1 year"],
            value="3 months"
        )
        report_type = st.selectbox(
            "Report Type",
            ["Full", "Summary", "Financial", "Product"],
            index=0
        )
        
        submitted = st.form_submit_button("Generate Report")
        
        if submitted and competitor:
            st.session_state.report_competitor = competitor
            st.session_state.report_time_period = time_period
            st.session_state.report_type = report_type
            st.session_state.report_submitted = True
    
    # Display report
    if st.session_state.get("report_submitted", False):
        with st.spinner("Generating report..."):
            # Try to call the API with API key
            try:
                response = requests.post(
                    f"{API_BASE_URL}/report",
                    json={
                        "competitor": st.session_state.report_competitor,
                        "time_period": st.session_state.report_time_period,
                        "report_type": st.session_state.report_type
                    },
                    headers=API_HEADERS
                )
                if response.status_code == 200:
                    report_data = response.json()
                    # Process the real report data
                    # For now, we'll still use the mock data
                else:
                    st.error(f"Error: API returned status code {response.status_code}")
                    st.json(response.json())
            except Exception as e:
                st.error(f"Error connecting to API: {str(e)}")
            
            # Mock data for demonstration
            time.sleep(2)  # Simulate API call
            
            # Create a mock report
            report_data = {
                "status": "success",
                "report_id": "r123456",
                "title": f"{st.session_state.report_competitor} Competitive Analysis - Q2 2023",
                "competitor": st.session_state.report_competitor,
                "date_generated": datetime.now().strftime("%Y-%m-%d"),
                "sections": [
                    {
                        "title": "Executive Summary",
                        "content": f"{st.session_state.report_competitor} has shown strong market performance in the last quarter with revenue growth exceeding market expectations. The company continues to innovate in its core product categories while expanding into adjacent markets.",
                        "insights": ["Revenue growth exceeded expectations", "Expanding into adjacent markets"]
                    },
                    {
                        "title": "Key Developments",
                        "content": f"{st.session_state.report_competitor} announced several key initiatives including new product launches and strategic partnerships. The company is investing heavily in R&D to maintain its competitive edge.",
                        "insights": ["New product launches planned", "Strategic partnerships formed", "Increasing R&D investment"]
                    },
                    {
                        "title": "Financial Performance",
                        "content": f"Financial results for {st.session_state.report_competitor} show a 15% year-over-year revenue increase and improved profit margins. The company's stock has outperformed the market index by 8% over the last quarter.",
                        "insights": ["15% YoY revenue growth", "Improved profit margins", "Stock outperforming market"]
                    }
                ],
                "summary": f"{st.session_state.report_competitor} maintains a strong competitive position with continued innovation and market expansion. Key areas to monitor include new product launches and strategic partnerships."
            }
            
            # Display report
            st.header(report_data["title"])
            st.caption(f"Generated on {report_data['date_generated']} • Report ID: {report_data['report_id']}")
            
            st.subheader("Executive Summary")
            st.info(report_data["summary"])
            
            # Display sections
            for section in report_data["sections"]:
                with st.expander(section["title"]):
                    st.write(section["content"])
                    
                    if section.get("insights"):
                        st.subheader("Key Insights")
                        for insight in section["insights"]:
                            st.success(insight)
            
            # Download option
            report_json = json.dumps(report_data, indent=2)
            st.download_button(
                "Download Report",
                data=report_json,
                file_name=f"{st.session_state.report_competitor.lower().replace(' ', '_')}_report.json",
                mime="application/json"
            )


def render_trends_page():
    """Render the trends page."""
    st.title("Market & Competitor Trends")
    
    # Trend selection
    competitors = st.multiselect(
        "Select Competitors",
        ["Tesla", "Apple", "Microsoft", "Google", "Amazon", "Meta"],
        ["Tesla", "Apple"]
    )
    
    time_range = st.select_slider(
        "Time Range",
        options=["1 month", "3 months", "6 months", "1 year", "2 years"],
        value="6 months"
    )
    
    metrics = st.multiselect(
        "Metrics to Track",
        ["Stock Price", "Market Share", "Revenue Growth", "Product Launches", "R&D Spending", "Media Sentiment"],
        ["Stock Price", "Media Sentiment"]
    )
    
    if st.button("Analyze Trends"):
        with st.spinner("Analyzing trends..."):
            # Mock data for demonstration
            time.sleep(2)  # Simulate API call
            
            # Display trend visualization
            st.subheader("Trend Analysis")
            
            # Sample trend chart
            trend_chart = create_trend_chart(competitors, time_range, metrics)
            st.plotly_chart(trend_chart, use_container_width=True)
            
            # Sample trend insights
            st.subheader("Trend Insights")
            
            trends = [
                {
                    "category": "Market Share",
                    "description": "Tesla gaining market share in electric vehicles despite increased competition",
                    "evidence": ["5% quarterly increase", "New Model Y sales exceeding forecasts"],
                    "impact": "Positive for long-term market position"
                },
                {
                    "category": "Media Sentiment",
                    "description": "Apple seeing improved media sentiment following product announcements",
                    "evidence": ["Sentiment score improved from 0.2 to 0.7", "Positive reviews for new devices"],
                    "impact": "May drive higher consumer interest and sales"
                }
            ]
            
            for trend in trends:
                with st.expander(f"{trend['category']}: {trend['description']}"):
                    st.write(f"**Impact:** {trend['impact']}")
                    
                    st.write("**Evidence:**")
                    for evidence in trend["evidence"]:
                        st.write(f"- {evidence}")


def render_settings_page():
    """Render the settings page."""
    st.title("System Settings")
    
    # API settings
    st.header("API Configuration")
    with st.form("api_settings"):
        api_key = st.text_input("API Key", value=settings.api_key if settings.api_key else "••••••••••••••••", type="password")
        host = st.text_input("API Host", value=settings.host)
        port = st.number_input("API Port", value=settings.port)
        
        st.form_submit_button("Save API Settings")
    
    # Data source settings
    st.header("Data Sources")
    with st.form("data_sources"):
        google_api_key = st.text_input("Google API Key", value=settings.google_api_key[:5] + "••••••••••••" if settings.google_api_key else "••••••••••••••••", type="password")
        google_cx = st.text_input("Google Search Engine ID", value=settings.google_search_engine_id[:5] + "••••••••••••" if settings.google_search_engine_id else "••••••••••••••••", type="password")
        
        news_sources = st.multiselect(
            "News Sources",
            ["TechCrunch", "Forbes", "Business Insider", "Wall Street Journal", "Financial Times", "Bloomberg"],
            ["TechCrunch", "Forbes", "Business Insider"]
        )
        
        st.form_submit_button("Save Data Source Settings")
    
    # Alert settings
    st.header("Alert Configuration")
    with st.form("alert_settings"):
        enable_alerts = st.toggle("Enable Alerts", value=settings.enable_alerts)
        alert_email = st.text_input("Alert Email", value=settings.alert_email if settings.alert_email else "user@example.com")
        
        alert_frequency = st.select_slider(
            "Alert Frequency",
            options=["Real-time", "Daily", "Weekly"],
            value="Daily"
        )
        
        st.form_submit_button("Save Alert Settings")


if __name__ == "__main__":
    main() 