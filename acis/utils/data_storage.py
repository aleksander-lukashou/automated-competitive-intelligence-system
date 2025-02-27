"""
Data Storage Utilities

This module provides data storage functionalities for the ACIS system.
It handles persistent storage of activity logs, insights, and search results.
"""

import json
import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import uuid

from acis.config.settings import settings

# Configure logging
logger = logging.getLogger(__name__)

# Define data file paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
ACTIVITY_FILE = os.path.join(DATA_DIR, "activity.json")
INSIGHTS_FILE = os.path.join(DATA_DIR, "insights.json")
SEARCH_RESULTS_FILE = os.path.join(DATA_DIR, "search_results.json")


def ensure_data_dir():
    """Ensure the data directory exists."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        logger.info(f"Created data directory: {DATA_DIR}")

    # Initialize files if they don't exist
    for file_path in [ACTIVITY_FILE, INSIGHTS_FILE, SEARCH_RESULTS_FILE]:
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                json.dump([], f)
            logger.info(f"Initialized data file: {file_path}")


def log_activity(activity_type: str, description: str) -> Dict[str, Any]:
    """
    Log a new activity.
    
    Args:
        activity_type: Type of activity (e.g., "Search", "Report", "Alert")
        description: Description of the activity
    
    Returns:
        The created activity record
    """
    ensure_data_dir()
    
    # Create activity record
    activity = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "type": activity_type,
        "description": description
    }
    
    # Load existing activities
    try:
        with open(ACTIVITY_FILE, 'r') as f:
            activities = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        activities = []
    
    # Add new activity and save
    activities.insert(0, activity)  # Add to beginning (newest first)
    activities = activities[:100]  # Keep only the most recent 100 activities
    
    with open(ACTIVITY_FILE, 'w') as f:
        json.dump(activities, f, indent=2)
    
    logger.info(f"Logged activity: {activity_type} - {description}")
    return activity


def get_recent_activities(limit: int = 5) -> List[Dict[str, Any]]:
    """
    Get recent activities.
    
    Args:
        limit: Maximum number of activities to return
    
    Returns:
        List of recent activities
    """
    ensure_data_dir()
    
    try:
        with open(ACTIVITY_FILE, 'r') as f:
            activities = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        activities = []
    
    # Format timestamps for display
    for activity in activities:
        if 'timestamp' in activity:
            try:
                dt = datetime.fromisoformat(activity['timestamp'])
                activity['timestamp'] = dt.strftime('%Y-%m-%d %H:%M')
            except (ValueError, TypeError):
                pass
    
    return activities[:limit]


def store_insight(competitor: str, content: str, sentiment: float, source: str = "Analysis") -> Dict[str, Any]:
    """
    Store a new competitive insight.
    
    Args:
        competitor: The competitor this insight is about
        content: The insight content
        sentiment: Sentiment score (-1 to 1)
        source: Source of the insight
    
    Returns:
        The created insight record
    """
    ensure_data_dir()
    
    # Create insight record
    insight = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "competitor": competitor,
        "content": content,
        "sentiment": sentiment,
        "source": source
    }
    
    # Load existing insights
    try:
        with open(INSIGHTS_FILE, 'r') as f:
            insights = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        insights = []
    
    # Add new insight and save
    insights.insert(0, insight)  # Add to beginning (newest first)
    
    with open(INSIGHTS_FILE, 'w') as f:
        json.dump(insights, f, indent=2)
    
    logger.info(f"Stored insight for {competitor}: {content[:50]}...")
    return insight


def get_latest_insights(limit: int = 5, competitor: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get latest competitive insights.
    
    Args:
        limit: Maximum number of insights to return
        competitor: Optional filter for a specific competitor
    
    Returns:
        List of insights
    """
    ensure_data_dir()
    
    try:
        with open(INSIGHTS_FILE, 'r') as f:
            insights = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        insights = []
    
    # Filter by competitor if specified
    if competitor:
        insights = [i for i in insights if i.get('competitor', '').lower() == competitor.lower()]
    
    return insights[:limit]


def store_search_results(query: str, results: List[Dict[str, Any]]) -> str:
    """
    Store search results.
    
    Args:
        query: The search query
        results: The search results
    
    Returns:
        ID of the stored search results
    """
    ensure_data_dir()
    
    # Create search record
    search_id = str(uuid.uuid4())
    search_record = {
        "id": search_id,
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "results": results
    }
    
    # Load existing search results
    try:
        with open(SEARCH_RESULTS_FILE, 'r') as f:
            searches = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        searches = []
    
    # Add new search and save
    searches.insert(0, search_record)
    searches = searches[:50]  # Keep only the most recent 50 searches
    
    with open(SEARCH_RESULTS_FILE, 'w') as f:
        json.dump(searches, f, indent=2)
    
    logger.info(f"Stored search results for query: {query}")
    return search_id


def get_search_results(search_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get stored search results.
    
    Args:
        search_id: Optional ID of a specific search
    
    Returns:
        Search results
    """
    ensure_data_dir()
    
    try:
        with open(SEARCH_RESULTS_FILE, 'r') as f:
            searches = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        searches = []
    
    # Return specific search if ID provided
    if search_id:
        for search in searches:
            if search.get('id') == search_id:
                return search
        return None
    
    return searches 