"""
Helper Utilities

This module provides utility functions for the ACIS system.
"""

import re
import logging
from typing import Optional
from datetime import datetime
import dateutil.parser

# Configure logging
logger = logging.getLogger(__name__)


def sanitize_query(query: str) -> str:
    """
    Sanitize a search query by removing special characters and extra whitespace.
    
    Args:
        query: The search query to sanitize
        
    Returns:
        Sanitized query
    """
    # Remove special characters except spaces, alphanumeric, and basic punctuation
    sanitized = re.sub(r'[^\w\s.,?!-]', '', query)
    
    # Replace multiple spaces with single space
    sanitized = re.sub(r'\s+', ' ', sanitized)
    
    return sanitized.strip()


def extract_date(headline: str, content: str) -> Optional[str]:
    """
    Extract a publication date from headline or content text.
    
    Args:
        headline: Article headline
        content: Article content
        
    Returns:
        Date string in YYYY-MM-DD format, or None if no date found
    """
    # Look for dates in common formats
    date_patterns = [
        # YYYY-MM-DD
        r'(\d{4}-\d{1,2}-\d{1,2})',
        # Month DD, YYYY
        r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}',
        # DD Month YYYY
        r'\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}',
        # MM/DD/YYYY
        r'(\d{1,2}/\d{1,2}/\d{4})',
    ]
    
    # Combine headline and content
    text = f"{headline} {content}"
    
    for pattern in date_patterns:
        matches = re.search(pattern, text)
        if matches:
            date_str = matches.group(0)
            try:
                # Parse date and convert to standard format
                date_obj = dateutil.parser.parse(date_str)
                return date_obj.strftime('%Y-%m-%d')
            except Exception as e:
                logger.debug(f"Failed to parse date '{date_str}': {e}")
    
    # If no date found, return None
    return None


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: The text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    # Truncate at word boundary
    truncated = text[:max_length].rsplit(' ', 1)[0]
    return f"{truncated}..."


def format_currency(value: float) -> str:
    """
    Format a value as currency.
    
    Args:
        value: The value to format
        
    Returns:
        Formatted currency string
    """
    if value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    elif value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    elif value >= 1_000:
        return f"${value / 1_000:.2f}K"
    else:
        return f"${value:.2f}"


def calculate_growth_rate(current: float, previous: float) -> float:
    """
    Calculate growth rate between two values.
    
    Args:
        current: Current value
        previous: Previous value
        
    Returns:
        Growth rate as a percentage
    """
    if previous == 0:
        return 0.0
    
    return ((current - previous) / previous) * 100


def sentiment_to_label(sentiment: float) -> str:
    """
    Convert a sentiment score to a human-readable label.
    
    Args:
        sentiment: Sentiment score (-1 to 1)
        
    Returns:
        Sentiment label
    """
    if sentiment >= 0.5:
        return "Very Positive"
    elif sentiment >= 0.1:
        return "Positive"
    elif sentiment > -0.1:
        return "Neutral"
    elif sentiment > -0.5:
        return "Negative"
    else:
        return "Very Negative" 