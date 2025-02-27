"""
ACIS Configuration Settings

This module contains all configuration settings for the ACIS application.
Settings can be overridden by environment variables.
"""

import os
import logging
from pydantic import Field
from pydantic_settings import BaseSettings

# Configure logging
logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    """Application settings."""
    
    # API settings
    host: str = Field(default="0.0.0.0", env="ACIS_HOST")
    port: int = Field(default=8000, env="ACIS_PORT")
    debug_mode: bool = Field(default=True, env="ACIS_DEBUG_MODE")
    
    # Dashboard settings
    dashboard_port: int = Field(default=8501, env="ACIS_DASHBOARD_PORT")
    auto_start_dashboard: bool = Field(default=True, env="ACIS_AUTO_START_DASHBOARD")
    
    # Database settings
    db_url: str = Field(default="mongodb://localhost:27017", env="ACIS_DB_URL")
    db_name: str = Field(default="acis", env="ACIS_DB_NAME")
    
    # External API settings
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    google_api_key: str = Field(default="", env="GOOGLE_API_KEY")
    google_search_engine_id: str = Field(default="", env="GOOGLE_CX")
    
    # Search settings
    search_results_limit: int = Field(default=10, env="ACIS_SEARCH_RESULTS_LIMIT")
    
    # Monitoring and alerts
    enable_alerts: bool = Field(default=True, env="ACIS_ENABLE_ALERTS")
    alert_email: str = Field(default="", env="ACIS_ALERT_EMAIL")
    
    # Security
    api_key_required: bool = Field(default=True, env="ACIS_API_KEY_REQUIRED")
    api_key: str = Field(default="", env="ACIS_API_KEY")
    
    # LLM settings
    llm_model: str = Field(default="gpt-4", env="ACIS_LLM_MODEL")
    
    # Updated configuration for Pydantic v2
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore"
    }


# Create settings instance
settings = Settings()

# Debug logging of critical settings
logger.info("Settings loaded:")
logger.info(f"GOOGLE_API_KEY: {settings.google_api_key[:5]}... (length: {len(settings.google_api_key) if settings.google_api_key else 0})")
logger.info(f"GOOGLE_CX: {settings.google_search_engine_id} (length: {len(settings.google_search_engine_id) if settings.google_search_engine_id else 0})")
logger.info(f"Environment variable GOOGLE_CX value: {os.environ.get('GOOGLE_CX', 'Not found')}")
logger.info(f"API key required: {settings.api_key_required}") 