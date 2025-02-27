"""
ACIS Configuration Settings

This module contains all configuration settings for the ACIS application.
Settings can be overridden by environment variables.
"""

import os
from pydantic import Field
from pydantic_settings import BaseSettings


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
    bing_api_key: str = Field(default="", env="BING_API_KEY")
    
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