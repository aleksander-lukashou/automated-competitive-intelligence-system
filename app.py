"""
ACIS - Automated Competitive Intelligence System

Main application entry point that initializes the FastAPI server
and integrates with the Streamlit dashboard.
"""

import os
import uvicorn
from fastapi import FastAPI
import subprocess
import threading
import time
from acis.api.routes import router as api_router
from acis.config.settings import settings

# Initialize FastAPI app
app = FastAPI(
    title="ACIS API",
    description="Automated Competitive Intelligence System API",
    version="1.0.0",
)

# Include API routes
app.include_router(api_router, prefix="/api/v1")

def start_dashboard():
    """Start the Streamlit dashboard in a separate process."""
    time.sleep(2)  # Give FastAPI server time to start
    subprocess.Popen(["streamlit", "run", "acis/dashboard/app.py"])

@app.on_event("startup")
async def startup_event():
    """Run startup tasks when the application starts."""
    # Start the dashboard in a separate thread
    if settings.auto_start_dashboard:
        threading.Thread(target=start_dashboard, daemon=True).start()

if __name__ == "__main__":
    # Run the FastAPI application
    uvicorn.run(
        "app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug_mode
    ) 