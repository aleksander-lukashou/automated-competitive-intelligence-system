# Automated Competitive Intelligence System (ACIS)

ACIS helps businesses track competitor activity, market trends, and industry changes using AI-powered reasoning and external tool integrations.

## Features

- **Real-time competitor tracking**
- **Web-based insights retrieval (web-search agent)**
- **Structured competitor strategy visualization (Mind Map agent)**
- **Automated competitor performance analysis (Coding agent)**
- **Scheduled reports & alerts**

## Project Structure

```
acis/
├── agents/                  # Agent implementations
│   ├── search_agent.py      # Web search agent
│   ├── mindmap_agent.py     # Mind map visualization agent
│   └── coding_agent.py      # Code analysis agent
├── api/                     # API endpoints
│   ├── __init__.py
│   ├── routes.py            # API route definitions
│   └── schemas.py           # API request/response schemas
├── core/                    # Core system functionality
│   ├── __init__.py
│   ├── data_collector.py    # Data collection logic
│   ├── processor.py         # Data processing pipeline
│   └── analyzer.py          # Analysis modules
├── models/                  # Data models
│   ├── __init__.py
│   ├── competitor.py        # Competitor data model
│   └── report.py            # Report data model
├── utils/                   # Utility functions
│   ├── __init__.py
│   └── helpers.py           # Helper functions
├── config/                  # Configuration files
│   ├── __init__.py
│   └── settings.py          # System settings
├── dashboard/               # User interface
│   ├── __init__.py
│   ├── app.py               # Streamlit dashboard
│   └── visualizations.py    # Data visualization components
├── tests/                   # Unit and integration tests
│   ├── __init__.py
│   ├── test_agents.py       # Agent tests
│   └── test_core.py         # Core functionality tests
├── .env.example             # Example environment variables
├── app.py                   # Main application entry point
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

## Setup and Installation

1. Clone the repository
2. Choose one of the following environment options:

### Google API Configuration

Before running the application, you need to set up the Google Custom Search API:

1. Create a Google Cloud project at https://console.cloud.google.com/
2. Enable the Custom Search API:
   - Navigate to "APIs & Services" > "Library"
   - Search for "Custom Search API" 
   - Click on it and select "Enable"
3. Create API credentials:
   - Navigate to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "API Key"
   - Copy the generated API key
4. Set up a Custom Search Engine:
   - Visit https://programmablesearchengine.google.com/
   - Click "Add" to create a new search engine
   - Configure your search engine settings
   - Note your Search Engine ID (cx value)
5. Add these values to your `.env` file:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   GOOGLE_CX=your_custom_search_engine_id_here
   ```

**Important:** Even if you have valid API credentials, you must enable the Custom Search API in your Google Cloud project or you'll receive a 403 error with message "API not enabled".

### Option 1: Using Conda (Recommended)

1. Create and activate a conda environment:
   ```bash
   # Using the setup script
   ./setup_conda.sh
   
   # Or manually
   conda env create -f environment.yml
   conda activate acis
   ```
2. Install the ACIS package:
   ```bash
   ./install_package.sh
   
   # Or manually
   pip install -e .
   ```
3. Copy `.env.example` to `.env` and configure environment variables
4. Run the application:
   ```bash
   python app.py
   ```

### Troubleshooting

We've identified a few common issues that you might encounter:

#### 1. Missing LangChain Community Module

If you encounter a `ModuleNotFoundError: No module named 'langchain_community'` error, you can fix it by running:

```bash
./fix_langchain.sh
```

This error occurs because newer versions of LangChain have separated some functionality into a separate package.

#### 2. Pydantic BaseSettings Import Error

If you encounter a `PydanticImportError: BaseSettings has been moved to the pydantic-settings package`, you can fix it by running:

```bash
./fix_dependencies.sh
```

This error occurs because in Pydantic V2, the BaseSettings class has been moved to a separate package.

#### 3. Pydantic Configuration Error

If you encounter validation errors with messages like `Extra inputs are not permitted [type=extra_forbidden]`, this is due to Pydantic V2 configuration changes. The fix for this issue is included in our comprehensive fix script.

This error occurs because Pydantic V2 replaced the inner `Config` class with a `model_config` dictionary and changed how environment variables are processed.

#### 4. Module Not Found Error ('acis')

If you encounter a `ModuleNotFoundError: No module named 'acis'` error, it means that the Python interpreter cannot find the ACIS package. This can be fixed by installing the package in development mode:

```bash
./install_package.sh

# Or manually
pip install -e .
```

This installs the package in "editable" mode, which means you can modify the code and have the changes immediately available without reinstalling.

#### 5. Comprehensive Fix

For the quickest solution to all dependency issues, use our comprehensive fix script:

```bash
./fix_dependencies.sh

# Or manually
pip install langchain>=0.1.0 langchain-community>=0.0.10 pydantic>=2.4.0 pydantic-settings>=2.0.0
pip install -e .
```

**Note:** You may still see some deprecation warnings after fixing these issues, but the application should function correctly.

#### 6. Google Custom Search API Errors

If you're receiving `"accessNotConfigured"` errors or the message "Custom Search API has not been used in project ... before or it is disabled", follow these steps:

1. Ensure you've enabled the Custom Search API for your project:
   - Go to https://console.cloud.google.com/apis/library/customsearch.googleapis.com
   - Make sure your project is selected in the dropdown at the top
   - Click "Enable" if it's not already enabled

2. Verify your API credentials:
   - Check that your API key in the `.env` file is correct
   - Ensure the API key has access to the Custom Search API

3. Confirm your Custom Search Engine setup:
   - Verify your Search Engine ID (cx value) is correct in the `.env` file
   - Check that your search engine is properly configured at https://programmablesearchengine.google.com/

Note that changes to API settings may take a few minutes to propagate through Google's systems.

### Option 2: Using Python Virtual Environment

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install the ACIS package:
   ```bash
   pip install -e .
   ```
4. Copy `.env.example` to `.env` and configure environment variables
5. Run the application:
   ```bash
   python app.py
   ```

## Dashboard

Access the dashboard at http://localhost:8501 after starting the application.

## Architecture

The system follows a modular architecture with the following components:

1. **Data Sources**: News websites, company websites, social media, industry reports
2. **Web-Search Agent**: LLM-driven search queries to fetch market updates
3. **Processing Pipeline**: Text extraction, summarization, sentiment analysis, trend detection
4. **Mind Map Agent**: Knowledge graph construction and visualization
5. **Competitive Insights & Reporting**: Custom reports, actionable insights, alerts

## API Documentation

### Web-Search Agent API

```
POST /api/v1/search
{
  "query": "Tesla new product launch",
  "max_results": 10
}
```

### Mind Map Agent API

```
POST /api/v1/mindmap
{
  "entity": "Tesla",
  "relations": ["Product Launch", "Revenue Impact"]
}
```

## Technology Stack

- **Backend**: Python (FastAPI)
- **Frontend**: Streamlit
- **Database**: MongoDB
- **NLP**: OpenAI API, LangChain
- **Visualization**: D3.js, Plotly
- **Deployment**: Docker, AWS/GCP 