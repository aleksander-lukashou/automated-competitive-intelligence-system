#!/bin/bash

# This script fixes the langchain dependency issue

echo "Fixing LangChain dependency issue..."

# Check if running in conda env
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "❌ Not running in a conda environment. Please activate your 'acis' environment first:"
    echo "    conda activate acis"
    exit 1
fi

# Install/update langchain and langchain-community
echo "Installing/updating required packages..."
pip install langchain>=0.1.0 langchain-community>=0.0.10

if [ $? -eq 0 ]; then
    echo "✅ Dependencies updated successfully!"
    echo ""
    echo "You can now run the application with: python app.py"
else
    echo "❌ Failed to update dependencies. Please try manually running:"
    echo "    pip install langchain>=0.1.0 langchain-community>=0.0.10"
fi 