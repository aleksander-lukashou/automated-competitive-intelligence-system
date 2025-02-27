#!/bin/bash

# This script installs the ACIS package in development mode

echo "Installing ACIS package in development mode..."

# Check if running in conda env
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "❌ Not running in a conda environment. Please activate your 'acis' environment first:"
    echo "    conda activate acis"
    exit 1
fi

# Install package in development mode
pip install -e .

if [ $? -eq 0 ]; then
    echo "✅ ACIS package installed successfully in development mode!"
    echo ""
    echo "You can now run the application with: python app.py"
    echo "Or run the dashboard directly with: streamlit run acis/dashboard/app.py"
else
    echo "❌ Failed to install ACIS package. Please check the error messages above."
fi 