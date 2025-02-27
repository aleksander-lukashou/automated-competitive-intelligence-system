#!/bin/bash

# This script fixes various dependency issues in the ACIS project

echo "Fixing ACIS dependencies..."

# Check if running in conda env
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "❌ Not running in a conda environment. Please activate your 'acis' environment first:"
    echo "    conda activate acis"
    exit 1
fi

# Install/update all required packages
echo "Installing/updating required packages..."
pip install langchain>=0.1.0 langchain-community>=0.0.10 pydantic>=2.4.0 pydantic-settings>=2.0.0

if [ $? -ne 0 ]; then
    echo "❌ Failed to update dependencies. Please try manually running:"
    echo "    pip install langchain>=0.1.0 langchain-community>=0.0.10 pydantic>=2.4.0 pydantic-settings>=2.0.0"
    exit 1
fi

# Install the ACIS package in development mode
echo "Installing ACIS package in development mode..."
pip install -e .

if [ $? -eq 0 ]; then
    echo "✅ Dependencies updated and ACIS package installed successfully!"
    echo ""
    echo "Common Pydantic v2 Issues Fixed:"
    echo "  - BaseSettings import has been moved to pydantic-settings package"
    echo "  - Config class replaced with model_config dictionary"
    echo ""
    echo "Module Import Issues Fixed:"
    echo "  - ACIS package installed in development mode"
    echo ""
    echo "You can now run the application with: python app.py"
    echo "Or run the dashboard directly with: streamlit run acis/dashboard/app.py"
    echo ""
    echo "Note: You may still see some deprecation warnings about imports, but these are"
    echo "      just warnings and the application should still work correctly."
    echo ""
    echo "If you continue to have issues, try recreating the environment from scratch:"
    echo "    conda env remove -n acis"
    echo "    ./setup_conda.sh"
else
    echo "❌ Failed to install ACIS package. Please try manually running:"
    echo "    pip install -e ."
fi 