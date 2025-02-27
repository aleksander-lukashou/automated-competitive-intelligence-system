#!/bin/bash

# This script sets up the conda environment for ACIS

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed. Please install Miniconda or Anaconda first."
    echo "Visit: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Check if environment exists
ENV_EXISTS=$(conda env list | grep -w "acis" || true)

if [ -n "$ENV_EXISTS" ]; then
    echo "Environment 'acis' already exists."
    read -p "Do you want to update it? (y/n): " ANSWER
    if [[ $ANSWER =~ ^[Yy]$ ]]; then
        echo "Updating environment 'acis'..."
        conda env update -f environment.yml
        UPDATE_RESULT=$?
        if [ $UPDATE_RESULT -eq 0 ]; then
            echo "✅ Environment 'acis' updated successfully!"
        else
            echo "❌ Failed to update environment. Check the error messages above."
            exit 1
        fi
    else
        echo "Skipping environment update."
    fi
else
    # Create conda environment from environment.yml
    echo "Creating conda environment for ACIS..."
    conda env create -f environment.yml
    
    # Check if environment was created successfully
    CREATE_RESULT=$?
    if [ $CREATE_RESULT -eq 0 ]; then
        echo "✅ Environment 'acis' created successfully!"
    else
        echo "❌ Failed to create environment. Check the error messages above."
        exit 1
    fi
fi

# Ask user if they want to install the package
echo ""
echo "Now we need to install the ACIS package in development mode."
read -p "Install ACIS package now? (y/n, default: y): " INSTALL_ANSWER
INSTALL_ANSWER=${INSTALL_ANSWER:-y}

if [[ $INSTALL_ANSWER =~ ^[Yy]$ ]]; then
    # Activate the environment for package installation
    echo "Activating 'acis' environment for package installation..."
    
    # Using source to ensure conda activate works in this script
    # Note: This might not work in all shells, a fallback message is provided
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
    conda activate acis 2>/dev/null
    
    if [ "$CONDA_DEFAULT_ENV" = "acis" ]; then
        echo "Installing ACIS package in development mode..."
        pip install -e .
        
        if [ $? -eq 0 ]; then
            echo "✅ ACIS package installed successfully!"
        else
            echo "❌ Failed to install ACIS package. You can try manually later with:"
            echo "    conda activate acis"
            echo "    pip install -e ."
        fi
    else
        echo "Could not automatically activate the environment."
        echo "Please manually run the following commands to complete setup:"
        echo "    conda activate acis"
        echo "    pip install -e ."
    fi
else
    echo "Skipping package installation."
    echo "You will need to manually install the package later with:"
    echo "    conda activate acis"
    echo "    pip install -e ."
fi

echo ""
echo "To activate the environment, run:"
echo "    conda activate acis"
echo ""
echo "Next steps:"
echo "1. Copy .env.example to .env and configure your environment variables"
echo "2. Run the application with: python app.py" 