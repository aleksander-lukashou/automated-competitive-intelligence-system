.PHONY: setup install clean test lint format

# Default target
all: install

# Setup conda environment
setup:
	./setup_conda.sh

# Install package in development mode
install:
	pip install -e .

# Fix dependencies
fix-deps:
	./fix_dependencies.sh

# Run tests
test:
	pytest

# Run with coverage
coverage:
	pytest --cov=acis tests/

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name "*.pyc" -delete

# Run linting
lint:
	flake8 acis/ tests/

# Format code
format:
	black acis/ tests/
	isort acis/ tests/

# Run the main application
run:
	python app.py

# Run only the dashboard
run-dashboard:
	streamlit run acis/dashboard/app.py 