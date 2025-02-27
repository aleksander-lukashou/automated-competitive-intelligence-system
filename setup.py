from setuptools import setup, find_packages
import os
import re

# Read version without importing
with open(os.path.join('acis', 'version.py'), 'r') as f:
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if version_match:
        version = version_match.group(1)
    else:
        raise RuntimeError("Unable to find version string in 'acis/version.py'")

# Read long description from README
with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name="acis",
    version=version,
    packages=find_packages(),
    description="Automated Competitive Intelligence System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="ACIS Team",
    author_email="example@example.com",
    url="https://github.com/yourusername/acis",
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn>=0.23.0",
        "pydantic>=2.4.0",
        "pydantic-settings>=2.0.0",
        "streamlit>=1.28.0",
        "plotly>=5.18.0",
        "pandas>=2.1.0",
        "numpy>=1.24.0",
        "langchain>=0.1.0",
        "langchain-community>=0.0.10",
        "openai>=1.3.0",
        "tiktoken>=0.5.0",
        "aiohttp>=3.8.0",
        "requests>=2.31.0",
        "beautifulsoup4>=4.12.0",
        "lxml>=4.9.0",
        "python-dateutil>=2.8.0",
        "networkx>=3.2.0",
        "python-dotenv>=1.0.0",
    ],
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
) 