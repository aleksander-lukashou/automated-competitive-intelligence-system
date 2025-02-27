# GitHub Setup Instructions

Follow these steps to push your ACIS project to a new GitHub repository:

## Prerequisites

1. GitHub account
2. Git installed on your local machine
3. Git properly configured with your credentials:
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "your.email@example.com"
   ```

## Steps to Create a New GitHub Repository and Push Your Code

### 1. Create a New Repository on GitHub

1. Go to [GitHub](https://github.com/) and sign in to your account
2. Click on the "+" icon in the upper right corner and select "New repository"
3. Enter a repository name (e.g., "acis" or "automated-competitive-intelligence-system")
4. Optionally add a description
5. Choose public or private visibility
6. **Do not** initialize with a README, .gitignore, or license (since you already have these files)
7. Click "Create repository"

### 2. Connect Your Local Repository to GitHub

After creating the repository, GitHub will show instructions. Follow the ones for "push an existing repository":

```bash
# Make sure you're in your project directory
cd path/to/your/acis/project

# If you haven't already initialized Git
git init

# Add all files to staging
git add .

# Commit the files
git commit -m "Initial commit"

# Add the GitHub repository as a remote
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git

# Push your code to GitHub
git push -u origin main
```

Note: If your default branch is named "master" instead of "main", use:

```bash
git push -u origin master
```

### 3. Verify Your Repository

After pushing, refresh your GitHub repository page to see your code. 

## Additional Tips

### Creating a Release

1. On your GitHub repository page, click on "Releases" in the right sidebar
2. Click "Create a new release"
3. Enter a tag version (e.g., "v0.1.0" - should match your `__version__` in `acis/version.py`)
4. Enter a release title and description
5. Optionally upload any additional files
6. Click "Publish release"

### Setting Up GitHub Pages (for Documentation)

If you want to set up documentation:

1. Go to repository "Settings"
2. Scroll down to "GitHub Pages" section
3. Select the source branch and folder
4. Click "Save"

### Using GitHub Actions

To set up automated testing or deployment:

1. Create a `.github/workflows` directory in your repository
2. Add workflow YAML files for your CI/CD pipelines

Example workflow for Python testing:
```yaml
name: Python Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.10'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
        pip install pytest
        pip install -e .
    - name: Test with pytest
      run: |
        pytest
``` 