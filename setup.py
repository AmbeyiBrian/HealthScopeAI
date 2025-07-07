#!/usr/bin/env python3
"""
Setup script for HealthScopeAI
Handles initial setup and dependency installation.
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    """Main setup function."""
    print("🌍 Welcome to HealthScopeAI Setup!")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python version: {sys.version}")
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    if not run_command("pip install -r requirements.txt", "Installing Python packages"):
        print("⚠️  Some packages may have failed to install. Please check the output above.")
    
    # Download NLTK data
    print("\n📚 Downloading NLTK data...")
    nltk_commands = [
        "python -c \"import nltk; nltk.download('punkt')\"",
        "python -c \"import nltk; nltk.download('stopwords')\"",
        "python -c \"import nltk; nltk.download('wordnet')\"",
        "python -c \"import nltk; nltk.download('vader_lexicon')\""
    ]
    
    for cmd in nltk_commands:
        run_command(cmd, "Downloading NLTK data")
    
    # Download spaCy model
    print("\n🔤 Downloading spaCy model...")
    run_command("python -m spacy download en_core_web_sm", "Downloading spaCy English model")
    
    # Create necessary directories
    print("\n📁 Creating directories...")
    directories = [
        "data/raw",
        "data/processed", 
        "models",
        "screenshots",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Create environment file template
    print("\n🔐 Creating environment file template...")
    env_template = """# HealthScopeAI Environment Variables
# Copy this file to .env and fill in your API keys

# Twitter API (optional)
TWITTER_BEARER_TOKEN=your_twitter_bearer_token_here

# Reddit API (optional)
REDDIT_CLIENT_ID=your_reddit_client_id_here
REDDIT_CLIENT_SECRET=your_reddit_client_secret_here
REDDIT_USER_AGENT=HealthScopeAI/1.0

# Database (optional)
DATABASE_URL=sqlite:///healthscope.db

# Other settings
DEBUG=True
LOG_LEVEL=INFO
"""
    
    with open('.env.template', 'w') as f:
        f.write(env_template)
    
    print("✅ Created .env.template file")
    
    # Test imports
    print("\n🧪 Testing imports...")
    test_imports = [
        "import pandas as pd",
        "import numpy as np",
        "import sklearn",
        "import nltk",
        "import matplotlib.pyplot as plt",
        "import seaborn as sns",
        "import plotly.express as px"
    ]
    
    for import_cmd in test_imports:
        try:
            exec(import_cmd)
            module_name = import_cmd.split()[-1].split('.')[0]
            print(f"✅ {module_name} imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import {import_cmd}: {e}")
    
    # Final setup
    print("\n🚀 Final setup steps...")
    
    # Run initial data collection
    print("🔍 Running initial data collection...")
    if Path("src/data_collection.py").exists():
        run_command("python src/data_collection.py", "Initial data collection")
    
    print("\n🎉 Setup completed!")
    print("=" * 50)
    print("📋 Next steps:")
    print("1. Copy .env.template to .env and add your API keys (optional)")
    print("2. Run 'jupyter notebook' to open the analysis notebooks")
    print("3. Run 'streamlit run streamlit_app/app.py' to start the dashboard")
    print("4. Check the README.md for detailed usage instructions")
    print("\n🌟 Happy analyzing with HealthScopeAI!")

if __name__ == "__main__":
    main()
