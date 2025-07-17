#!/usr/bin/env python3
"""
Script to execute the data collection notebook programmatically for demo deployment.
This generates the required data files that the Streamlit app needs.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_notebook():
    """Execute the data collection notebook to generate demo data."""
    print("🚀 Starting notebook execution for demo data generation...")
    
    notebook_path = "notebooks/01_data_collection.ipynb"
    
    if not os.path.exists(notebook_path):
        print(f"❌ Notebook not found: {notebook_path}")
        sys.exit(1)
    
    try:
        # Use jupyter nbconvert to execute the notebook
        cmd = [
            "python", "-m", "jupyter", "nbconvert", 
            "--to", "notebook", 
            "--execute", 
            notebook_path,
            "--output", "/tmp/executed_notebook.ipynb"
        ]
        
        print(f"📝 Executing command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Notebook executed successfully!")
            
            # Check if required data files were generated
            required_files = [
                "data/processed/dashboard_data.csv",
                "data/processed/health_data.geojson"
            ]
            
            missing_files = []
            for file_path in required_files:
                if not os.path.exists(file_path):
                    missing_files.append(file_path)
            
            if missing_files:
                print(f"⚠️  Warning: Some expected files were not generated: {missing_files}")
            else:
                print("✅ All required data files generated successfully!")
                
        else:
            print(f"❌ Notebook execution failed with return code: {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            sys.exit(1)
            
    except subprocess.TimeoutExpired:
        print("❌ Notebook execution timed out after 5 minutes")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error executing notebook: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_notebook()
