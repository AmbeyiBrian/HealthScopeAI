#!/usr/bin/env python3
"""
Simple test runner for HealthScopeAI tests.
"""

import sys
import os
import subprocess
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def run_simple_test():
    """Run a simple test to verify our setup."""
    print("🧪 HealthScopeAI Test Runner")
    print("=" * 50)
    
    try:
        # Test basic imports
        print("✓ Testing basic imports...")
        import pandas as pd
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        print("  ✓ Core dependencies imported successfully")
        
        # Test our modules
        print("✓ Testing HealthScopeAI modules...")
        try:
            from preprocessing import DataPreprocessor
            print("  ✓ DataPreprocessor imported")
            
            from model import HealthClassifier
            print("  ✓ HealthClassifier imported")
            
            from data_collection import DataCollector
            print("  ✓ DataCollector imported")
            
            from geo_analysis import GeoAnalyzer
            print("  ✓ GeoAnalyzer imported")
            
        except ImportError as e:
            print(f"  ❌ Module import failed: {e}")
            return False
        
        # Test basic functionality
        print("✓ Testing basic functionality...")
        try:
            # Test DataPreprocessor
            preprocessor = DataPreprocessor()
            test_text = "I feel sick today"
            cleaned = preprocessor.clean_text(test_text)
            print(f"  ✓ Text preprocessing: '{test_text}' -> '{cleaned}'")
            
            # Test DataCollector
            collector = DataCollector()
            sample_data = collector.generate_synthetic_data(num_samples=5)
            print(f"  ✓ Data collection: Generated {len(sample_data)} samples")
            
            # Test GeoAnalyzer
            analyzer = GeoAnalyzer()
            aggregated = analyzer.aggregate_health_data(sample_data)
            print(f"  ✓ Geo analysis: Aggregated data for {len(aggregated)} locations")
            
        except Exception as e:
            print(f"  ❌ Functionality test failed: {e}")
            return False
        
        print("\n🎉 All tests passed! System is ready for comprehensive testing.")
        return True
        
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        return False

def run_pytest():
    """Run pytest if available."""
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "tests/", "-v", "--tb=short", "--no-header"
        ], capture_output=True, text=True, cwd=project_root)
        
        if result.returncode == 0:
            print("\n🧪 Pytest Results:")
            print("=" * 50)
            print(result.stdout)
        else:
            print("\n⚠️  Some tests failed:")
            print("=" * 50)
            print(result.stdout)
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ Could not run pytest: {e}")

if __name__ == "__main__":
    print("Starting HealthScopeAI test validation...\n")
    
    # Run simple functionality tests first
    if run_simple_test():
        print("\n" + "=" * 50)
        print("🚀 Running comprehensive test suite...")
        run_pytest()
    else:
        print("\n❌ Basic tests failed. Please fix issues before running full suite.")
        sys.exit(1)
