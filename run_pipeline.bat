@echo off
echo.
echo 🌍 HealthScopeAI - Complete Pipeline Runner
echo ==========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH
    pause
    exit /b 1
)

echo ✅ Python is available

REM Change to project directory
cd /d "%~dp0"

REM Step 1: Data Collection
echo.
echo 📊 Step 1: Running data collection...
echo --------------------------------------
python src/data_collection.py
if %errorlevel% neq 0 (
    echo ❌ Data collection failed
    pause
    exit /b 1
)
echo ✅ Data collection completed

REM Step 2: Data Preprocessing
echo.
echo 🔧 Step 2: Running data preprocessing...
echo ----------------------------------------
python src/preprocessing.py
if %errorlevel% neq 0 (
    echo ❌ Data preprocessing failed
    pause
    exit /b 1
)
echo ✅ Data preprocessing completed

REM Step 3: Model Training
echo.
echo 🤖 Step 3: Training machine learning model...
echo ----------------------------------------------
python src/model.py
if %errorlevel% neq 0 (
    echo ❌ Model training failed
    pause
    exit /b 1
)
echo ✅ Model training completed

REM Step 4: Geographic Analysis
echo.
echo 🗺️ Step 4: Running geographic analysis...
echo ------------------------------------------
python src/geo_analysis.py
if %errorlevel% neq 0 (
    echo ❌ Geographic analysis failed
    pause
    exit /b 1
)
echo ✅ Geographic analysis completed

REM Summary
echo.
echo 🎉 Pipeline completed successfully!
echo ==================================
echo.
echo 📊 Generated files:
echo • Raw data in data/raw/
echo • Processed data in data/processed/
echo • Trained models in models/
echo • Analysis results in screenshots/
echo.
echo 🚀 Next steps:
echo • Run 'streamlit run streamlit_app/app.py' to start the dashboard
echo • Open notebooks/ folder to explore Jupyter notebooks
echo • Check README.md for detailed documentation
echo.
echo 🌟 HealthScopeAI is ready to monitor health trends!
echo.
pause
