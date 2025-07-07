# 🚀 HealthScopeAI Quick Start Guide

Welcome to HealthScopeAI! This guide will help you get up and running quickly.

## 📋 Prerequisites

- Python 3.8 or higher
- Git (optional, for cloning)
- Internet connection (for downloading dependencies)

## 🔧 Installation

### Option 1: Automated Setup (Recommended)
```bash
# Clone or download the project
git clone https://github.com/yourusername/healthscope-ai.git
cd healthscope-ai

# Run the setup script
python setup.py
```

### Option 2: Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Download spaCy model
python -m spacy download en_core_web_sm

# Create directories
mkdir -p data/raw data/processed models screenshots
```

## 🎯 Quick Start

### 1. Run the Complete Pipeline
```bash
# On Windows
run_pipeline.bat

# On macOS/Linux
python src/data_collection.py
python src/preprocessing.py
python src/model.py
python src/geo_analysis.py
```

### 2. Launch the Dashboard
```bash
streamlit run streamlit_app/app.py
```

### 3. Explore the Notebooks
```bash
jupyter notebook
# Open notebooks/01_data_collection.ipynb
```

## 📊 Understanding the Output

After running the pipeline, you'll have:

- **Raw Data**: `data/raw/` - Original collected data
- **Processed Data**: `data/processed/` - Cleaned and feature-engineered data
- **Models**: `models/` - Trained machine learning models
- **Analysis**: `screenshots/` - Generated visualizations and maps

## 🌐 Dashboard Features

The Streamlit dashboard includes:
- **Overview**: Key metrics and system status
- **Map View**: Geographic visualization of health trends
- **Time Trends**: Time series analysis
- **Text Classifier**: Interactive text classification
- **Alerts**: Health alert system
- **Analytics**: Detailed performance metrics

## 🔍 Sample Usage

### Classify Text
```python
from src.model import HealthClassifier

# Load trained model
classifier = HealthClassifier.load_model('models/health_classifier.pkl')

# Classify text
result = classifier.predict_single("I've been feeling anxious lately")
print(f"Health-related: {result['is_health_related']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Analyze Geographic Trends
```python
from src.geo_analysis import GeoAnalyzer
import pandas as pd

# Load data
data = pd.read_csv('data/processed/processed_data.csv')

# Analyze
analyzer = GeoAnalyzer()
aggregated = analyzer.aggregate_health_data(data)

# Create map
map_viz = analyzer.create_choropleth_map(aggregated)
map_viz.save('health_trends_map.html')
```

## 🎨 Customization

### Adding New Health Keywords
Edit `src/preprocessing.py`:
```python
self.health_keywords = {
    'mental_health': ['anxiety', 'depression', 'stress', 'your_keywords'],
    'physical_health': ['flu', 'fever', 'pain', 'your_keywords']
}
```

### Changing Geographic Regions
Edit `src/geo_analysis.py`:
```python
self.location_coordinates = {
    'your_city': {'lat': -1.2921, 'lon': 36.8219},
    # Add more locations
}
```

## 📁 Project Structure

```
HealthScopeAI/
├── data/                    # Data storage
├── models/                  # Trained models
├── notebooks/              # Jupyter notebooks
├── src/                    # Source code
├── streamlit_app/          # Dashboard
├── screenshots/            # Generated visualizations
├── requirements.txt        # Dependencies
├── setup.py               # Setup script
├── run_pipeline.bat       # Windows pipeline runner
└── README.md              # Documentation
```

## 🆘 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install -r requirements.txt
   ```

2. **NLTK Data Missing**
   ```bash
   python -c "import nltk; nltk.download('all')"
   ```

3. **spaCy Model Missing**
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Permission Errors**
   - Run as administrator (Windows)
   - Use `sudo` (macOS/Linux)

### Getting Help

- Check the [README.md](README.md) for detailed documentation
- Review the Jupyter notebooks in `notebooks/`
- Open an issue on GitHub (if applicable)

## 🌟 Next Steps

1. **Customize Keywords**: Add domain-specific health terms
2. **Add Data Sources**: Integrate with real social media APIs
3. **Extend Geography**: Add more regions and cities
4. **Improve Models**: Experiment with different algorithms
5. **Deploy**: Host on cloud platforms

## 🎯 Use Cases

- **Public Health Monitoring**: Track disease outbreaks
- **Mental Health Awareness**: Monitor community mental health
- **Resource Planning**: Allocate medical resources
- **Research**: Study health trends and patterns
- **Early Warning**: Detect health emergencies

---

**Happy monitoring with HealthScopeAI!** 🌍

*"Giving Public Health a Social Pulse."*
