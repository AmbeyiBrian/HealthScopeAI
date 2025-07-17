# HealthScopeAI - Production Docker Image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed models screenshots

# Download required NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Generate demo data for deployment (with error handling)
RUN python generate_demo_data.py || echo "Data generation failed, using fallback" && \
    if [ ! -f "data/processed/dashboard_data.csv" ]; then \
        echo "Creating minimal fallback data..." && \
        mkdir -p data/processed && \
        echo "text,timestamp,location,source,is_health_related,category,sentiment,latitude,longitude,label,date,hour,day_of_week" > data/processed/dashboard_data.csv && \
        echo "Sample health post,2025-07-17T12:00:00,Nairobi,demo,1,physical_health,neutral,-1.2921,36.8219,1,2025-07-17,12,Wednesday" >> data/processed/dashboard_data.csv && \
        echo '{"type":"FeatureCollection","features":[{"type":"Feature","properties":{"name":"Nairobi","health_mentions":1,"total_posts":1,"health_ratio":1.0,"risk_level":"medium"},"geometry":{"type":"Point","coordinates":[36.8219,-1.2921]}}]}' > data/processed/health_data.geojson; \
    fi

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the application
CMD ["streamlit", "run", "streamlit_app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
