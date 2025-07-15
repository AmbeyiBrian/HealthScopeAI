-- HealthScopeAI Database Initialization Script
-- This script sets up the initial database schema for production deployment

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create health_data table for storing processed health records
CREATE TABLE IF NOT EXISTS health_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    text_content TEXT NOT NULL,
    health_category VARCHAR(100),
    sentiment_score REAL,
    severity_level VARCHAR(20),
    location VARCHAR(100),
    latitude REAL,
    longitude REAL,
    source VARCHAR(50),
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_health_data_category ON health_data(health_category);
CREATE INDEX IF NOT EXISTS idx_health_data_location ON health_data(location);
CREATE INDEX IF NOT EXISTS idx_health_data_processed_at ON health_data(processed_at);
CREATE INDEX IF NOT EXISTS idx_health_data_sentiment ON health_data(sentiment_score);

-- Create model_metrics table for tracking model performance
CREATE TABLE IF NOT EXISTS model_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_version VARCHAR(50) NOT NULL,
    accuracy REAL,
    precision_score REAL,
    recall_score REAL,
    f1_score REAL,
    training_date TIMESTAMP,
    evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create predictions table for logging predictions
CREATE TABLE IF NOT EXISTS predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    input_text TEXT NOT NULL,
    predicted_category VARCHAR(100),
    confidence_score REAL,
    model_version VARCHAR(50),
    prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create system_metrics table for monitoring
CREATE TABLE IF NOT EXISTS system_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(100) NOT NULL,
    metric_value REAL,
    metric_unit VARCHAR(20),
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create user_sessions table (optional, for analytics)
CREATE TABLE IF NOT EXISTS user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255) UNIQUE,
    user_agent TEXT,
    ip_address INET,
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    page_views INTEGER DEFAULT 0
);

-- Insert sample data (optional)
INSERT INTO model_metrics (model_version, accuracy, precision_score, recall_score, f1_score, training_date)
VALUES ('v1.0.0', 0.95, 0.94, 0.92, 0.93, CURRENT_TIMESTAMP)
ON CONFLICT DO NOTHING;

-- Create a view for recent health trends
CREATE OR REPLACE VIEW recent_health_trends AS
SELECT 
    health_category,
    location,
    COUNT(*) as mention_count,
    AVG(sentiment_score) as avg_sentiment,
    DATE(processed_at) as trend_date
FROM health_data
WHERE processed_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY health_category, location, DATE(processed_at)
ORDER BY trend_date DESC, mention_count DESC;

-- Create a function to clean old data (data retention)
CREATE OR REPLACE FUNCTION clean_old_data()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    -- Delete health_data older than 2 years
    DELETE FROM health_data 
    WHERE processed_at < CURRENT_DATE - INTERVAL '2 years';
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    -- Delete old predictions (keep only last 6 months)
    DELETE FROM predictions 
    WHERE prediction_time < CURRENT_DATE - INTERVAL '6 months';
    
    -- Delete old system metrics (keep only last 1 year)
    DELETE FROM system_metrics 
    WHERE recorded_at < CURRENT_DATE - INTERVAL '1 year';
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions to application user
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO healthscope;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO healthscope;

-- Create healthscope user if it doesn't exist
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'healthscope') THEN
        CREATE ROLE healthscope LOGIN PASSWORD 'healthscope_secure_pass';
    END IF;
END
$$;
