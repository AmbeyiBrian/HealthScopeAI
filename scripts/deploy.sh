#!/bin/bash
# HealthScopeAI Production Deployment Script
# Usage: ./deploy.sh [environment] [version]

set -e

# Default values
ENVIRONMENT=${1:-production}
VERSION=${2:-latest}
PROJECT_NAME="healthscope-ai"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if Docker is installed and running
check_docker() {
    log "Checking Docker installation..."
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        error "Docker is not running. Please start Docker first."
        exit 1
    fi
    
    success "Docker is ready"
}

# Check if Docker Compose is available
check_docker_compose() {
    log "Checking Docker Compose..."
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        error "Docker Compose is not available. Please install Docker Compose."
        exit 1
    fi
    success "Docker Compose is ready"
}

# Create necessary directories
create_directories() {
    log "Creating necessary directories..."
    mkdir -p logs data/raw data/processed models nginx monitoring
    success "Directories created"
}

# Set up environment variables
setup_environment() {
    log "Setting up environment for: $ENVIRONMENT"
    
    if [ ! -f ".env.${ENVIRONMENT}" ]; then
        warning "Environment file .env.${ENVIRONMENT} not found. Creating default..."
        cat > .env.${ENVIRONMENT} << EOF
# HealthScopeAI Environment Configuration
ENVIRONMENT=${ENVIRONMENT}
POSTGRES_PASSWORD=healthscope_secure_pass_$(openssl rand -hex 8)
REDIS_PASSWORD=redis_secure_pass_$(openssl rand -hex 8)
SECRET_KEY=$(openssl rand -hex 32)
DEBUG=false
ALLOWED_HOSTS=localhost,127.0.0.1
DATABASE_URL=postgresql://healthscope:\${POSTGRES_PASSWORD}@postgres:5432/healthscope_ai
REDIS_URL=redis://:\${REDIS_PASSWORD}@redis:6379/0
EOF
        success "Created default environment file: .env.${ENVIRONMENT}"
    fi
    
    # Copy environment file
    cp .env.${ENVIRONMENT} .env
}

# Build Docker image
build_image() {
    log "Building Docker image for version: $VERSION"
    docker build -t ${PROJECT_NAME}:${VERSION} -t ${PROJECT_NAME}:latest .
    success "Docker image built successfully"
}

# Run tests in container
run_tests() {
    log "Running tests in Docker container..."
    docker run --rm -v $(pwd)/tests:/app/tests ${PROJECT_NAME}:${VERSION} python -m pytest tests/ -v
    success "All tests passed"
}

# Start services
start_services() {
    log "Starting HealthScopeAI services..."
    
    # Stop existing services
    docker-compose down --remove-orphans 2>/dev/null || true
    
    # Start services based on environment
    if [ "$ENVIRONMENT" = "production" ]; then
        docker-compose --profile production up -d
    else
        docker-compose up -d
    fi
    
    success "Services started successfully"
}

# Health check
health_check() {
    log "Performing health check..."
    
    # Wait for services to be ready
    log "Waiting for services to be ready..."
    sleep 30
    
    # Check application health
    max_attempts=30
    attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f http://localhost:8501/_stcore/health &> /dev/null; then
            success "Application is healthy"
            return 0
        fi
        
        log "Attempt $attempt/$max_attempts: Waiting for application..."
        sleep 10
        ((attempt++))
    done
    
    error "Health check failed. Application is not responding."
    return 1
}

# Show status
show_status() {
    log "Service status:"
    docker-compose ps
    
    log "\nApplication URLs:"
    echo "  - Main Application: http://localhost:8501"
    echo "  - Database: localhost:5432"
    echo "  - Redis: localhost:6379"
    
    if [ "$ENVIRONMENT" = "production" ]; then
        echo "  - Web Server: http://localhost:80"
        echo "  - Monitoring: http://localhost:9090 (if enabled)"
    fi
}

# Backup data
backup_data() {
    log "Creating data backup..."
    timestamp=$(date +%Y%m%d_%H%M%S)
    backup_dir="backups/${timestamp}"
    mkdir -p "$backup_dir"
    
    # Backup database
    docker-compose exec -T postgres pg_dump -U healthscope healthscope_ai > "${backup_dir}/database_backup.sql"
    
    # Backup application data
    cp -r data/ "${backup_dir}/"
    cp -r models/ "${backup_dir}/"
    
    success "Backup created: $backup_dir"
}

# Main deployment function
deploy() {
    log "Starting HealthScopeAI deployment..."
    log "Environment: $ENVIRONMENT"
    log "Version: $VERSION"
    
    # Pre-deployment checks
    check_docker
    check_docker_compose
    create_directories
    setup_environment
    
    # Build and test
    build_image
    run_tests
    
    # Create backup if production
    if [ "$ENVIRONMENT" = "production" ]; then
        backup_data
    fi
    
    # Deploy services
    start_services
    
    # Verify deployment
    if health_check; then
        show_status
        success "🎉 HealthScopeAI deployment completed successfully!"
        
        log "\nNext steps:"
        echo "1. Access the application at http://localhost:8501"
        echo "2. Check logs: docker-compose logs -f"
        echo "3. Monitor services: docker-compose ps"
        echo "4. Stop services: docker-compose down"
    else
        error "Deployment failed during health check"
        log "Check logs: docker-compose logs"
        exit 1
    fi
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy")
        deploy
        ;;
    "stop")
        log "Stopping HealthScopeAI services..."
        docker-compose down
        success "Services stopped"
        ;;
    "restart")
        log "Restarting HealthScopeAI services..."
        docker-compose restart
        success "Services restarted"
        ;;
    "logs")
        docker-compose logs -f
        ;;
    "status")
        show_status
        ;;
    "backup")
        backup_data
        ;;
    "clean")
        log "Cleaning up Docker resources..."
        docker-compose down -v --remove-orphans
        docker system prune -f
        success "Cleanup completed"
        ;;
    *)
        echo "Usage: $0 {deploy|stop|restart|logs|status|backup|clean} [environment] [version]"
        echo ""
        echo "Commands:"
        echo "  deploy    - Deploy the application (default)"
        echo "  stop      - Stop all services"
        echo "  restart   - Restart all services"
        echo "  logs      - Show service logs"
        echo "  status    - Show service status"
        echo "  backup    - Create data backup"
        echo "  clean     - Clean up Docker resources"
        echo ""
        echo "Environments: development, staging, production"
        echo "Default environment: production"
        exit 1
        ;;
esac
