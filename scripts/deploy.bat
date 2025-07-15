@echo off
REM HealthScopeAI Windows Deployment Script
REM Usage: deploy.bat [environment] [version]

setlocal enabledelayedexpansion

REM Default values
set ENVIRONMENT=%1
if "%ENVIRONMENT%"=="" set ENVIRONMENT=production

set VERSION=%2
if "%VERSION%"=="" set VERSION=latest

set PROJECT_NAME=healthscope-ai

echo [INFO] Starting HealthScopeAI deployment...
echo [INFO] Environment: %ENVIRONMENT%
echo [INFO] Version: %VERSION%

REM Check if Docker is installed
echo [INFO] Checking Docker installation...
docker --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not installed. Please install Docker Desktop first.
    exit /b 1
)

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not running. Please start Docker Desktop first.
    exit /b 1
)

echo [SUCCESS] Docker is ready

REM Create necessary directories
echo [INFO] Creating necessary directories...
if not exist "logs" mkdir logs
if not exist "data\raw" mkdir data\raw
if not exist "data\processed" mkdir data\processed
if not exist "models" mkdir models
if not exist "nginx" mkdir nginx
if not exist "monitoring" mkdir monitoring

echo [SUCCESS] Directories created

REM Set up environment
echo [INFO] Setting up environment for: %ENVIRONMENT%
if not exist ".env.%ENVIRONMENT%" (
    echo [WARNING] Environment file .env.%ENVIRONMENT% not found. Creating default...
    (
        echo # HealthScopeAI Environment Configuration
        echo ENVIRONMENT=%ENVIRONMENT%
        echo POSTGRES_PASSWORD=healthscope_secure_pass_123
        echo REDIS_PASSWORD=redis_secure_pass_456
        echo SECRET_KEY=your_secret_key_here
        echo DEBUG=false
        echo ALLOWED_HOSTS=localhost,127.0.0.1
        echo DATABASE_URL=postgresql://healthscope:${POSTGRES_PASSWORD}@postgres:5432/healthscope_ai
        echo REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
    ) > .env.%ENVIRONMENT%
    echo [SUCCESS] Created default environment file: .env.%ENVIRONMENT%
)

copy .env.%ENVIRONMENT% .env >nul

REM Build Docker image
echo [INFO] Building Docker image for version: %VERSION%
docker build -t %PROJECT_NAME%:%VERSION% -t %PROJECT_NAME%:latest .
if errorlevel 1 (
    echo [ERROR] Docker build failed
    exit /b 1
)
echo [SUCCESS] Docker image built successfully

REM Stop existing services
echo [INFO] Stopping existing services...
docker-compose down --remove-orphans >nul 2>&1

REM Start services
echo [INFO] Starting HealthScopeAI services...
if "%ENVIRONMENT%"=="production" (
    docker-compose --profile production up -d
) else (
    docker-compose up -d
)

if errorlevel 1 (
    echo [ERROR] Failed to start services
    exit /b 1
)

echo [SUCCESS] Services started successfully

REM Wait for services to be ready
echo [INFO] Waiting for services to be ready...
timeout /t 30 >nul

REM Health check
echo [INFO] Performing health check...
set /a attempt=1
set /a max_attempts=10

:health_check_loop
curl -f http://localhost:8501/_stcore/health >nul 2>&1
if not errorlevel 1 (
    echo [SUCCESS] Application is healthy
    goto health_check_done
)

echo [INFO] Attempt %attempt%/%max_attempts%: Waiting for application...
timeout /t 10 >nul
set /a attempt+=1

if %attempt% leq %max_attempts% goto health_check_loop

echo [ERROR] Health check failed. Application is not responding.
echo [INFO] Check logs: docker-compose logs
exit /b 1

:health_check_done

REM Show status
echo [INFO] Service status:
docker-compose ps

echo.
echo [INFO] Application URLs:
echo   - Main Application: http://localhost:8501
echo   - Database: localhost:5432
echo   - Redis: localhost:6379

if "%ENVIRONMENT%"=="production" (
    echo   - Web Server: http://localhost:80
    echo   - Monitoring: http://localhost:9090 ^(if enabled^)
)

echo.
echo [SUCCESS] 🎉 HealthScopeAI deployment completed successfully!
echo.
echo Next steps:
echo 1. Access the application at http://localhost:8501
echo 2. Check logs: docker-compose logs -f
echo 3. Monitor services: docker-compose ps
echo 4. Stop services: docker-compose down

exit /b 0
