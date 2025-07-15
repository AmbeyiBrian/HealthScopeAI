# 🚀 HealthScopeAI Deployment Guide

This guide provides comprehensive instructions for deploying HealthScopeAI in various environments using Docker and Docker Compose.

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Environment Configuration](#environment-configuration)
- [Deployment Options](#deployment-options)
- [Monitoring & Maintenance](#monitoring--maintenance)
- [Troubleshooting](#troubleshooting)
- [Security Considerations](#security-considerations)

## 🔧 Prerequisites

### System Requirements

**Minimum Requirements:**
- CPU: 2 cores
- RAM: 4GB
- Storage: 10GB free space
- OS: Windows 10+, macOS 10.14+, or Linux

**Recommended for Production:**
- CPU: 4+ cores
- RAM: 8GB+
- Storage: 50GB+ SSD
- OS: Ubuntu 20.04+ or CentOS 8+

### Software Dependencies

1. **Docker** (v20.10+)
   ```bash
   # Install Docker (Linux)
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   
   # Or download Docker Desktop for Windows/macOS
   ```

2. **Docker Compose** (v2.0+)
   ```bash
   # Usually included with Docker Desktop
   # For Linux, install separately:
   sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
   sudo chmod +x /usr/local/bin/docker-compose
   ```

3. **Git** (for cloning repository)
   ```bash
   # Install Git
   sudo apt-get install git  # Ubuntu/Debian
   brew install git          # macOS
   ```

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/AmbeyiBrian/HealthScopeAI.git
cd HealthScopeAI
```

### 2. One-Command Deployment

**Windows:**
```cmd
scripts\deploy.bat
```

**Linux/macOS:**
```bash
chmod +x scripts/deploy.sh
./scripts/deploy.sh
```

### 3. Access Application
- **Main Application:** http://localhost:8501
- **Health Check:** http://localhost:8501/_stcore/health

## ⚙️ Environment Configuration

### Environment Files

Create environment-specific configuration files:

**Development (.env.development):**
```env
ENVIRONMENT=development
POSTGRES_PASSWORD=dev_password
REDIS_PASSWORD=dev_redis_pass
SECRET_KEY=dev_secret_key_123
DEBUG=true
ALLOWED_HOSTS=localhost,127.0.0.1
```

**Production (.env.production):**
```env
ENVIRONMENT=production
POSTGRES_PASSWORD=secure_production_password_123
REDIS_PASSWORD=secure_redis_password_456
SECRET_KEY=super_secure_secret_key_production
DEBUG=false
ALLOWED_HOSTS=yourdomain.com,www.yourdomain.com
DATABASE_URL=postgresql://healthscope:${POSTGRES_PASSWORD}@postgres:5432/healthscope_ai
REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
```

### SSL Configuration (Production)

For HTTPS in production:

1. **Obtain SSL certificates:**
   ```bash
   # Using Let's Encrypt (recommended)
   sudo apt-get install certbot
   sudo certbot certonly --standalone -d yourdomain.com
   ```

2. **Copy certificates:**
   ```bash
   mkdir -p nginx/ssl
   sudo cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem nginx/ssl/cert.pem
   sudo cp /etc/letsencrypt/live/yourdomain.com/privkey.pem nginx/ssl/key.pem
   sudo chown $USER:$USER nginx/ssl/*
   ```

## 🎯 Deployment Options

### Option 1: Basic Development
```bash
# Start with basic services
docker-compose up -d

# Services included:
# - HealthScopeAI App (port 8501)
# - PostgreSQL Database (port 5432)
# - Redis Cache (port 6379)
```

### Option 2: Production with Reverse Proxy
```bash
# Start with production profile
docker-compose --profile production up -d

# Additional services:
# - Nginx Reverse Proxy (ports 80, 443)
# - SSL termination
# - Rate limiting
# - Security headers
```

### Option 3: Full Monitoring Stack
```bash
# Start with monitoring
docker-compose --profile production --profile monitoring up -d

# Additional services:
# - Prometheus (port 9090)
# - Metrics collection
# - Alerting rules
# - Performance monitoring
```

### Option 4: Custom Build
```bash
# Build custom image
docker build -t healthscope-ai:custom .

# Run with custom configuration
docker run -d \
  --name healthscope-app \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  healthscope-ai:custom
```

## 🔍 Monitoring & Maintenance

### Service Management

**Check Status:**
```bash
docker-compose ps
```

**View Logs:**
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f healthscope-app
```

**Restart Services:**
```bash
# Restart all
docker-compose restart

# Restart specific service
docker-compose restart healthscope-app
```

### Health Checks

**Application Health:**
```bash
curl -f http://localhost:8501/_stcore/health
```

**Database Health:**
```bash
docker-compose exec postgres pg_isready -U healthscope
```

**Redis Health:**
```bash
docker-compose exec redis redis-cli ping
```

### Backup & Restore

**Create Backup:**
```bash
# Automated backup
./scripts/deploy.sh backup

# Manual database backup
docker-compose exec postgres pg_dump -U healthscope healthscope_ai > backup_$(date +%Y%m%d).sql
```

**Restore Database:**
```bash
# Restore from backup
docker-compose exec -T postgres psql -U healthscope healthscope_ai < backup_20250714.sql
```

### Performance Monitoring

**Prometheus Metrics (if monitoring enabled):**
- Application metrics: http://localhost:9090
- Custom dashboards available in `/monitoring/dashboards/`

**Resource Usage:**
```bash
# Container stats
docker stats

# Detailed container info
docker-compose exec healthscope-app top
```

## 🐛 Troubleshooting

### Common Issues

**1. Port Already in Use**
```bash
# Check what's using the port
sudo lsof -i :8501  # Linux/macOS
netstat -ano | findstr :8501  # Windows

# Kill the process or change port in docker-compose.yml
```

**2. Permission Denied (Linux)**
```bash
# Add user to docker group
sudo usermod -aG docker $USER
# Logout and login again
```

**3. Out of Memory**
```bash
# Increase Docker memory limit in Docker Desktop settings
# Or add memory limits to docker-compose.yml:
deploy:
  resources:
    limits:
      memory: 2G
```

**4. Database Connection Failed**
```bash
# Check database logs
docker-compose logs postgres

# Verify database is running
docker-compose ps postgres

# Test connection
docker-compose exec postgres psql -U healthscope -d healthscope_ai -c "SELECT version();"
```

**5. Model Loading Issues**
```bash
# Download required models manually
docker-compose exec healthscope-app python -c "
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
"

docker-compose exec healthscope-app python -m spacy download en_core_web_sm
```

### Debug Mode

**Enable Debug Logging:**
```bash
# Modify docker-compose.yml
environment:
  - DEBUG=true
  - LOG_LEVEL=DEBUG

# Restart services
docker-compose restart
```

**Interactive Shell:**
```bash
# Access container shell
docker-compose exec healthscope-app /bin/bash

# Or start a debug container
docker run -it --rm -v $(pwd):/app healthscope-ai:latest /bin/bash
```

### Performance Issues

**1. Slow Predictions**
```bash
# Check CPU/memory usage
docker stats healthscope-app

# Enable caching
# Add Redis configuration in application
```

**2. Database Slow Queries**
```bash
# Check database performance
docker-compose exec postgres psql -U healthscope -d healthscope_ai -c "
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;
"
```

## 🔒 Security Considerations

### Production Security Checklist

- [ ] **Change default passwords** in environment files
- [ ] **Use strong, unique passwords** (min 16 characters)
- [ ] **Enable SSL/TLS** for all external connections
- [ ] **Configure firewall** to limit exposed ports
- [ ] **Regular security updates** for base images
- [ ] **Monitor security logs** for suspicious activity
- [ ] **Backup encryption** for sensitive data
- [ ] **Access control** with proper user permissions

### Network Security

**Firewall Configuration:**
```bash
# Allow only necessary ports
sudo ufw allow 80    # HTTP
sudo ufw allow 443   # HTTPS
sudo ufw allow 22    # SSH (if needed)
sudo ufw deny 5432   # Block direct database access
sudo ufw deny 6379   # Block direct Redis access
sudo ufw enable
```

**Docker Network Isolation:**
```yaml
# In docker-compose.yml
networks:
  internal:
    driver: bridge
    internal: true
  external:
    driver: bridge
```

### Data Protection

**Encryption at Rest:**
```bash
# Use encrypted volumes
docker volume create --driver local \
  --opt type=none \
  --opt o=bind \
  --opt device=/encrypted/path \
  encrypted_data
```

**Secrets Management:**
```bash
# Use Docker secrets (Swarm mode)
echo "my_secret_password" | docker secret create postgres_password -

# Or use external secret management (recommended)
```

## 📊 Scaling & High Availability

### Horizontal Scaling

**Multiple App Instances:**
```yaml
# In docker-compose.yml
healthscope-app:
  deploy:
    replicas: 3
  # Load balancer will distribute traffic
```

**Database Scaling:**
```yaml
# Read replicas
postgres-replica:
  image: postgres:15-alpine
  environment:
    POSTGRES_MASTER_SERVICE: postgres
    POSTGRES_SLAVE_USER: replica
```

### Load Balancing

**Nginx Load Balancing:**
```nginx
upstream healthscope_backend {
    server healthscope-app-1:8501;
    server healthscope-app-2:8501;
    server healthscope-app-3:8501;
}
```

## 📝 Maintenance Schedule

### Daily Tasks
- [ ] Check application health
- [ ] Monitor error logs
- [ ] Verify backup completion

### Weekly Tasks
- [ ] Review performance metrics
- [ ] Update security patches
- [ ] Clean old log files

### Monthly Tasks
- [ ] Database optimization
- [ ] Security audit
- [ ] Capacity planning review

---

## 📞 Support

For issues and questions:

1. **Check logs:** `docker-compose logs -f`
2. **Review documentation:** This README and project docs
3. **GitHub Issues:** https://github.com/AmbeyiBrian/HealthScopeAI/issues
4. **Health Check:** http://localhost:8501/_stcore/health

---

**Last Updated:** July 14, 2025  
**Version:** 1.0  
**Maintainer:** Brian Ambeyi

---

*This deployment guide ensures secure, scalable, and maintainable deployment of HealthScopeAI in any environment.*
