# Docker Guide - Gemini Deep Research Agent

Run the Gemini Deep Research Agent using Docker for easy deployment and consistent environment across different systems.

## 🚀 Quick Start

### Prerequisites

- Docker installed on your system
- Gemini API key from [Google AI Studio](https://ai.google.dev/)

### Using Pre-built Image

```bash
# Pull the latest image
docker pull betashow/gemini-deep-research:latest

# Run with your API key
docker run -p 5357:5357 \
  -e GEMINI_API_KEY=your_gemini_api_key_here \
  betashow/gemini-deep-research:latest
```

The API will be available at **http://localhost:5357**

## 🐳 Docker Installation

### Install Docker

#### On macOS:
```bash
# Install Docker Desktop
brew install --cask docker
# Or download from https://docker.com/products/docker-desktop
```

#### On Ubuntu/Debian:
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group (logout/login required)
sudo usermod -aG docker $USER
```

#### On Windows:
- Download Docker Desktop from https://docker.com/products/docker-desktop
- Follow the installation wizard
- Enable WSL 2 backend if prompted

### Verify Installation:
```bash
docker --version
docker run hello-world
```

## 📦 Using the Docker Image

### Basic Usage
```bash
# Run with minimal configuration
docker run -p 5357:5357 \
  -e GEMINI_API_KEY=your_key \
  betashow/gemini-deep-research
```

### Advanced Configuration
```bash
# Run with custom settings
docker run -p 5357:5357 \
  -e GEMINI_API_KEY=your_key \
  -e MAX_SEARCHES_PER_TASK=10 \
  -e MAX_CONCURRENT_CRAWLS=5 \
  -e AI_POLISH_CONTENT=2 \
  -e GEMINI_MODEL=gemini-2.5-pro \
  -e DEBUG_MODE=false \
  --name research-agent \
  betashow/gemini-deep-research
```

### Running in Background
```bash
# Run as daemon (background)
docker run -d -p 5357:5357 \
  -e GEMINI_API_KEY=your_key \
  --name research-agent \
  --restart unless-stopped \
  betashow/gemini-deep-research

# Check status
docker ps
docker logs research-agent
```

### Using Docker Compose
Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  research-agent:
    image: betashow/gemini-deep-research:latest
    ports:
      - "5357:5357"
    environment:
      - GEMINI_API_KEY=your_gemini_api_key_here
      - MAX_SEARCHES_PER_TASK=10
      - AI_POLISH_CONTENT=2
      - DEBUG_MODE=false
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5357/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
```

Run with Docker Compose:
```bash
docker-compose up -d
```

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | required | Your Gemini API key |
| `API_HOST` | 0.0.0.0 | Server host |
| `API_PORT` | 5357 | Server port |
| `DEBUG_MODE` | false | Enable debug logging |
| `MAX_SEARCHES_PER_TASK` | 10 | Default max searches |
| `MAX_CONCURRENT_CRAWLS` | 5 | Default max crawls |
| `AI_POLISH_CONTENT` | false | Default AI polishing |
| `GEMINI_MODEL` | gemini-2.5-pro | Gemini model |

### Volume Mounts
```bash
# Mount custom prompts
docker run -p 5357:5357 \
  -e GEMINI_API_KEY=your_key \
  -v $(pwd)/custom_prompts:/app/prompts \
  betashow/gemini-deep-research

# Mount logs directory
docker run -p 5357:5357 \
  -e GEMINI_API_KEY=your_key \
  -v $(pwd)/logs:/app/logs \
  betashow/gemini-deep-research
```

## 🔧 Building Your Own Image

### From Source
```bash
# Clone the repository
git clone https://github.com/preangelleo/gemini_deep_research.git
cd gemini_deep_research

# Build the image
docker build -t my-research-agent .

# Run your custom image
docker run -p 5357:5357 \
  -e GEMINI_API_KEY=your_key \
  my-research-agent
```

### Custom Dockerfile
```dockerfile
FROM betashow/gemini-deep-research:latest

# Add custom prompts
COPY my_prompts/ /app/prompts/

# Add custom configuration
COPY my_config.env /app/.env

# Override default configuration
ENV MAX_SEARCHES_PER_TASK=15
ENV AI_POLISH_CONTENT=2
```

## 🌐 API Usage

### Health Check
```bash
curl http://localhost:5357/health
```

### Basic Research
```bash
curl -X POST http://localhost:5357/research \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Latest AI developments 2024",
    "max_searches": 5,
    "ai_polish": true
  }'
```

### Generate Markdown Report
```bash
curl -X POST http://localhost:5357/research \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Climate change solutions",
    "max_searches": 8,
    "ai_polish": 2,
    "output_format": "markdown"
  }' > research_report.md
```

## 🔍 Monitoring and Debugging

### Container Logs
```bash
# View real-time logs
docker logs -f research-agent

# View last 100 lines
docker logs --tail 100 research-agent
```

### Container Shell Access
```bash
# Access container shell for debugging
docker exec -it research-agent /bin/bash

# Check processes
docker exec research-agent ps aux

# Check disk usage
docker exec research-agent df -h
```

### Health Monitoring
```bash
# Check container health
docker inspect research-agent --format='{{.State.Health.Status}}'

# Health check endpoint
curl http://localhost:5357/health | jq
```

## 🚀 Production Deployment

### Using Docker Swarm
```bash
# Initialize swarm
docker swarm init

# Deploy service
docker service create \
  --name research-agent \
  --publish 5357:5357 \
  --env GEMINI_API_KEY=your_key \
  --replicas 2 \
  betashow/gemini-deep-research
```

### Using Kubernetes
Create `k8s-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: research-agent
spec:
  replicas: 2
  selector:
    matchLabels:
      app: research-agent
  template:
    metadata:
      labels:
        app: research-agent
    spec:
      containers:
      - name: research-agent
        image: betashow/gemini-deep-research:latest
        ports:
        - containerPort: 5357
        env:
        - name: GEMINI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: gemini-api-key
        resources:
          limits:
            memory: "2Gi"
            cpu: "1000m"
          requests:
            memory: "1Gi" 
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: research-agent-service
spec:
  selector:
    app: research-agent
  ports:
  - protocol: TCP
    port: 80
    targetPort: 5357
  type: LoadBalancer
```

Deploy:
```bash
kubectl apply -f k8s-deployment.yaml
```

### Reverse Proxy (Nginx)
```nginx
upstream research_agent {
    server localhost:5357;
}

server {
    listen 80;
    server_name research.yourdomain.com;
    
    location / {
        proxy_pass http://research_agent;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## 🔒 Security Considerations

### API Key Security
```bash
# Use Docker secrets (Docker Swarm)
echo "your_api_key" | docker secret create gemini_api_key -

# Use environment file (not recommended for production)
echo "GEMINI_API_KEY=your_key" > .env
docker run --env-file .env betashow/gemini-deep-research
```

### Network Security
```bash
# Run on specific network
docker network create research-network
docker run --network research-network betashow/gemini-deep-research

# Restrict to localhost only
docker run -p 127.0.0.1:5357:5357 betashow/gemini-deep-research
```

### Resource Limits
```bash
# Limit memory and CPU
docker run --memory=2g --cpus=1.5 \
  -p 5357:5357 \
  -e GEMINI_API_KEY=your_key \
  betashow/gemini-deep-research
```

## 🐛 Troubleshooting

### Common Issues:

**Container won't start:**
```bash
# Check logs
docker logs research-agent

# Check resource usage
docker stats research-agent
```

**API returns 401 errors:**
- Verify `GEMINI_API_KEY` is set correctly
- Check API key validity at https://ai.google.dev/

**Port already in use:**
```bash
# Use different port
docker run -p 8357:5357 betashow/gemini-deep-research

# Find what's using port 5357
netstat -tlnp | grep 5357
```

**Container running but API not responding:**
```bash
# Check if port is exposed correctly
docker port research-agent

# Test from inside container
docker exec research-agent curl localhost:5357/health
```

**Performance issues:**
- Increase container memory: `--memory=4g`
- Reduce concurrent operations via environment variables
- Check host system resources

### Getting Help:
- Check container logs: `docker logs research-agent`
- Use health endpoint: `curl localhost:5357/health`
- Visit [GitHub repository](https://github.com/preangelleo/gemini_deep_research)

## 🔗 Related Documentation

- [Flask API Guide](FLASK_API.md) - Local development
- [Main README](../README.md) - Project overview
- [Docker Hub](https://hub.docker.com/r/betashow/gemini-deep-research) - Official image