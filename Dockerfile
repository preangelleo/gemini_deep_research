# ==============================================================================
# Gemini Deep Research Agent - Docker Image
# ==============================================================================
# 
# This Dockerfile creates a production-ready container for the Gemini Deep 
# Research Agent with Flask API server.
#
# Features:
# - Python 3.11+ for optimal compatibility
# - All dependencies pre-installed including Playwright browsers
# - Optimized for production use
# - Non-root user for security
# - Health check endpoint
#
# Usage:
#   docker build -t gemini-deep-research .
#   docker run -p 5357:5357 -e GEMINI_API_KEY=your_key gemini-deep-research
#
# ==============================================================================

FROM python:3.11-slim

# Set metadata
LABEL maintainer="Deep Research Team"
LABEL version="1.0.0"
LABEL description="Gemini Deep Research Agent with AI-powered research capabilities"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Set API configuration
ENV API_HOST=0.0.0.0
ENV API_PORT=5357
ENV DEBUG_MODE=false

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser -m appuser

# Install system dependencies required for Playwright and Python packages
RUN apt-get update && apt-get install -y \
    # System tools
    curl \
    wget \
    gnupg \
    ca-certificates \
    # Build dependencies
    gcc \
    g++ \
    make \
    # Playwright browser dependencies
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxss1 \
    libgconf-2-4 \
    libxrandr2 \
    libasound2 \
    libpangocairo-1.0-0 \
    libatk1.0-0 \
    libcairo-gobject2 \
    libgtk-3-0 \
    libgdk-pixbuf2.0-0 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxfixes3 \
    libxi6 \
    libxtst6 \
    # Additional dependencies
    fonts-liberation \
    xdg-utils \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Install Playwright and browsers
RUN playwright install chromium && \
    playwright install-deps chromium

# Copy application code
COPY src/ ./src/
COPY prompts/ ./prompts/
COPY examples/ ./examples/
COPY app.py ./
COPY .env.example ./

# Create necessary directories
RUN mkdir -p /app/logs /app/cache /app/temp_results

# Change ownership to non-root user
RUN chown -R appuser:appuser /app && \
    chown -R appuser:appuser /home/appuser

# Set HOME environment variable
ENV HOME=/home/appuser

# Switch to non-root user
USER appuser

# Expose the API port
EXPOSE 5357

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5357/health || exit 1

# Set entrypoint
CMD ["python", "app.py"]

# ==============================================================================
# Build Instructions:
# 
# 1. Build the image:
#    docker build -t betashow/gemini-deep-research:latest .
#
# 2. Run with API key:
#    docker run -p 5357:5357 -e GEMINI_API_KEY=your_key betashow/gemini-deep-research
#
# 3. Run with custom configuration:
#    docker run -p 5357:5357 \
#      -e GEMINI_API_KEY=your_key \
#      -e MAX_SEARCHES_PER_TASK=10 \
#      -e AI_POLISH_CONTENT=2 \
#      betashow/gemini-deep-research
#
# ==============================================================================