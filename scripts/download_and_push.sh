#!/bin/bash

# ==============================================================================
# Download Docker Image from GitHub Actions and Push to Docker Hub
# ==============================================================================

set -e

# Configuration
REPO="preangelleo/gemini_deep_research"
ARTIFACT_NAME="docker-image"
IMAGE_NAME="betashow/gemini-deep-research:latest"
DOCKER_USERNAME="${DOCKER_USERNAME:-betashow}"
DOCKER_TOKEN="${DOCKER_TOKEN:-}"

# Check for required environment variables
if [ -z "$DOCKER_TOKEN" ]; then
    echo -e "\033[0;31m❌ DOCKER_TOKEN environment variable is required\033[0m"
    echo "Please set it with your Docker Hub personal access token:"
    echo "  export DOCKER_TOKEN=your_docker_token_here"
    echo "  ./scripts/download_and_push.sh"
    exit 1
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🐳 Docker Image Download and Push Script${NC}"
echo "=================================================="

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo -e "${RED}❌ GitHub CLI (gh) is not installed${NC}"
    echo "Please install it: https://cli.github.com/"
    exit 1
fi

# Check if docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed${NC}"
    exit 1
fi

# Get the latest workflow run
echo -e "${YELLOW}📥 Finding latest successful workflow run...${NC}"
RUN_ID=$(gh run list --repo $REPO --workflow "Build Docker Image" --status success --limit 1 --json databaseId --jq '.[0].databaseId')

if [ -z "$RUN_ID" ]; then
    echo -e "${RED}❌ No successful workflow runs found${NC}"
    echo "Please trigger a workflow run or wait for one to complete successfully."
    exit 1
fi

echo -e "${GREEN}✅ Found workflow run ID: $RUN_ID${NC}"

# Download the artifact
echo -e "${YELLOW}📦 Downloading Docker image artifact...${NC}"
gh run download $RUN_ID --repo $REPO --name $ARTIFACT_NAME --dir ./temp_download

if [ ! -f "./temp_download/gemini-deep-research.tar" ]; then
    echo -e "${RED}❌ Docker image file not found in artifact${NC}"
    rm -rf ./temp_download
    exit 1
fi

# Load the Docker image
echo -e "${YELLOW}🔄 Loading Docker image...${NC}"
docker load -i ./temp_download/gemini-deep-research.tar

# Clean up downloaded files
rm -rf ./temp_download

# Test the image
echo -e "${YELLOW}🧪 Testing Docker image...${NC}"
if docker run --rm $IMAGE_NAME python -c "import sys; print('Python version:', sys.version); import flask, crawl4ai, google.generativeai; print('✅ All dependencies OK')"; then
    echo -e "${GREEN}✅ Image test passed${NC}"
else
    echo -e "${RED}❌ Image test failed${NC}"
    exit 1
fi

# Login to Docker Hub
echo -e "${YELLOW}🔐 Logging in to Docker Hub...${NC}"
echo "$DOCKER_TOKEN" | docker login --username "$DOCKER_USERNAME" --password-stdin

# Push the image
echo -e "${YELLOW}🚀 Pushing to Docker Hub...${NC}"
docker push $IMAGE_NAME

# Cleanup
docker logout

echo -e "${GREEN}🎉 Successfully pushed $IMAGE_NAME to Docker Hub!${NC}"
echo ""
echo "You can now run:"
echo "  docker pull $IMAGE_NAME"
echo "  docker run -p 5357:5357 -e GEMINI_API_KEY=your_key $IMAGE_NAME"