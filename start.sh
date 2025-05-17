#!/bin/bash

# Exit on error
set -e

# Create necessary directories
mkdir -p ./data

# Build and start containers
echo "Starting HoarderMCP with Docker Compose..."
docker-compose up -d --build

echo ""
echo "HoarderMCP is starting up..."
echo "- API: http://localhost:8000"
echo "- API Docs: http://localhost:8000/docs"
echo "- Milvus Dashboard: http://localhost:9091"
echo ""
echo "To view logs: docker-compose logs -f"
echo "To stop: docker-compose down"
