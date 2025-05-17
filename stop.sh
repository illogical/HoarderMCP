#!/bin/bash

echo "Stopping HoarderMCP containers..."
docker-compose down

echo ""
echo "All containers have been stopped and removed."
echo "To remove volumes as well, run: docker-compose down -v"
