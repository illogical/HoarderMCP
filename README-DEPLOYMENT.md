# HoarderMCP Deployment Guide

This guide explains how to deploy HoarderMCP using Docker and Docker Compose.

## Prerequisites

- Docker 20.10.0 or higher
- Docker Compose 2.0.0 or higher
- At least 4GB of free RAM (8GB recommended)
- At least 10GB of free disk space

## Quick Start

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone https://github.com/yourusername/hoardermcp.git
   cd hoardermcp
   ```

2. **Create a `.env` file** (if you don't have one):
   ```bash
   cp .env.example .env
   ```
   Edit the `.env` file to set your configuration values.

3. **Start the application**:
   ```bash
   # Make the start script executable (Linux/macOS)
   chmod +x start.sh
   
   # Start the application
   ./start.sh
   ```
   On Windows, you can run the start script directly from Git Bash or use Docker Desktop.

4. **Access the application**:
   - API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Milvus Dashboard: http://localhost:9091 (username: minioadmin, password: minioadmin)

## Stopping the Application

To stop the application and clean up resources:

```bash
./stop.sh
```

## Configuration

### Environment Variables

Create a `.env` file in the project root with the following variables:

```env
# Application
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Server
MCP_SERVER_HOST=0.0.0.0
MCP_SERVER_PORT=8000

# Milvus
MILVUS_HOST=milvus-standalone
MILVUS_PORT=19530

# Other configurations as needed
```

### Volumes

The following Docker volumes are created by default:

- `milvus_data`: Stores Milvus data
- `etcd_data`: Stores etcd data
- `minio_data`: Stores MinIO object storage data

## Production Deployment

For production deployments, consider the following:

1. **Use a reverse proxy** like Nginx or Traefik in front of the API
2. **Enable HTTPS** using Let's Encrypt or your preferred SSL certificate provider
3. **Monitor** the application using the health check endpoint at `/health`
4. **Back up** the data volumes regularly
5. **Set resource limits** in `docker-compose.override.yml`

## Troubleshooting

### View Logs

```bash
# View all logs
docker-compose logs -f

# View logs for a specific service
docker-compose logs -f api
```

### Common Issues

1. **Port conflicts**: Ensure ports 8000, 19530, and 9091 are available
2. **Insufficient resources**: The application requires significant RAM and CPU
3. **Permission issues**: Ensure Docker has proper permissions to create volumes

## Upgrading

To upgrade to a new version:

```bash
git pull
docker-compose pull
docker-compose up -d --build
```
