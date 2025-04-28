# Docker Guide for MindNest

## Overview

This guide covers how to deploy MindNest using Docker. Containerization provides consistent environments, easier deployment, and isolation from the host system.

## Prerequisites

- Docker installed on your system
- Docker Compose installed (for multi-container setup)
- 16GB RAM recommended for Docker
- 20GB free disk space

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/MindNest.git
   cd MindNest
   ```

2. Start MindNest using Docker Compose:
   ```bash
   docker-compose -f docker/docker-compose.yml up
   ```

3. Access the application at:
   ```
   http://localhost:8000
   ```

## Docker Configuration Files

MindNest includes the following Docker configuration files:

- `Dockerfile`: Defines the MindNest container image (in the root directory)
- `docker/docker-compose.yml`: Defines the multi-container setup (in the docker directory)
- `.dockerignore`: Lists files excluded from Docker builds

## Single Container Setup

To build and run MindNest as a single container:

```bash
# Build the Docker image
docker build -t mindnest -f docker/Dockerfile .

# Run the container
docker run -p 8000:8000 \
  -v $(pwd)/docs:/app/docs \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/chroma_db:/app/chroma_db \
  mindnest
```

## Multi-Container Setup

The included Docker Compose file sets up:

1. MindNest application container
2. ChromaDB container for vector storage

To use it:

```bash
# Start all services
docker-compose -f docker/docker-compose.yml up

# Or run in detached mode
docker-compose -f docker/docker-compose.yml up -d
```

To stop services:

```bash
docker-compose -f docker/docker-compose.yml down
```

## Environment Variables

Configure MindNest in Docker by setting environment variables:

```bash
docker run -p 8000:8000 \
  -e USE_SMALL_MODEL=true \
  -e DEBUG=true \
  -v $(pwd)/docs:/app/docs \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/chroma_db:/app/chroma_db \
  mindnest
```

Or in docker-compose.yml:

```yaml
services:
  mindnest:
    image: mindnest
    environment:
      - USE_SMALL_MODEL=true
      - DEBUG=true
```

## Volume Management

MindNest uses Docker volumes for persistent data:

- `/app/docs`: Documentation to be processed
- `/app/models`: Language model files
- `/app/chroma_db`: Vector database storage
- `/app/logs`: Application logs

Example:

```bash
docker run -p 8000:8000 \
  -v /path/to/your/docs:/app/docs \
  -v /path/to/your/models:/app/models \
  -v mindnest_chroma_db:/app/chroma_db \
  -v mindnest_logs:/app/logs \
  mindnest
```

## GPU Support

To use GPU acceleration with Docker:

1. Install the NVIDIA Container Toolkit
2. Use the `--gpus` flag:

```bash
docker run --gpus all -p 8000:8000 \
  -v $(pwd)/docs:/app/docs \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/chroma_db:/app/chroma_db \
  mindnest
```

Or in docker-compose.yml:

```yaml
services:
  mindnest:
    image: mindnest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Production Deployment

For production deployment:

1. Use specific image tags, not `latest`
2. Set `DEBUG=false`
3. Use proper network configuration
4. Consider adding a reverse proxy for SSL termination
5. Implement proper monitoring and health checks

Example production docker-compose.yml:

```yaml
version: '3.8'

services:
  mindnest:
    image: mindnest:1.0.0
    restart: always
    environment:
      - USE_SMALL_MODEL=false
      - DEBUG=false
    volumes:
      - ./docs:/app/docs
      - ./models:/app/models
      - mindnest_chroma_db:/app/chroma_db
      - mindnest_logs:/app/logs
    ports:
      - "8000:8000"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  chroma:
    image: chromadb/chroma:latest
    volumes:
      - chroma_data:/chroma/chroma

volumes:
  mindnest_chroma_db:
  mindnest_logs:
  chroma_data:
```

## Common Docker Issues

### Troubleshooting

- **Container exits immediately**: Check logs with `docker logs <container_id>`
- **Out of memory**: Increase Docker memory limit or use smaller models
- **Permission issues**: Check volume mount permissions
- **Network issues**: Ensure ports are correctly mapped

### Viewing Logs

```bash
# View container logs
docker logs -f mindnest

# For Docker Compose
docker-compose -f docker/docker-compose.yml logs -f
```

## Customizing the Docker Image

To customize the MindNest Docker image:

1. Modify the Dockerfile:
   ```dockerfile
   # Example: Add custom dependencies
   RUN pip install my-extra-package
   ```

2. Build a custom image:
   ```bash
   docker build -t mindnest-custom -f docker/Dockerfile .
   ```

## Resource Management

Limit container resources:

```bash
docker run -p 8000:8000 \
  --memory=4g \
  --cpus=2 \
  mindnest
```

Or in docker-compose.yml:

```yaml
services:
  mindnest:
    image: mindnest
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
``` 