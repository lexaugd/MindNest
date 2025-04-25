# MindNest Docker Setup Guide

This document explains how to set up and run the MindNest application using Docker with Colima as an alternative to Docker Desktop.

## Prerequisites

- macOS system
- Colima installed (`brew install colima`)
- Docker CLI installed (`brew install docker`)

## Setup Steps

1. **Start Colima:**
   ```bash
   colima start
   ```

2. **Configure Docker to use Colima:**
   ```bash
   export DOCKER_HOST=unix:///Users/$(whoami)/.colima/default/docker.sock
   ```

3. **Edit .dockerignore:**
   We modified the `.dockerignore` file to include necessary directories that were previously excluded:
   ```
   # Git
   .git
   .gitignore
   
   # Python
   __pycache__/
   *.py[cod]
   *$py.class
   *.so
   .Python
   venv/
   ENV/
   env/
   .env
   
   # IDE
   .idea/
   .vscode/
   *.swp
   *.swo
   
   # Project specific
   chroma_db/
   *.log
   ```

4. **Build Docker image:**
   ```bash
   export DOCKER_HOST=unix:///Users/$(whoami)/.colima/default/docker.sock
   docker build -t mindnest .
   ```

5. **Run the container:**
   ```bash
   export DOCKER_HOST=unix:///Users/$(whoami)/.colima/default/docker.sock
   docker run -d -p 8000:8000 --name mindnest mindnest
   ```

## Optimized Colima Configuration for AI Models

For running large language models and AI applications, we use an optimized Colima configuration with higher resource allocation:

1. **Start Colima with optimized settings:**
   ```bash
   colima start --cpu 8 --memory 24 --disk 200 --vm-type=vz --vz-rosetta --network-address
   ```

2. **Configuration details:**
   - **CPU:** 8 cores (out of 12 total cores)
   - **Memory:** 24GB (out of 32GB total)
   - **Disk:** 200GB
   - **VM Type:** vz (Virtualization.Framework) for better performance
   - **Rosetta:** Enabled for improved performance on ARM architecture
   - **Network Address:** Automatically assigned for easier access

3. **Checking current configuration:**
   ```bash
   colima list
   ```

4. **Reverting to default settings if needed:**
   ```bash
   colima stop
   colima start
   ```

> **Note:** This configuration is optimized for a MacBook Pro with 12 cores (8 performance, 4 efficiency) and 32GB RAM. Adjust according to your system specifications.

## Stopping the Container

To properly stop the MindNest container, follow these steps:

1. **Check if the container is running:**
   ```bash
   export DOCKER_HOST=unix:///Users/$(whoami)/.colima/default/docker.sock
   docker ps
   ```
   
   You should see output like this if the container is running:
   ```
   CONTAINER ID   IMAGE      COMMAND                  CREATED         STATUS         PORTS                    NAMES
   4cf3ddcd0787   mindnest   "uvicorn main:app --…"   5 minutes ago   Up 5 minutes   0.0.0.0:8000->8000/tcp   mindnest
   ```

2. **Stop the container:**
   ```bash
   export DOCKER_HOST=unix:///Users/$(whoami)/.colima/default/docker.sock
   docker stop mindnest
   ```
   
   This command will return the container name (`mindnest`) when successful.

3. **Verify the container is stopped:**
   ```bash
   export DOCKER_HOST=unix:///Users/$(whoami)/.colima/default/docker.sock
   docker ps
   ```
   
   You should see no running containers.

4. **To see all containers, including stopped ones:**
   ```bash
   export DOCKER_HOST=unix:///Users/$(whoami)/.colima/default/docker.sock
   docker ps -a
   ```
   
   You should see the stopped MindNest container:
   ```
   CONTAINER ID   IMAGE      COMMAND                  CREATED         STATUS                      PORTS     NAMES
   4cf3ddcd0787   mindnest   "uvicorn main:app --…"   5 minutes ago   Exited (0) 14 seconds ago             mindnest
   ```

5. **If you want to completely remove the container:**
   ```bash
   export DOCKER_HOST=unix:///Users/$(whoami)/.colima/default/docker.sock
   docker rm mindnest
   ```

## Useful Docker Commands with Colima

Always set the Docker host first:
```bash
export DOCKER_HOST=unix:///Users/$(whoami)/.colima/default/docker.sock
```

Then you can use these commands:

- **List running containers:**
  ```bash
  docker ps
  ```

- **View container logs:**
  ```bash
  docker logs mindnest
  ```

- **Stop the container:**
  ```bash
  docker stop mindnest
  ```

- **Start the container:**
  ```bash
  docker start mindnest
  ```

- **Restart the container:**
  ```bash
  docker restart mindnest
  ```

- **Remove the container (must be stopped first):**
  ```bash
  docker rm mindnest
  ```

- **Remove the image (all containers using it must be removed first):**
  ```bash
  docker rmi mindnest
  ```

## Accessing the Application

Once the container is running, access the application in your web browser:
```
http://localhost:8000
```

## Why Colima Instead of Docker Desktop?

Colima provides a lightweight alternative to Docker Desktop, especially useful when:
- Docker Desktop licensing is an issue
- You need a more lightweight Docker solution
- You want to avoid Docker Desktop's resource usage
- You want a command-line focused Docker environment

Colima runs the Docker daemon in a lightweight VM, providing the Docker API without the overhead of Docker Desktop. 