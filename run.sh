#!/bin/bash
# Script to manage MindNest Docker container or run directly

set -e

# Check for models directory and required model files
if [ ! -d "models" ]; then
    echo "Creating models directory..."
    mkdir -p models
fi

MODELS_WARNING=0
if [ ! -f "models/Wizard-Vicuna-13B-Uncensored.Q4_K_M.gguf" ] && [ ! -f "models/llama-2-7b.Q4_K_M.gguf" ]; then
    MODELS_WARNING=1
fi

# Show usage if no arguments provided
function show_usage {
    echo "MindNest Management Script"
    echo ""
    echo "Usage: ./run.sh [command] [options]"
    echo ""
    echo "Commands:"
    echo "  docker:start    - Build and start the MindNest container"
    echo "  docker:stop     - Stop the MindNest container"
    echo "  docker:restart  - Restart the MindNest container"
    echo "  docker:logs     - Show logs from the container"
    echo "  docker:shell    - Open a shell in the container"
    echo "  docker:clean    - Remove the container and built images"
    echo "  start           - Start MindNest directly (not in Docker)"
    echo "  start:light     - Start MindNest in lightweight mode"
    echo ""
    echo "Options for 'start' command:"
    echo "  --lightweight-model - Use the lightweight model"
    echo "  --no-browser        - Don't open browser automatically" 
    echo ""
}

if [ $# -eq 0 ]; then
    show_usage
    exit 0
fi

# Check if Docker is available for docker commands
function check_docker {
    if ! command -v docker &> /dev/null || ! command -v docker-compose &> /dev/null; then
        echo "Error: Docker and/or Docker Compose are not installed."
        echo "Please install them first: https://docs.docker.com/get-docker/"
        exit 1
    fi
}

case "$1" in
    docker:start)
        check_docker
        echo "Starting MindNest container..."
        docker-compose up -d --build
        
        if [ $MODELS_WARNING -eq 1 ]; then
            echo ""
            echo "WARNING: Model files not found in models/ directory."
            echo "You need to download at least one of these models:"
            echo "  - models/Wizard-Vicuna-13B-Uncensored.Q4_K_M.gguf (standard model)"
            echo "  - models/llama-2-7b.Q4_K_M.gguf (lightweight model)"
            echo ""
            echo "MindNest will still start in lightweight mode without models, but full features will be unavailable."
        fi
        
        echo ""
        echo "MindNest is available at: http://localhost:8080"
        ;;
    
    docker:stop)
        check_docker
        echo "Stopping MindNest container..."
        docker-compose down
        ;;
    
    docker:restart)
        check_docker
        echo "Restarting MindNest container..."
        docker-compose restart
        ;;
    
    docker:logs)
        check_docker
        echo "Showing logs from MindNest container..."
        docker-compose logs -f
        ;;
    
    docker:shell)
        check_docker
        echo "Opening shell in MindNest container..."
        docker-compose exec mindnest bash
        ;;
    
    docker:clean)
        check_docker
        echo "Removing MindNest container and images..."
        docker-compose down --rmi all
        ;;
    
    start)
        shift
        echo "Starting MindNest directly..."
        python run_direct.py "$@"
        ;;
    
    start:light)
        shift
        echo "Starting MindNest in lightweight mode..."
        python run_direct.py --lightweight "$@"
        ;;
        
    *)
        echo "Unknown command: $1"
        show_usage
        exit 1
        ;;
esac 