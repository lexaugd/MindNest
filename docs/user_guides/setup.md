# MindNest Setup Guide

## System Requirements

- Python 3.9+ (3.10 recommended)
- 8GB RAM minimum (16GB+ recommended)
- 10GB free disk space
- CUDA-compatible GPU (optional but recommended)

## Installation

### Standard Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/MindNest.git
   cd MindNest
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   # Standard installation
   pip install -r requirements.txt
   
   # For development setup (includes testing tools)
   pip install -r requirements/development.txt
   
   # For lightweight mode (minimal dependencies)
   pip install -r requirements/lightweight.txt
   ```

4. Copy and configure environment variables:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` to match your configuration.

### Development Installation

For development work, install development dependencies:

```bash
pip install -r requirements/development.txt
```

### GPU Support (Optional)

For GPU acceleration:

1. Ensure you have compatible NVIDIA drivers installed
2. Install CUDA Toolkit 11.7+
3. Set the environment variable:
   ```bash
   export USE_GPU=true  # On Windows: set USE_GPU=true
   ```

## Model Setup

MindNest requires language model files to function:

1. Create a `models` directory if it doesn't exist:
   ```bash
   mkdir -p models
   ```

2. Download the model files:
   
   **Option 1**: Download automatically (requires 7-15GB of disk space):
   ```bash
   ./scripts/download_models.sh  # On Windows: scripts\download_models.bat
   ```
   
   **Option 2**: Download manually and place in the `models` directory:
   - For large model: `Wizard-Vicuna-13B-Uncensored.Q4_K_M.gguf`
   - For small model: `llama-2-7b.Q4_K_M.gguf`

## Document Preparation

1. Place your documentation files in the `docs` directory:
   ```bash
   mkdir -p docs
   cp /path/to/your/documentation/* docs/
   ```

2. Run the document processor to initialize the vector database:
   ```bash
   python scripts/doc_chunker.py
   ```

## Configuration

Edit `.env` file to set configuration options:

```
# Model settings
USE_SMALL_MODEL=false  # Set to true for smaller, faster model
MODEL_NAME=Wizard-Vicuna-13B-Uncensored.Q4_K_M.gguf
SMALL_MODEL_NAME=llama-2-7b.Q4_K_M.gguf

# API settings
HOST=0.0.0.0
PORT=8000
DEBUG=false

# Document settings
DOCS_DIR=docs
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

## Running MindNest

### Direct Execution

```bash
python -m mindnest.app
```

Or use the convenience wrapper:

```bash
./run.sh  # On Windows: run.bat
```

### Development Mode

For development with automatic reloading:

```bash
uvicorn mindnest.app:app --reload --host 0.0.0.0 --port 8000
```

## Verifying the Installation

1. Open a web browser and navigate to:
   ```
   http://localhost:8000
   ```

2. You should see the MindNest interface.

3. Test the API:
   ```bash
   curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"text":"What is MindNest?"}'
   ```

## Troubleshooting

### Common Issues

- **Model Loading Errors**: Ensure model files are correctly named and placed in the `models` directory
- **Out of Memory Errors**: Reduce `CHUNK_SIZE` in `.env` or switch to small model with `USE_SMALL_MODEL=true`
- **Slow Response Time**: Enable GPU acceleration or switch to small model
- **Import Errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`

### Logs

Check log files for detailed error information:

```bash
cat logs/mindnest.log
```

## Next Steps

- Continue to the [Usage Guide](usage.md) for information on using MindNest
- See [Docker Guide](docker_guide.md) for containerized deployment 