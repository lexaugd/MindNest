# MindNest - AI Documentation System

An intelligent documentation system that uses AI to understand and answer questions about your codebase.

## Features

- **Multi-Format Support**: Understands Java, Groovy, Python, JavaScript, TypeScript, and plain text documentation
- **Smart Document Processing**: Automatically processes and indexes your documentation with incremental updates
- **Intelligent Q&A**: Answers questions about your code accurately using LLM-powered responses
- **Source Attribution**: Always shows which files the information comes from
- **Optimized Performance**: Efficient caching and incremental updates for faster responses
- **Lightweight Mode**: Option to run with vector search only for better performance
- **Model Switching**: Choose between larger (high quality) or smaller (faster) language models
- **Multiple Launch Options**: Several ways to start the application including launcher script, direct run, or shell script
- **Automatic Dependency Management**: Detects and installs missing dependencies automatically
- **Context Window Optimization**: Intelligent document truncation based on model capabilities, with different strategies for small and large models
- **Model-Aware Processing**: Different prompt templates and document limits based on model size
- **Response Quality Controls**: Validation and formatting to ensure consistent, high-quality responses
- **Comprehensive Test Suite**: Automated testing for context optimization and document processing
- **Docker Support**: Run in containers with optimized resource configuration

## Docker Installation

The easiest way to run MindNest is using Docker, which eliminates dependency issues.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

### Getting Started

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/MindNest.git
   cd MindNest
   ```

2. Download the required LLM models:
   - Place model files in the `models/` directory:
     - For standard model: `Wizard-Vicuna-13B-Uncensored.Q4_K_M.gguf`
     - For lightweight model: `llama-2-7b.Q4_K_M.gguf`

3. Build and start the Docker container:
   ```bash
   ./run.sh docker:start
   ```

4. Access the application at http://localhost:8000

### Modes and Options

- **Application Mode**:
  - Standard: Full features with LLM capabilities
  - Lightweight: Vector search only

- **Model Options**:
  - Standard: Wizard-Vicuna-13B (larger, more powerful)
  - Lightweight: Llama-2-7B (faster, less memory intensive)

### Stopping the Application

```bash
./run.sh docker:stop
```

## Manual Installation

If you prefer to run without Docker:

1. Install dependencies:
   ```bash
   # Standard installation
   pip install -r requirements.txt
   
   # Development setup (includes testing tools)
   pip install -r requirements/development.txt
   
   # Lightweight mode (minimal dependencies)
   pip install -r requirements/lightweight.txt
   ```

2. Place model files in the `models/` directory

3. Run the application directly:
   ```bash
   # Standard mode
   ./run.sh start
   
   # Lightweight mode
   ./run.sh start:light
   
   # Alternatively, run directly with Python
   python run_server.py  # Standard mode
   USE_SMALL_MODEL=true python run_server.py  # Lightweight mode
   ```

4. Access the application at http://localhost:8000

## Docker Deployment Options

For Docker deployment, there are several options:

1. Using `docker-compose.yml`:
```bash
docker-compose up -d
```

2. Using the shell script helper:
```bash
./run.sh docker:start    # Build and start container
./run.sh docker:stop     # Stop container
./run.sh docker:logs     # View logs
```

3. For detailed instructions and optimized Colima setup for macOS, see [docker-setup.md](docker-setup.md).

## Troubleshooting

If you encounter issues:

1. Check the logs:
   ```bash
   docker-compose logs -f
   ```

2. Ensure model files are correctly placed in the `models/` directory

3. Run the test suite to verify core functionality:
   ```bash
   python -m unittest tests/test_context_optimizer.py
   ```

4. For dependency conflicts in non-Docker setup, try using Python 3.10 with a fresh virtual environment

## Project Structure

```
MindNest/
├── docs/                       # Documentation and code examples
│   ├── examples/               # Code examples by language
│   │   ├── java/               # Java code examples
│   │   ├── groovy/             # Groovy code examples
│   │   ├── python/             # Python code examples
│   │   └── javascript/         # JavaScript examples
│   ├── features/               # Feature documentation
│   │   ├── context_window.md   # Context window optimization documentation
│   │   ├── query_processing.md # Query processing documentation
│   │   └── model_support.md    # Model capabilities documentation
│   └── user_guides/            # User-focused documentation
│       ├── setup.md            # Setup instructions
│       ├── docker_guide.md     # Docker configuration guide
│       └── usage.md            # Usage instructions
├── mindnest/                   # Main package directory
│   ├── __init__.py             # Package initialization
│   ├── api/                    # API endpoints
│   │   ├── __init__.py         # API package initialization
│   │   ├── endpoints.py        # API endpoints implementation
│   │   └── models.py           # API data models
│   ├── core/                   # Core application logic
│   │   ├── __init__.py         # Core package initialization
│   │   ├── config.py           # Configuration management
│   │   ├── llm_manager.py      # LLM initialization and management
│   │   ├── document_processor.py # Document processing logic
│   │   └── response_generator.py # Response generation logic
│   ├── utils/                  # Utility modules
│   │   ├── __init__.py         # Utils package initialization
│   │   ├── document_compression.py # Document compression utilities
│   │   ├── document_tracker.py # Document change tracking
│   │   ├── incremental_vectorstore.py # Vector database management
│   │   ├── query_cache.py      # Query caching
│   │   ├── query_classification/ # Query classification
│   │   │   ├── __init__.py      # Classification package initialization
│   │   │   ├── classifier.py    # Query classifier implementation
│   │   │   ├── model_loader.py  # ML model loader
│   │   │   └── example_queries.py # Training examples
│   │   ├── query_optimization.py # Query optimization
│   │   ├── responses.py        # Response formatting
│   │   └── token_counter.py    # Token counting utilities
│   └── app.py                  # Primary application entry point
├── scripts/                    # Utility scripts
│   ├── cleanup_docs.py         # Document cleanup utilities
│   ├── doc_chunker.py          # Semantic document chunker
│   └── run_tests.py            # Test runner script
├── tests/                      # Test suite
│   ├── __init__.py             # Test package initialization
│   ├── test_context_optimizer.py # Context optimization tests
│   ├── test_document_processor.py # Document processing tests
│   └── test_api_integration.py # API integration tests
├── docker/                     # Docker configuration
│   ├── Dockerfile              # Container definition
│   └── docker-compose.yml      # Multi-container setup
├── static/                     # Static web files
├── models/                     # AI model files
├── pyproject.toml              # Python project configuration
├── setup.py                    # Package setup script
├── requirements.txt            # Python dependencies (redirects to production)
├── requirements/               # Organized requirements files
│   ├── base.txt                # Base dependencies for all environments
│   ├── production.txt          # Production dependencies
│   ├── development.txt         # Development dependencies
│   └── lightweight.txt         # Lightweight mode dependencies
├── README.md                   # Project overview
├── CHANGELOG.md                # Release history and changes
├── LICENSE                     # License information
├── .gitignore                  # Git ignore file
└── run.sh                      # Shell script launcher
```

## Usage

### Installation with pip

MindNest can now be installed as a Python package:

```bash
# Install from local directory
pip install -e .

# Run using the package
python -m mindnest.app
```

### Adding Documentation

1. Place your documentation in the `docs/` directory
   - Supported formats: `.java`, `.groovy`, `.py`, `.js`, `.ts`, `.txt`, `.md`
   - Organize in subdirectories if desired

2. Start the server:
   ```bash
   # Using the package
   python -m mindnest.app
   
   # Shell script (on Unix/Mac systems)
   ./run.sh start
   ```

### Docker Deployment

Docker files have been moved to the `docker/` directory:

```bash
# Build and start using docker-compose
docker-compose -f docker/docker-compose.yml up -d

# Or use the helper script
./run.sh docker:start
```

## Development

### Project Organization

MindNest now follows standard Python package structure:
- Code is organized in the `mindnest/` package
- Configuration in `pyproject.toml` and `setup.py`
- Development tools defined in `requirements-dev.txt`

### Running Tests

Tests are in the `tests/` directory and can be run with pytest:

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest
```

## Documentation

Detailed documentation is available in the `docs/` directory:

- **User Guides**: `docs/user_guides/`
  - [Setup Guide](docs/user_guides/setup.md)
  - [Docker Guide](docs/user_guides/docker_guide.md)
  - [Usage Guide](docs/user_guides/usage.md)

- **Feature Documentation**: `docs/features/`
  - [Context Window Optimization](docs/features/context_window.md)
  - [Query Processing](docs/features/query_processing.md)
  - [Model Support](docs/features/model_support.md)

## License

[MIT License](LICENSE)

## Acknowledgements

- [LangChain](https://github.com/hwchase17/langchain) for document processing and LLM integration
- [ChromaDB](https://github.com/chroma-core/chroma) for vector storage
- [Llama.cpp](https://github.com/ggerganov/llama.cpp) for efficient LLM inference
- [HuggingFace](https://huggingface.co/) for embedding models and transformers

<!-- Screenshots removed until we have proper screenshots -->


