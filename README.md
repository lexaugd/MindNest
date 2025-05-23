# MindNest - AI Documentation System

An intelligent documentation system that uses AI to understand and answer questions about your codebase.

## What's New in v1.3

- **Enhanced Conversational Capabilities**: Intelligent detection of conversational queries with specialized processing
- **Improved Query Classification**: Advanced classification using semantic embeddings for better intent understanding
- **Hybrid Document Retrieval**: Combined dense and sparse retrieval methods with reranking for more relevant results
- **Conversation History Management**: Context tracking across multiple interactions for more coherent dialogues
- **Clean Code Architecture**: Complete refactoring of legacy code, simplified directory structure, and improved modularity
- **Robust Error Handling**: Graceful fallbacks ensure the system always responds appropriately to all query types

See the [full release notes](release_notes/v1.3-conversational-query-improvements.md) for details.

## Configuration Options

MindNest provides several settings to customize behavior:

- **Model Selection**: Switch between standard (higher quality) and lightweight (faster) models
- **Context Size**: Control how much context is sent to the language model (200-2000 tokens)
- **Conversation Style**: Choose between Professional, Passive Aggressive, or Humorous styles for fallback responses
- **Query Classification**: Select different methods to classify query intent:
  - Embeddings: Fastest method using vector similarity
  - Neural: Most accurate using a fine-tuned model
  - Hybrid: Balanced approach combining multiple methods
  - Regex: Legacy fallback using pattern matching

Changes to models require restart, while other settings take effect immediately.

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
- **Conversational Query Processing**: Intelligent detection and handling of conversational vs. document-based queries
- **Enhanced Hybrid Retrieval**: Combined dense and sparse retrieval with cross-encoder reranking for better results
- **Conversation History Management**: Maintains context across multiple interactions for coherent dialogues

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
   python -m mindnest.app  # Standard mode
   USE_SMALL_MODEL=true python -m mindnest.app  # Lightweight mode
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
│   ├── features/               # Feature documentation
│   └── user_guides/            # User-focused documentation
├── mindnest/                   # Main package directory
│   ├── __init__.py             # Package initialization
│   ├── api/                    # API endpoints
│   │   ├── __init__.py         # API package initialization
│   │   └── endpoints.py        # API endpoints implementation
│   ├── core/                   # Core application logic
│   │   ├── __init__.py         # Core package initialization
│   │   ├── config.py           # Configuration management
│   │   ├── llm_manager.py      # LLM initialization and management
│   │   └── document_processor.py # Document processing logic
│   ├── utils/                  # Utility modules
│   │   ├── __init__.py         # Utils package initialization
│   │   ├── document_compression.py    # Document compression utilities
│   │   ├── document_processor.py      # Document processing utilities
│   │   ├── document_tracker.py        # Document change tracking
│   │   ├── incremental_vectorstore.py # Vector database management
│   │   ├── enhanced_vectorstore.py    # Enhanced vector storage capabilities
│   │   ├── hybrid_retriever.py        # Hybrid retrieval implementation
│   │   ├── bm25_retriever.py          # BM25 sparse retrieval
│   │   ├── cross_encoder_reranker.py  # Result reranking with cross-encoders
│   │   ├── conversational_response.py # Conversational response generation
│   │   ├── query_cache.py             # Query caching
│   │   ├── query_classification/      # Query classification
│   │   │   ├── __init__.py            # Classification package initialization
│   │   │   ├── classifier.py          # Query classifier implementation
│   │   │   ├── model_loader.py        # ML model loader
│   │   │   ├── feedback.py            # Feedback collection
│   │   │   └── feedback_learning.py   # Continuous learning from feedback
│   │   ├── query_optimization.py      # Query optimization
│   │   ├── query_preprocessing.py     # Query preprocessing
│   │   ├── response_formatter.py      # Response formatting
│   │   ├── response_evaluation.py     # Response evaluation metrics
│   │   ├── responses.py               # Response handling
│   │   ├── logger.py                  # Logging utilities
│   │   ├── models.py                  # Shared data models
│   │   └── token_counter.py           # Token counting utilities
│   └── app.py                  # Primary application entry point
├── scripts/                    # Utility scripts
│   ├── cleanup_docs.py         # Document cleanup utilities
│   ├── doc_chunker.py          # Semantic document chunker
│   ├── query_docs.py           # Document query utilities
│   ├── run_tests.py            # Test runner script
│   ├── test_query_classification.py  # Query classifier testing
│   └── move_files.py           # File organization utility
├── tests/                      # Test suite
│   ├── __init__.py             # Test package initialization
│   ├── test_context_optimizer.py    # Context optimization tests
│   ├── test_document_processor.py   # Document processing tests
│   ├── test_config.py               # Configuration tests
│   ├── test_query_classification_script.py  # Classification script tests
│   ├── test_zero_shot_classifier.py        # Classifier model tests
│   └── test_response_evaluation.py         # Response evaluation tests
├── docker/                     # Docker configuration
│   ├── Dockerfile              # Container definition
│   └── docker-compose.yml      # Multi-container setup
├── static/                     # Static web files
├── models/                     # AI model files
├── release_notes/              # Release documentation
├── logs/                       # Application logs
├── cache/                      # Cache storage
├── chroma_db/                  # Vector database files
├── data/                       # Data files
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
├── .dockerignore               # Docker ignore file
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
![image](https://github.com/user-attachments/assets/f8ff2f95-4bbf-4689-a202-2e6f2c95cff9)


