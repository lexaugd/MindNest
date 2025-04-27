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
<<<<<<< Updated upstream
- **Multiple Launch Options**: Several ways to start the application including launcher script, direct run, or shell script
- **Automatic Dependency Management**: Detects and installs missing dependencies automatically
- **Context Window Optimization**: Intelligent document truncation based on model capabilities
- **Model-Aware Processing**: Different prompt templates and document limits based on model size
- **Response Quality Controls**: Validation and formatting to ensure consistent, high-quality responses
- **Comprehensive Test Suite**: Automated testing for core functionality
- **Docker Support**: Run in containers with optimized resource configuration
=======
- **Direct Runner**: Simple command-line interface to run the application with various options
- **Docker Support**: Run the application in Docker for easier deployment
>>>>>>> Stashed changes

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
   pip install -r requirements.txt
   ```

<<<<<<< Updated upstream
5. Run the application (multiple options available):
   - **Python Launcher**: `python mindnest_launcher.py` (see `release_notes/launcher_guide.md` for options)
   - **Shell Script**: `./run.sh start` (Unix/Mac) or use Docker commands with `./run.sh docker:start`
   - **Direct Run**: `python run_direct.py` with optional arguments like `--lightweight-model`
   - **Manual Start**: `python main.py` (full mode) or `python run_server.py` (lightweight mode)
=======
2. Place model files in the `models/` directory
>>>>>>> Stashed changes

3. Run the application directly:
   ```bash
   # Standard mode
   ./run.sh start
   
   # Lightweight mode
   ./run.sh start:light
   
   # With additional options
   ./run.sh start --lightweight-model --no-browser
   ```

<<<<<<< Updated upstream
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
=======
4. Access the application at http://localhost:8000

## Troubleshooting

If you encounter issues:

1. Check the logs:
   ```bash
   docker-compose logs -f
   ```

2. Ensure model files are correctly placed in the `models/` directory

3. For dependency conflicts in non-Docker setup, try using Python 3.10 with a fresh virtual environment
>>>>>>> Stashed changes

## Project Structure

```
MindNest/
<<<<<<< Updated upstream
├── docs/                       # Documentation and code examples
│   ├── java/                   # Java code examples
│   ├── groovy/                 # Groovy code examples
│   ├── py/                     # Python code examples
│   └── txt/                    # Text documentation
├── models/                     # AI model files
├── static/                     # Static web files
├── utils/                      # Utility modules
│   ├── config.py               # Configuration management
│   ├── document_processor.py   # Document processing
│   ├── document_tracker.py     # Document change tracking
│   ├── incremental_vectorstore.py # Vector database management
│   ├── llm_manager.py          # LLM initialization and management
│   ├── logger.py               # Logging system
│   ├── models.py               # Data models
│   ├── query_cache.py          # Query caching
│   ├── query_classification/   # Query classification
│   │   ├── classifier.py       # Query classifier implementation
│   │   ├── model_loader.py     # ML model loader
│   │   └── example_queries.py  # Training examples
│   ├── query_optimization.py   # Query classification and optimization
│   └── responses.py            # Response formatting and quality control
├── tests/                      # Test suite
│   ├── test_config.py          # Configuration tests
│   ├── test_context_optimizer.py # Context optimization tests
│   └── test_document_processor.py # Document processing tests
├── release_notes/              # Version release notes and documentation
│   ├── environment_update.md   # Config migration guide
│   ├── launcher_guide.md       # Launcher usage documentation
│   ├── model_switching.md      # Model switching guide
│   ├── v1.0.md                 # Version 1.0 notes
│   ├── v1.1-ai-accuracy-improvements.md # AI improvements details
│   └── v1.1-release.md         # Version 1.1 release summary
├── doc_chunks/                 # Processed document chunks directory
├── Dockerfile                  # Container definition
├── docker-compose.yml          # Multi-container setup
├── docker-setup.md             # Docker deployment guide
├── .dockerignore               # Docker build exclusions
├── mindnest_launcher.py        # Python launcher with options
├── run.sh                      # Shell script with Docker support
├── run_direct.py               # Direct runner script
├── main.py                     # Main application with LLM
├── run_server.py               # Lightweight server (vector search only)
├── run_tests.py                # Test runner
├── requirements.txt            # Python dependencies
├── requirements-lightweight.txt # Minimal dependencies
├── env.example                 # Environment variables template
├── cleanup_docs.py             # Document cleanup utilities
├── doc_chunker.py              # Semantic document chunker
├── basic_app_with_docs.py      # Legacy application (deprecated)
├── test_docs.py                # Old test script (deprecated)
└── README.md                   # This file
=======
├── docs/                 # Documentation and code examples
│   ├── java/            # Java code examples
│   ├── groovy/          # Groovy code examples
│   ├── py/              # Python code examples
│   └── txt/             # Text documentation
├── models/              # AI model files
├── static/              # Static web files
├── utils/               # Utility modules
│   ├── config.py        # Configuration management
│   ├── document_processor.py  # Document processing
│   ├── document_tracker.py    # Document change tracking
│   ├── incremental_vectorstore.py  # Vector database management
│   ├── llm_manager.py   # LLM initialization and management
│   ├── logger.py        # Logging system
│   ├── models.py        # Data models
│   ├── query_cache.py   # Query caching
│   └── query_optimization.py  # Query classification and optimization
├── release_notes/       # Version release notes and documentation
├── run_direct.py        # Direct runner for the application
├── main.py              # Main application with LLM
├── run_server.py        # Lightweight server (vector search only)
├── requirements.txt     # Python dependencies
├── env.example          # Example environment variables
├── Dockerfile           # Docker configuration
└── README.md            # This file
>>>>>>> Stashed changes
```

## Usage

### Adding Documentation

1. Place your documentation in the `docs/` directory
   - Supported formats: `.java`, `.groovy`, `.py`, `.js`, `.ts`, `.txt`, `.md`
   - Organize in subdirectories if desired

<<<<<<< Updated upstream
2. Start the server using one of the launcher options:
   ```bash
   # Standard launcher
   python mindnest_launcher.py
   
   # Shell script (on Unix/Mac systems)
   ./run.sh start
   
   # Direct run
   python run_direct.py
   
   # Docker container
   ./run.sh docker:start
=======
2. Start the server:
   ```bash
   # Full mode with direct runner
   python run_direct.py
   
   # Lightweight mode
   python run_direct.py --lightweight
   
   # Specify port and host
   python run_direct.py --port 8080 --host 127.0.0.1
   
   # Use lightweight model
   python run_direct.py --lightweight-model
   
   # Don't open browser automatically
   python run_direct.py --no-browser
>>>>>>> Stashed changes
   ```

3. Access the web interface at `http://localhost:8000` (or port 8080 if using Docker)

### Document Processing Tools

For optimizing your documentation:

1. **Document Cleanup**:
   ```bash
   python cleanup_docs.py [--directory docs]
   ```
   This removes duplicates and organizes content better.

2. **Semantic Chunking**:
   ```bash
   python doc_chunker.py [--directory docs]
   ```
   This processes documents into semantically coherent chunks.

### Querying Documentation

The system handles different types of queries:

1. **Document Queries**: Ask questions about your code and documentation
   - Example: "How does the login system work?"

2. **Concise Queries**: Request brief answers
   - Example: "Briefly describe the authentication process"

3. **Document Search**: Directly search for documents
   - Example: "Find files related to authentication"

## Configuration

MindNest is highly configurable via environment variables or the `.env` file:

- **Server Settings**: Host, port
- **Vector Store Settings**: Storage directory, embedding model
- **Document Processing**: Chunk size, overlap
- **LLM Settings**: Model selection, temperature, context window
- **Cache Settings**: Memory/disk cache configuration
- **Query Settings**: Classifier mode, conversation style
- **Logging Settings**: Log level, log file location

See `env.example` for all available options.

## AI Response Quality

MindNest v1.1 includes several improvements for AI response accuracy:

1. **Context Window Optimization**: 
   - Small models: Prioritizes beginnings of documents
   - Large models: Keeps balanced content from both beginning and end
   - Prevents token limit overflows for more reliable responses

2. **Model-Specific Processing**:
   - Different prompt templates optimized for small or large models
   - Adjusted document retrieval limits based on model capabilities
   - Customized response formatting based on model strengths

3. **Quality Control**:
   - Validation checks for small model responses
   - Fallback mechanisms for potentially low-quality outputs
   - Confidence assessment for hybrid response generation
   - Consistent formatting for better readability

For a complete list of v1.1 improvements, see `release_notes/v1.1-release.md`.

## Testing

Run the test suite to verify functionality:

```bash
python run_tests.py
```

Tests cover core functionality including:
- Context window optimization for different model sizes
- Configuration management
- Document processing and chunking
- Query classification and optimization

## Technical Details

- Built with FastAPI for high-performance API endpoints
- Uses LangChain for document processing and LLM integration
- Supports multiple LLM models (tested with Llama family models)
- Vector search with ChromaDB and HuggingFace embeddings
- Optimized for incremental updates to avoid rebuilding the vector database
- Multi-tiered caching system for improved response times
- Model-aware processing pipeline for optimal results

## Limitations

- Best suited for codebases under 10,000 files
- Response time depends on hardware and model size
- Large language models require significant memory

## License

[MIT License](LICENSE)

## Acknowledgements

- [LangChain](https://github.com/hwchase17/langchain) for document processing and LLM integration
- [ChromaDB](https://github.com/chroma-core/chroma) for vector storage
- [Llama.cpp](https://github.com/ggerganov/llama.cpp) for efficient LLM inference
- [HuggingFace](https://huggingface.co/) for embedding models and transformers

<!-- Screenshots removed until we have proper screenshots -->


