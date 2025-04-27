# MindNest Documentation

## Overview

MindNest is an intelligent documentation system that uses AI to understand and answer questions about your codebase. It leverages large language models (LLMs) to provide accurate, context-aware responses to questions about your documentation and code.

## System Architecture

MindNest uses a layered architecture:

1. **Web Interface Layer**:
   - FastAPI backend for handling API requests
   - Simple HTML/CSS/JS frontend for user interaction

2. **Processing Layer**:
   - Query classification and optimization
   - Document retrieval and context optimization
   - Response formatting and quality control

3. **Intelligence Layer**:
   - LLM integration through LangChain
   - Vector embeddings using HuggingFace models
   - ChromaDB vector store for efficient retrieval

4. **Storage Layer**:
   - Document tracking for incremental updates
   - Query caching for improved performance
   - Vector storage for semantic search

## Data Flow

The system follows this workflow for handling queries:

1. **Document Processing**:
   - Documents are loaded from the `docs/` directory
   - Text is split into chunks with semantic awareness
   - Each chunk is embedded and stored in the vector database
   - Document metadata tracks changes for incremental updates

2. **Query Processing**:
   - User submits a query through the web interface
   - Query is classified (document query, concise query, search, conversation)
   - Relevant documents are retrieved from vector store
   - Context is optimized based on model capabilities

3. **Response Generation**:
   - For document queries, the LLM generates answers using retrieved context
   - For search queries, document snippets are returned directly
   - For conversation, template-based responses are generated
   - Quality control ensures responses are formatted properly

4. **Output Delivery**:
   - Response is sent back to the web interface
   - Sources are cited for transparency
   - Results are cached for future similar queries

## Key Components

### `main.py`
The heart of the application, containing the FastAPI server, endpoint definitions, and the core document processing and query handling logic. It includes:

- API endpoint implementations
- Context window optimization
- Response quality control
- Document retrieval logic

### `utils/` Directory
Contains modular components:

- `llm_manager.py`: Manages LLM initialization and capabilities
- `incremental_vectorstore.py`: Handles vector database operations
- `document_tracker.py`: Tracks document changes for incremental updates
- `query_cache.py`: Caches query results for performance
- `config.py`: Centralizes configuration settings
- `query_classification/`: Contains AI-based query classification

### Runner Scripts
Multiple ways to start the application:

- `run.sh`: Shell script with Docker support and command-line options
- `run_direct.py`: Direct Python runner with command-line arguments
- `run_server.py`: Lightweight server for vector search only

## Context Window Optimization

A key feature is intelligent context window optimization:

1. **How It Works**:
   - Adjusts document context based on model capabilities
   - Small models get more aggressive truncation (1536 char limit)
   - Large models get balanced truncation (3072 char limit)
   - Preserves metadata regardless of truncation

2. **Model-Aware Processing**:
   - Different document limits for small vs. large models
   - Adjusted prompt templates for each model size
   - Retrieval strategies optimized for model context size

## Installation & Setup

### Prerequisites
- Python 3.10+ 
- LLM model files:
  - Standard model: `Wizard-Vicuna-13B-Uncensored.Q4_K_M.gguf`
  - Lightweight model: `llama-2-7b.Q4_K_M.gguf`

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
   pip install -r requirements.txt
   ```

4. Place model files in the `models/` directory

5. Create a `.env` file (use `env.example` as template)

6. Run the application:
   ```bash
   ./run.sh start
   ```

### Docker Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/MindNest.git
   cd MindNest
   ```

2. Build and start the container:
   ```bash
   ./run.sh docker:start
   ```

3. Access at http://localhost:8080

## Configuration Options

MindNest is highly configurable through environment variables or `.env` file:

### Server Settings
- `HOST`: Server hostname (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)

### Vector Store Settings
- `VECTORSTORE_DIR`: Directory for vector storage (default: chroma_db)
- `EMBEDDING_MODEL`: Embedding model name (default: all-MiniLM-L6-v2)

### Document Processing Settings
- `CHUNK_SIZE`: Document chunk size (default: 1000)
- `CHUNK_OVERLAP`: Chunk overlap (default: 200)

### LLM Settings
- `MODEL_NAME`: Large model filename (default: Wizard-Vicuna-13B-Uncensored.Q4_K_M.gguf)
- `SMALL_MODEL_NAME`: Small model filename (default: llama-2-7b.Q4_K_M.gguf)
- `USE_SMALL_MODEL`: Whether to use small model (default: false)
- `TEMPERATURE`: LLM temperature (default: 0.3)
- `MAX_TOKENS`: Maximum output tokens (default: 2000)
- `CONTEXT_WINDOW`: Model context window size (default: 2048)

### Cache Settings
- `CACHE_ENABLED`: Enable query caching (default: true)
- `MEMORY_CACHE_SIZE`: Memory cache size (default: 1024)
- `DISK_CACHE_ENABLED`: Enable disk caching (default: true)

### Query Settings
- `QUERY_CLASSIFIER_MODE`: Classification method (default: embeddings)
- `CONVERSATION_MODE`: Conversation style (default: professional)

## Usage Guide

### Adding Documentation

1. Place your documentation in the `docs/` directory
   - Supported formats: `.java`, `.groovy`, `.py`, `.js`, `.ts`, `.txt`, `.md`
   - Organize in subdirectories if desired

2. The system automatically processes documents on startup
   - Incremental updates track which files have changed
   - Only changed files are reprocessed

### Running the Application

You have several options to start MindNest:

1. **Using run.sh script**:
   ```bash
   # Standard mode
   ./run.sh start
   
   # Lightweight mode (vector search only)
   ./run.sh start:light
   
   # Docker mode
   ./run.sh docker:start
   ```

2. **Using run_direct.py**:
   ```bash
   # Standard mode
   python run_direct.py
   
   # With smaller model
   python run_direct.py --lightweight-model
   
   # Specify port
   python run_direct.py --port 8080
   ```

3. **Using Docker Compose**:
   ```bash
   docker-compose up -d
   ```

### Querying the System

The system handles different types of queries:

1. **Document Queries**: Ask questions about your code
   - Example: "How does the login system work?"

2. **Concise Queries**: Request brief answers
   - Example: "Briefly describe the authentication process"

3. **Document Search**: Directly search for documents
   - Example: "Find files related to authentication"

4. **Conversation**: Chat-like interactions
   - These don't use document context and provide templated responses

### Monitoring and Maintenance

1. Check system status:
   ```bash
   curl http://localhost:8000/health
   ```

2. View document statistics:
   ```bash
   curl http://localhost:8000/documents
   ```

3. Get detailed system stats:
   ```bash
   curl http://localhost:8000/stats
   ```

4. Clear caches when needed:
   ```bash
   curl -X POST http://localhost:8000/clear-cache
   ```

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

You can also run individual test modules:

```bash
python -m unittest tests/test_context_optimizer.py
```

## Troubleshooting

Common issues and solutions:

1. **Model loading errors**:
   - Verify model files are in the correct location
   - Check memory requirements (large models need 8GB+ RAM)
   - Try using the small model with `--lightweight-model`

2. **Document processing issues**:
   - Check file formats (use supported formats only)
   - Ensure proper permissions for reading files
   - Try clearing the document cache by deleting `utils/doc_tracking.json`

3. **Server connection problems**:
   - Verify port is not already in use
   - Check firewall settings
   - Use `http://localhost:8000` (standard) or `http://localhost:8080` (Docker)

4. **Slow response times**:
   - Use the smaller model for faster responses
   - Enable caching (it's on by default)
   - Reduce document context with smaller `CONTEXT_WINDOW`

## Advanced Configuration

For optimal performance in production:

1. **Model Optimization**:
   - Use quantized models for better performance
   - Adjust context window size based on typical document length
   - Set temperature lower (0.1-0.3) for more consistent responses

2. **Docker Resource Allocation**:
   - For large models: 16GB RAM, 4+ CPU cores
   - For small models: 8GB RAM, 2+ CPU cores
   - Colima optimized settings included in `docker-setup.md`

3. **Caching Tuning**:
   - Increase `MEMORY_CACHE_SIZE` for frequently accessed documents
   - Set `DISK_CACHE_ENABLED=true` for persistence between restarts
   - Clear cache periodically when updating documents frequently

## Development Workflow

If you're developing or extending MindNest:

1. **Code Organization**:
   - Core logic in `main.py`
   - Modular components in `utils/`
   - Tests in `tests/`
   - Configuration in `config.py` and `.env`

2. **Adding Features**:
   - Create new utility modules in `utils/`
   - Update API endpoints in `main.py`
   - Add configuration options to `config.py`
   - Create tests in `tests/` directory

3. **Testing Changes**:
   - Run unit tests with `python run_tests.py`
   - Test with real documents
   - Validate changes with and without model loaded

4. **Documentation**:
   - Update README.md with user-facing changes
   - Update in-code documentation
   - Consider adding test cases demonstrating new features 