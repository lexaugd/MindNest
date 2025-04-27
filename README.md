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
- **Convenient Launcher**: Launch the application with different configurations using the included launcher script
- **Automatic Dependency Management**: Detects and installs missing dependencies automatically

## Setup

### Standard Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Unix/Mac
# or
venv\Scripts\activate     # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
# Or use the launcher's auto-install feature:
python mindnest_launcher.py --install-deps
```

3. Download the LLM model:
   - Place your LLM model file (e.g., `llama-2-7b.Q4_K_M.gguf`) in the `models/` directory
   - You can use different model sizes based on your hardware capabilities
   - MindNest supports switching between models via the UI (see `release_notes/model_switching.md`)

4. Configure the application:
   - Copy `env.example` to `.env` and adjust settings as needed
   - Environment variables override default configuration
   - **Note:** If upgrading from a previous version, see `release_notes/environment_update.md` for important configuration changes

5. Run the application:
   - Using the launcher: `./mindnest_launcher.py` (see `release_notes/launcher_guide.md` for options)
   - Using the launcher with auto-dependency installation: `./mindnest_launcher.py --auto-install`
   - Manually: `python main.py` (full mode) or `python run_server.py` (lightweight mode)

### Docker Setup

For Docker deployment, see detailed instructions in [docker-setup.md](docker-setup.md).

## Project Structure

```
MindNest/
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
├── mindnest_launcher.py # Application launcher script
├── main.py              # Main application with LLM
├── run_server.py        # Lightweight server (vector search only)
├── requirements.txt     # Python dependencies
├── env.example          # Example environment variables
├── Dockerfile           # Docker configuration
└── README.md            # This file
```

## Usage

### Adding Documentation

1. Place your documentation in the `docs/` directory
   - Supported formats: `.java`, `.groovy`, `.py`, `.js`, `.ts`, `.txt`, `.md`
   - Organize in subdirectories if desired

2. Start the server:
   - Full mode: `python main.py`
   - Lightweight mode: `python run_server.py`

3. Access the web interface at `http://localhost:8000`

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

## Technical Details

- Built with FastAPI for high-performance API endpoints
- Uses LangChain for document processing and LLM integration
- Supports multiple LLM models (tested with Llama family models)
- Vector search with ChromaDB and HuggingFace embeddings
- Optimized for incremental updates to avoid rebuilding the vector database
- Multi-tiered caching system for improved response times

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

<!-- Screenshots removed until we have proper screenshots -->

