# MindNest Usage Guide

## Overview

This guide covers how to use MindNest for document processing, querying, and configuration.

## Core Features

- **Document Query**: Ask questions about your documentation
- **Document Search**: Search for specific documents
- **Conversation**: Have general conversations with the AI
- **System Configuration**: Adjust model settings and behavior

## Web Interface

### Accessing the Interface

1. Ensure MindNest is running (see [Setup Guide](setup.md))
2. Open a web browser and navigate to:
   ```
   http://localhost:8000
   ```

### Ask Questions

1. Type your question in the input box at the bottom of the screen
2. Press Enter or click the send button
3. View the AI response and any source references

### View Document List

1. Click on the "Documents" tab in the navigation menu
2. Browse the list of loaded documents
3. Click on any document to view its contents

### System Statistics

1. Click on the "Stats" tab in the navigation menu
2. View information about:
   - Document count and types
   - Cache performance
   - Model configuration
   - System status

## API Usage

MindNest provides a REST API for programmatic access:

### Ask Questions

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"text":"How do I install MindNest?"}'
```

Response format:
```json
{
  "text": "To install MindNest, you need to clone the repository, create a virtual environment, and install dependencies with pip install -r requirements.txt...",
  "sources": ["setup.md", "README.md"]
}
```

### List Documents

```bash
curl -X GET "http://localhost:8000/documents"
```

Response format:
```json
{
  "total_documents": 15,
  "sources": [
    "setup.md",
    "api_reference.md",
    "installation.md"
  ]
}
```

### Get Statistics

```bash
curl -X GET "http://localhost:8000/stats"
```

### Configure Model

```bash
curl -X POST "http://localhost:8000/config/model" \
  -H "Content-Type: application/json" \
  -d '{
    "use_small_model": true,
    "max_context_tokens": 600,
    "conversation_mode": "professional",
    "query_classifier_mode": "embeddings"
  }'
```

### Check Health

```bash
curl -X GET "http://localhost:8000/health"
```

### Clear Cache

```bash
curl -X POST "http://localhost:8000/clear-cache"
```

## Query Types

MindNest supports different types of queries:

### Document Queries

Questions that need information from your documentation:

- "How do I configure MindNest?"
- "What are the system requirements?"
- "Explain how to use the Docker setup"

### Document Search Queries

Explicit requests to find documents:

- "Find documents about Docker"
- "Show me files related to GPU setup"
- "List all documentation on configuration"

### Conversational Queries

General conversation with the AI:

- "Hello, how does this work?"
- "Can you help me with something?"
- "What can you do?"

### Concise Queries

Requests for short, direct answers:

- "What version of Python is required? Be concise."
- "Briefly explain the installation process"
- "In one sentence, what is MindNest?"

## Adding New Documents

To add new documentation:

1. Place new files in the `docs` directory
2. The files can be in various formats:
   - Markdown (.md)
   - Text (.txt)
   - PDF (.pdf)
   - HTML (.html)
   - Code files (.py, .js, etc.)

3. Run the document processor to update the vector database:
   ```bash
   python scripts/doc_chunker.py
   ```

4. Alternatively, use the API to trigger reprocessing:
   ```bash
   curl -X POST "http://localhost:8000/reprocess-docs"
   ```

## Configuration Options

### Model Size

You can switch between models:

- Small model: Faster but less capable
- Large model: More capable but requires more resources

```bash
curl -X POST "http://localhost:8000/config/model" \
  -H "Content-Type: application/json" \
  -d '{"use_small_model": true}'
```

### Context Window Size

Adjust how much context is provided to the model:

```bash
curl -X POST "http://localhost:8000/config/model" \
  -H "Content-Type: application/json" \
  -d '{"max_context_tokens": 800}'
```

### Conversation Mode

Change the AI's conversation style:

```bash
curl -X POST "http://localhost:8000/config/model" \
  -H "Content-Type: application/json" \
  -d '{"conversation_mode": "professional"}'
```

Options:
- `professional`: Formal, business-like responses
- `humorous`: More casual with light humor
- `passive_aggressive`: Sarcastic, mildly irritated tone

### Query Classification Mode

Change how queries are classified:

```bash
curl -X POST "http://localhost:8000/config/model" \
  -H "Content-Type: application/json" \
  -d '{"query_classifier_mode": "hybrid"}'
```

Options:
- `embeddings`: Uses vector similarity (default)
- `neural`: Uses a neural classifier
- `hybrid`: Combines multiple approaches
- `regex`: Uses pattern matching (fastest, least accurate)

## Best Practices

1. **Specific Questions**: The more specific your questions, the better the answers
2. **Reference Documents**: Mention specific documents if you know them
3. **Clear Formatting**: Format documents clearly with proper headings
4. **Regular Updates**: Reprocess documents when they change
5. **Cache Management**: Clear the cache periodically for fresh results

## Keyboard Shortcuts

- **Enter**: Send message
- **Shift+Enter**: Add newline in input
- **Esc**: Clear input
- **Up Arrow**: Navigate to previous query
- **Down Arrow**: Navigate to next query

## Advanced Features

### Command Queries

Special commands using the '/' prefix:

- `/search [term]`: Explicitly search for documents
- `/clear`: Clear conversation history
- `/help`: Show help information
- `/config`: Show current configuration 