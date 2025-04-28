# Query Processing System

## Overview

MindNest implements a sophisticated query processing system that intelligently categorizes, optimizes, and handles different types of user queries. This system ensures that each query is processed in the most efficient way based on its nature and the available context.

## Query Classification

The system uses multiple strategies to classify queries:

- **Document Search**: Queries that explicitly request document retrieval
- **Conversation**: General conversational queries not requiring document lookup
- **Concise Query**: Queries that need concise, direct answers from documents
- **Document Query**: Queries that need detailed information extracted from documents

### Classification Methods

MindNest supports multiple classification methods that can be configured:

- **Embeddings-based**: Uses semantic similarity with example queries
- **Neural**: Employs a specialized neural model for classification
- **Hybrid**: Combines multiple classification approaches
- **Regex**: Simple pattern matching for quick classification

## Query Optimization

Once classified, queries are optimized for processing:

- **Keyword Extraction**: Identifies important terms in the query
- **Entity Recognition**: Recognizes named entities for better retrieval
- **Query Expansion**: Expands queries with related terms when beneficial
- **Query Reformulation**: Restructures queries for better semantic matching

## Context Adaptation

The system adapts queries based on the conversation context:

- **Pronoun Resolution**: Resolves pronouns using conversation history
- **Context Carry-over**: Maintains relevant context across turns
- **Topic Tracking**: Tracks the current topic for better continuity

## Response Strategy Selection

Different query types trigger different response strategies:

- **Document Retrieval**: For document search queries
- **RAG Processing**: For document and concise queries
- **Direct Response**: For conversation queries

## Caching System

An efficient caching system improves performance:

- **Query Cache**: Stores results of previous queries
- **Memory & Disk Tiers**: Two-level caching for optimal performance
- **Automatic Invalidation**: Cache entries are invalidated when documents change

## Configuration Options

The query processing system can be configured through the API:

```python
@router.post("/config/model", status_code=200)
async def configure_model(config: ModelConfig):
    # Configuration options include:
    # - query_classifier_mode: "embeddings", "neural", "hybrid", or "regex"
    # - max_context_tokens: Control context window size
    # - use_small_model: Toggle between different model sizes
    ...
```

## Usage Example

The query processing system automatically handles incoming queries:

```python
# Process the query and determine its type
query_type, query = process_query(text, model_capabilities)

# Different handling based on query type
if query_type == "DOCUMENT_SEARCH":
    # Handle document search
elif query_type == "CONVERSATION":
    # Handle conversation
elif query_type == "CONCISE_QUERY":
    # Handle concise query with RAG
else:  # DOCUMENT_QUERY
    # Handle document query with RAG
```

## Benefits

- **Improved Accuracy**: Better responses by matching query intent
- **Efficiency**: Resource optimization by using the right strategy
- **Flexibility**: Multiple classification methods to suit different needs
- **Speed**: Caching and optimization for faster responses 