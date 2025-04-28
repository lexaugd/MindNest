# MINDNEST SPECIFICATION

## OVERVIEW
MindNest is an intelligent documentation system that uses AI to understand and answer questions about codebases. It leverages vector embeddings and language models to process documents and provide accurate responses to queries. The system supports both powerful LLMs for high-quality responses and lightweight models for faster performance, with enhanced conversational capabilities for natural interaction.

## RECOMMENDATIONS
- Ensure model-specific optimizations for all response generation
- Maintain separate prompt templates for different model sizes
- Implement quality control for smaller model responses
- Balance context window usage based on model capabilities
- Use hybrid retrieval combining dense and sparse methods for improved document retrieval
- Implement dynamic prompt templates based on query classification
- Consider factoring complex queries into sub-queries for improved retrieval
- Maintain conversation context for multi-turn interactions to improve response relevance
- Consider implementing user preference tracking for personalized responses
- Monitor conversation history management to optimize memory usage

## DIRECTORY TREE
- main.py - Main application with FastAPI endpoints
- utils/ - Utility modules for various functionality
  - llm_manager.py - Language model initialization and management
  - query_preprocessing.py - Neural query rewriting and expansion
  - conversational_response.py - Conversational response generation with context management
  - query_classification/ - AI-based query classification
    - classifiers/ - Different classifier implementations
      - semantic_classifier.py - Neural classifier using embeddings
  - enhanced_vectorstore.py - Advanced vector store with hybrid retrieval and reranking
  - bm25_retriever.py - BM25 keyword-based retrieval
  - hybrid_retriever.py - Combined dense-sparse retrieval system
  - cross_encoder_reranker.py - Result reranking with cross-encoders
  - document_tracker.py - Tracks document changes for incremental updates
  - document_processor.py - Processes documents for indexing
  - conversation_manager.py - Manages conversation context and history
- core/ - Core application logic
  - llm_chain.py - Main reasoning chain
  - chain_factory.py - Chain construction and configuration
  - response_formatter.py - Formats responses based on context and query type
- static/ - Web interface files
- models/ - Language model files storage
- docs/ - Documentation storage
- monitoring/ - System monitoring and metrics

## TECH STACK
- Python 3.10+
- FastAPI - Web framework
- LangChain - Document processing and LLM integration
- ChromaDB - Vector database for embeddings
- Llama.cpp - Efficient LLM inference
- HuggingFace - Embedding models and rerankers
- SentenceTransformers - Text embeddings and cross-encoders
- Prometheus/Grafana - Metrics and monitoring (optional)

## KEY FEATURES
- Multi-format document processing and indexing
- Neural query classification and preprocessing
  - Neural classification using sentence embeddings
  - Query rewriting and expansion
  - Semantic routing based on query type
- Enhanced document retrieval system
  - Hybrid retrieval combining dense and sparse methods
  - BM25 keyword-based search for precision
  - Minimum similarity thresholds for relevance filtering
  - Cross-encoder reranking for improved ranking precision
  - Reciprocal rank fusion for combining multiple retrievers
- Adaptive response generation
  - Dynamic prompt templates based on query classification
  - Response quality scoring and confidence metrics
  - Proper citation formatting and attribution
  - Context-aware conversational responses
  - Conversation history tracking for multi-turn coherence
  - Automatic detection of conversation mode vs. document queries
  - Fallback mechanisms for handling out-of-domain queries
  - Conversational memory with configurable retention policies
  - Contextual entity tracking across conversation turns
- System improvements
  - Incremental document indexing with efficient updates
  - Cached retrieval results for improved performance
  - Configurable retrieval parameters based on query complexity
  - Comprehensive monitoring and metrics dashboard
  - Conversation state persistence for session continuity 