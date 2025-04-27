# MINDNEST SPECIFICATION

## OVERVIEW
MindNest is an intelligent documentation system that uses AI to understand and answer questions about codebases. It leverages vector embeddings and language models to process documents and provide accurate responses to queries. The system supports both powerful LLMs for high-quality responses and lightweight models for faster performance.

## RECOMMENDATIONS
- Ensure model-specific optimizations for all response generation
- Maintain separate prompt templates for different model sizes
- Implement quality control for smaller model responses
- Balance context window usage based on model capabilities

## DIRECTORY TREE
- main.py - Main application with FastAPI endpoints
- utils/ - Utility modules for various functionality
  - llm_manager.py - Language model initialization and management
  - query_optimization.py - Query processing and categorization
  - responses.py - Response templates for conversation
  - query_classification/ - AI-based query classification
- static/ - Web interface files
- models/ - Language model files storage
- docs/ - Documentation storage

## TECH STACK
- Python 3.10+
- FastAPI - Web framework
- LangChain - Document processing and LLM integration
- ChromaDB - Vector database for embeddings
- Llama.cpp - Efficient LLM inference
- HuggingFace - Embedding models

## KEY FEATURES
- Multi-format document processing and indexing
- Intelligent query classification
- RAG-based question answering
- Model-aware response generation
- Optimized performance for different hardware capabilities
- Flexible conversation modes for non-document queries 