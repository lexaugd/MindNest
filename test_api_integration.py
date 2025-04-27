"""
Test script to verify context optimization integration with the API.

This script directly calls functions from main.py to test the context 
optimization with the same workflow as the /ask endpoint.
"""

import sys
from langchain.schema import Document
from main import optimize_context_for_model
from utils.token_counter import count_tokens
from utils.document_compression import compress_documents

def simulate_api_request(query, docs, model_size="small"):
    """
    Simulate an API request to test context optimization.
    
    Args:
        query: The query string
        docs: List of documents
        model_size: Model size (small or large)
    
    Returns:
        Optimized list of documents
    """
    print(f"\nSimulating API request for query: '{query}'")
    
    # Set model capabilities based on size
    model_capabilities = {
        "model_size": model_size,
        "context_window": 2048 if model_size == "small" else 4096,
        "document_limit": 3 if model_size == "small" else 5,
        "concise_limit": 2 if model_size == "small" else 3
    }
    
    # Print document stats before optimization
    total_chars = sum(len(doc.page_content) for doc in docs)
    total_tokens = sum(count_tokens(doc.page_content) for doc in docs)
    print(f"Before optimization: {len(docs)} docs, {total_chars} chars, ~{total_tokens} tokens")
    
    # Apply context optimization
    optimized_docs = optimize_context_for_model(docs, query, model_capabilities)
    
    # Print document stats after optimization
    opt_chars = sum(len(doc.page_content) for doc in optimized_docs)
    opt_tokens = sum(count_tokens(doc.page_content) for doc in optimized_docs)
    print(f"After optimization: {len(optimized_docs)} docs, {opt_chars} chars, ~{opt_tokens} tokens")
    
    # Verify optimization actually occurred
    if len(optimized_docs) < len(docs) or opt_chars < total_chars:
        print("✅ Context optimization worked: Reduced context size")
    else:
        print("⚠️ Context optimization didn't reduce size")
    
    return optimized_docs

def create_test_documents():
    """Create test documents with varying sizes."""
    # Create some test documents of different sizes
    docs = [
        Document(
            page_content="A" * 10000,  # 10K chars
            metadata={"source": "doc1.txt"}
        ),
        Document(
            page_content="B" * 8000,  # 8K chars
            metadata={"source": "doc2.txt"}
        ),
        Document(
            page_content="C" * 6000,  # 6K chars
            metadata={"source": "doc3.txt"}
        ),
        Document(
            page_content="D" * 4000,  # 4K chars
            metadata={"source": "doc4.txt"}
        )
    ]
    return docs

def test_context_window_integration():
    """Test context window optimization integration."""
    print("Testing context window optimization integration with API...")
    
    # Create test documents
    docs = create_test_documents()
    print(f"Created {len(docs)} test documents")
    for i, doc in enumerate(docs):
        print(f"Doc {i+1}: {len(doc.page_content)} chars, source: {doc.metadata.get('source')}")
    
    # Test queries
    queries = [
        "What is the architecture of the system?",
        "How does context window optimization work?",
        "Tell me about token-aware document compression"
    ]
    
    # Test with small model
    print("\n=== Testing with Small Model ===")
    for query in queries:
        optimize_small = simulate_api_request(query, docs, model_size="small")
    
    # Test with large model
    print("\n=== Testing with Large Model ===")
    for query in queries:
        optimize_large = simulate_api_request(query, docs, model_size="large")
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    test_context_window_integration() 