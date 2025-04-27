"""
Test script to verify context optimization in the running application.

This script directly tests the optimize_context_for_model function with
both small and large models using documents from the docs directory.
"""

from langchain.schema import Document
from main import optimize_context_for_model

def load_sample_docs():
    """Load sample documents from the docs directory."""
    import os
    from langchain_community.document_loaders import TextLoader, DirectoryLoader, UnstructuredMarkdownLoader
    
    print("Loading sample documents...")
    docs_dir = "docs"
    
    # Create large test documents to force optimization
    from langchain.schema import Document
    
    # Create a large document to test optimization
    with open("docs/MindNest_Documentation.md", "r") as f:
        base_content = f.read()
    
    # Create large test documents by duplicating content
    large_docs = [
        Document(
            page_content=base_content * 3,  # Duplicate the content to make it larger
            metadata={"source": "large_doc1.md"}
        ),
        Document(
            page_content=base_content * 2,
            metadata={"source": "large_doc2.md"}
        ),
        Document(
            page_content=base_content,
            metadata={"source": "large_doc3.md"}
        )
    ]
    
    print(f"Created {len(large_docs)} large test documents")
    for i, doc in enumerate(large_docs):
        print(f"Large test doc {i+1}: {len(doc.page_content)} chars")
    
    return large_docs

def test_small_model():
    """Test context optimization for small models with real queries."""
    print("\n=== Testing Small Model Context Optimization ===")
    
    # Load sample documents
    docs = load_sample_docs()
    
    # Print document sizes
    for i, doc in enumerate(docs):
        content_length = len(doc.page_content)
        source = doc.metadata.get("source", "unknown")
        print(f"Document {i+1}: {content_length} chars, source: {source}")
    
    # Define small model capabilities with a very small context window to force truncation
    model_capabilities = {
        "model_size": "small",
        "context_window": 1024  # Smaller context window to force truncation
    }
    
    # Test queries that should trigger different optimization paths
    queries = [
        "What is MindNest's architecture?",
        "Explain the purpose of context window optimization",
        "How does the token-aware document compression work?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        
        # Optimize context
        optimized_docs = optimize_context_for_model(docs, query, model_capabilities)
        
        # Print results
        print(f"Original documents: {len(docs)}, Optimized documents: {len(optimized_docs)}")
        for i, doc in enumerate(optimized_docs):
            content_length = len(doc.page_content)
            source = doc.metadata.get("source", "unknown")
            print(f"Optimized doc {i+1}: {content_length} chars, source: {source}")

def test_large_model():
    """Test context optimization for large models with real queries."""
    print("\n=== Testing Large Model Context Optimization ===")
    
    # Load sample documents
    docs = load_sample_docs()
    
    # Define large model capabilities with a smaller context window to force truncation
    model_capabilities = {
        "model_size": "large",
        "context_window": 2048  # Smaller window to force truncation
    }
    
    # Test queries that should trigger different optimization paths
    queries = [
        "What is MindNest's architecture?",
        "Explain the purpose of context window optimization",
        "How does the token-aware document compression work?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        
        # Optimize context
        optimized_docs = optimize_context_for_model(docs, query, model_capabilities)
        
        # Print results
        print(f"Original documents: {len(docs)}, Optimized documents: {len(optimized_docs)}")
        for i, doc in enumerate(optimized_docs):
            content_length = len(doc.page_content)
            source = doc.metadata.get("source", "unknown")
            print(f"Optimized doc {i+1}: {content_length} chars, source: {source}")

if __name__ == "__main__":
    print("Testing context optimization in the running application...")
    test_small_model()
    test_large_model()
    print("\nAll tests completed!") 