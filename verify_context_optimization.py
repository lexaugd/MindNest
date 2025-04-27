"""
Simple verification script for context window optimization.

This script directly tests our context window optimization implementation 
with real documents from the docs folder.
"""

import os
import glob
from langchain.schema import Document
from main import optimize_context_for_model
from utils.token_counter import count_tokens

def load_actual_docs(max_docs=5):
    """Load actual documents from the docs folder."""
    print("Loading actual documents from docs folder...")
    
    # Get markdown files from docs directory
    md_files = glob.glob("docs/**/*.md", recursive=True)[:max_docs]
    
    if not md_files:
        print("No markdown files found in docs folder")
        return []
    
    docs = []
    for file_path in md_files:
        try:
            with open(file_path, "r") as f:
                content = f.read()
                docs.append(Document(
                    page_content=content,
                    metadata={"source": file_path}
                ))
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    print(f"Loaded {len(docs)} documents from docs folder")
    for i, doc in enumerate(docs):
        print(f"Doc {i+1}: {len(doc.page_content)} chars, source: {doc.metadata.get('source')}")
    
    return docs

def verify_small_model_optimization():
    """Verify context optimization for small models with actual docs."""
    print("\n=== Verifying Small Model Context Optimization ===")
    
    # Load actual documents
    docs = load_actual_docs()
    if not docs:
        return
    
    # Define small model capabilities
    small_model = {
        "model_size": "small",
        "context_window": 2048
    }
    
    # Print document stats before optimization
    total_chars = sum(len(doc.page_content) for doc in docs)
    total_tokens = sum(count_tokens(doc.page_content) for doc in docs)
    print(f"Before optimization: {len(docs)} docs, {total_chars} chars, ~{total_tokens} tokens")
    
    # Verify with a relevant query
    query = "How does context window optimization work in MindNest?"
    print(f"\nOptimizing for query: '{query}'")
    
    # Optimize context
    optimized_docs = optimize_context_for_model(docs, query, small_model)
    
    # Print document stats after optimization
    opt_chars = sum(len(doc.page_content) for doc in optimized_docs)
    opt_tokens = sum(count_tokens(doc.page_content) for doc in optimized_docs)
    print(f"After optimization: {len(optimized_docs)} docs, {opt_chars} chars, ~{opt_tokens} tokens")
    
    # Check if optimization worked
    if len(optimized_docs) < len(docs) or opt_chars < total_chars:
        print("✅ Context optimization worked: Reduced context size")
    else:
        print("⚠️ Context still within limits, no optimization needed")

def verify_large_model_optimization():
    """Verify context optimization for large models with actual docs."""
    print("\n=== Verifying Large Model Context Optimization ===")
    
    # Load many more documents to ensure we exceed large model context window
    docs = load_actual_docs(max_docs=15)
    if not docs:
        return
    
    # Make copies of the docs to ensure we have a large context
    docs = docs + docs  # Double the documents
    
    # Define large model capabilities
    large_model = {
        "model_size": "large",
        "context_window": 4096
    }
    
    # Print document stats before optimization
    total_chars = sum(len(doc.page_content) for doc in docs)
    total_tokens = sum(count_tokens(doc.page_content) for doc in docs)
    print(f"Before optimization: {len(docs)} docs, {total_chars} chars, ~{total_tokens} tokens")
    
    # Verify with a relevant query
    query = "Explain document compression techniques in detail"
    print(f"\nOptimizing for query: '{query}'")
    
    # Optimize context
    optimized_docs = optimize_context_for_model(docs, query, large_model)
    
    # Print document stats after optimization
    opt_chars = sum(len(doc.page_content) for doc in optimized_docs)
    opt_tokens = sum(count_tokens(doc.page_content) for doc in optimized_docs)
    print(f"After optimization: {len(optimized_docs)} docs, {opt_chars} chars, ~{opt_tokens} tokens")
    
    # Check if optimization worked
    if len(optimized_docs) < len(docs) or opt_chars < total_chars:
        print("✅ Context optimization worked: Reduced context size")
        
        # Print some stats about the optimized documents
        print("\nOptimized documents:")
        for i, doc in enumerate(optimized_docs):
            source = doc.metadata.get("source", "unknown")
            doc_tokens = count_tokens(doc.page_content)
            print(f"Doc {i+1}: {len(doc.page_content)} chars, ~{doc_tokens} tokens, source: {source}")
    else:
        print("⚠️ Context optimization didn't reduce size")

if __name__ == "__main__":
    print("Verifying context window optimization with actual documents...")
    verify_small_model_optimization()
    verify_large_model_optimization()
    print("\nVerification complete!") 