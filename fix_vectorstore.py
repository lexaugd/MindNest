#!/usr/bin/env python
"""
Script to fix the vector store initialization and ensure it's properly loaded.
This will rebuild the vector store with the correct parameters and update the global variable.
"""

import sys
import os
import shutil
from pathlib import Path

# Ensure script can find the application modules
script_path = Path(__file__).resolve().parent
if str(script_path) not in sys.path:
    sys.path.insert(0, str(script_path))

def clean_vector_store():
    """Remove the existing vector store completely to start fresh."""
    print("Cleaning up the vector store...")
    
    # Clear the chroma_db directory completely
    chroma_dir = os.path.join(script_path, "chroma_db")
    if os.path.exists(chroma_dir):
        print(f"Removing existing vector store at {chroma_dir}")
        shutil.rmtree(chroma_dir)
    
    # Create a fresh directory with proper permissions
    os.makedirs(chroma_dir, exist_ok=True)
    os.chmod(chroma_dir, 0o777)  # Full permissions
    print(f"Created fresh vector store directory at {chroma_dir}")
    
    return True

def rebuild_vector_store():
    """Rebuild the vector store and correctly update the global variable."""
    print("Rebuilding vector store...")
    
    # First, import the necessary modules
    from mindnest.utils.incremental_vectorstore import IncrementalVectorStore
    from mindnest.core import document_processor
    
    # Create a fresh vector store instance
    inc_vectorstore = IncrementalVectorStore()
    
    # Force a rebuild of the vector store
    vs = inc_vectorstore.initialize_or_update(force_rebuild=True)
    
    # Update the global variable in the document_processor module
    document_processor.vectorstore = vs
    
    # Check if the update was successful
    if document_processor.vectorstore is not None:
        doc_count = len(document_processor.vectorstore.get()['ids'])
        print(f"✅ Vector store rebuilt successfully with {doc_count} document chunks")
        return True
    else:
        print("❌ Failed to update the global vector store variable")
        return False

def check_vector_store():
    """Check if the vector store is properly loaded and accessible."""
    from mindnest.core.document_processor import get_vectorstore
    
    vs = get_vectorstore()
    if vs is None:
        print("❌ Vector store is not initialized")
        return False
    
    try:
        doc_count = len(vs.get()['ids'])
        print(f"✅ Vector store is properly initialized with {doc_count} document chunks")
        return True
    except Exception as e:
        print(f"❌ Error accessing vector store: {e}")
        return False

if __name__ == "__main__":
    # Always clean the vector store first to avoid schema issues
    clean_vector_store()
    
    # Then rebuild it
    if rebuild_vector_store():
        # Verify the rebuild worked
        check_vector_store()
    else:
        print("Failed to fix vector store")
        sys.exit(1)
    
    print("\nVector store verification:")
    print("---------------------------")
    
    # Test a simple query to verify functionality
    from mindnest.core.document_processor import get_vectorstore
    vs = get_vectorstore()
    
    if vs:
        print("Testing query...")
        results = vs.similarity_search("What is EDI?", k=2)
        if results:
            print(f"Found {len(results)} results")
            for i, result in enumerate(results):
                print(f"\nResult {i+1}:")
                print(f"Source: {result.metadata['source']}")
                print(f"Content: {result.page_content[:150]}...")
        else:
            print("No results found for test query")
    
    print("\nDone.") 