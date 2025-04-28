"""
Document processing and vectorstore management for MindNest application
"""

import os
from typing import Optional, Any, Dict, List

# Initialize global variables
vectorstore = None

def initialize_vectorstore():
    """Initialize or update the vector store with documents."""
    global vectorstore
    try:
        print("Initializing vector store using incremental implementation...")
        # Import here to avoid circular imports
        from mindnest.utils.incremental_vectorstore import IncrementalVectorStore
        from mindnest.core.llm_manager import embeddings
        
        # Initialize embeddings if not already initialized
        if embeddings is None:
            from mindnest.core.llm_manager import initialize_embeddings
            initialize_embeddings()
        
        # Use our optimized incremental vector store implementation
        inc_vectorstore = IncrementalVectorStore(docs_directory=os.environ.get("DOCS_DIR", "docs"))
        
        # Option to force rebuild the vector store
        force_rebuild = os.environ.get("FORCE_REBUILD", "false").lower() == "true"
        
        # Initialize or update the vector store
        vectorstore = inc_vectorstore.initialize_or_update(force_rebuild=force_rebuild)
        
        if vectorstore is not None:
            print(f"Vector store ready with {len(vectorstore.get()['ids'])} documents")
        else:
            print("Failed to initialize vector store")
            
    except Exception as e:
        print(f"Error initializing vector store: {e}")
        raise

def get_vectorstore():
    """Get the current vectorstore instance."""
    global vectorstore
    return vectorstore

def process_documents(directory: str = "docs", force_rebuild: bool = False) -> int:
    """
    Process documents in the specified directory.
    
    Args:
        directory (str): The directory to process
        force_rebuild (bool): Whether to force a rebuild of the vector store
        
    Returns:
        int: Number of documents processed
    """
    from mindnest.utils.incremental_vectorstore import IncrementalVectorStore
    
    # Create an incremental vector store instance with the correct directory
    inc_vectorstore = IncrementalVectorStore(docs_directory=directory)
    
    # Process documents and rebuild if needed
    global vectorstore
    vectorstore = inc_vectorstore.initialize_or_update(force_rebuild=force_rebuild)
    
    # Return document count
    if vectorstore is not None:
        return len(vectorstore.get()['ids'])
    return 0

def optimize_document_storage():
    """
    Optimize document storage by cleaning up chunks and index files.
    """
    try:
        # Clean up any stale chunk files
        chunk_dir = "doc_chunks"
        if os.path.exists(chunk_dir):
            for filename in os.listdir(chunk_dir):
                if filename.endswith(".json") and "_temp_" in filename:
                    os.remove(os.path.join(chunk_dir, filename))
        
        # Compact vector store if possible
        if vectorstore is not None and hasattr(vectorstore, "persist"):
            vectorstore.persist()
            
        return True
    except Exception as e:
        print(f"Error optimizing document storage: {e}")
        return False 