"""
Server runner for MindNest.
Starts the FastAPI server with vector search capabilities.
"""

import os
import sys
import uvicorn
from time import time

from utils.config import config
from utils.logger import app_logger as logger
from utils.incremental_vectorstore import IncrementalVectorStore

# Import app after logger initialization to ensure proper logging
from main import app

def initialize_vectorstore():
    """Initialize just the vector store (skip LLM)."""
    logger.info("Initializing vector store only...")
    vector_store = IncrementalVectorStore()
    return vector_store.initialize_or_update()

if __name__ == "__main__":
    # Record start time
    start_time = time()
    
    # Make sure we're in the correct directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Initialize vector store
    vectorstore = initialize_vectorstore()
    
    if vectorstore:
        logger.info(f"Vector store initialized with {len(vectorstore.get()['ids'])} documents")
    else:
        logger.warning("Vector store initialization failed")
    
    # Log initialization time
    logger.info(f"Initialization completed in {time() - start_time:.2f} seconds")
    
    # Run the FastAPI server
    logger.info(f"Starting server on {config.host}:{config.port} with vector search capabilities only...")
    uvicorn.run(app, host=config.host, port=config.port) 