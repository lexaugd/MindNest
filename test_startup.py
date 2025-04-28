#!/usr/bin/env python
"""
Verification script to test the MindNest startup process.
This script imports the necessary components and verifies they can be initialized.
"""

import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_imports():
    """Test importing critical modules."""
    logger.info("Verifying imports...")
    try:
        # Core imports
        from mindnest.core.llm_manager import initialize_embeddings, get_model_capabilities
        from mindnest.core.document_processor import initialize_vectorstore
        from mindnest.core.config import initialize_query_classifier
        
        # Utils imports
        from mindnest.utils.document_compression import optimize_context_for_model
        from mindnest.utils.responses import format_response
        
        # API imports
        from mindnest.api.endpoints import router
        
        # App import
        from mindnest.app import app, run_server
        
        logger.info("All imports successful!")
        return True
    except ImportError as e:
        logger.error(f"Import error: {e}")
        return False

def verify_initialization():
    """Test initializing key components."""
    logger.info("Verifying component initialization...")
    try:
        # Initialize vector store
        from mindnest.core.document_processor import initialize_vectorstore
        vectorstore = initialize_vectorstore()
        logger.info("Vector store initialized successfully")
        
        # Initialize embeddings
        from mindnest.core.llm_manager import initialize_embeddings
        embeddings = initialize_embeddings()
        logger.info("Embeddings initialized successfully")
        
        # Get model capabilities
        from mindnest.core.llm_manager import get_model_capabilities
        capabilities = get_model_capabilities()
        logger.info(f"Model capabilities retrieved: {capabilities['model_size']} model")
        
        return True
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("Starting MindNest verification script")
    
    # Verify imports
    if not verify_imports():
        logger.error("Import verification failed")
        sys.exit(1)
    
    # Verify initialization
    if not verify_initialization():
        logger.error("Initialization verification failed")
        sys.exit(1)
    
    logger.info("Verification completed successfully!")
    sys.exit(0) 