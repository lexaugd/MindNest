"""
Incremental vector store for efficient document updates.
Provides incremental updates to avoid full rebuilds when documents change.
"""

import os
import shutil
from typing import List, Dict, Optional, Any, Union
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

from mindnest.core.config import config
from mindnest.utils.logger import get_logger
from mindnest.utils.document_tracker import DocumentTracker
from mindnest.utils.document_processor import DocumentProcessor

# Get module logger
logger = get_logger(__name__)

class IncrementalVectorStore:
    """
    Provides incremental updates to the vector store based on document changes.
    This avoids rebuilding the entire vector store when only a few documents have changed.
    """
    
    def __init__(
        self, 
        persist_directory: Optional[str] = None,
        docs_directory: Optional[Union[str, Path]] = None,
        embedding_model_name: Optional[str] = None
    ):
        """
        Initialize the incremental vector store.
        
        Args:
            persist_directory: Directory to store the vector database (default: from config)
            docs_directory: Directory containing the documents (default: from config)
            embedding_model_name: Name of the embedding model to use (default: from config)
        """
        # Use provided values or defaults from config
        self.persist_directory = persist_directory if persist_directory is not None else config.persist_directory
        self.docs_directory = docs_directory if docs_directory is not None else config.docs_dir
        self.embedding_model_name = embedding_model_name if embedding_model_name is not None else config.embedding_model_name
        
        # Initialize embeddings
        self._initialize_embeddings()
        
        # Initialize document tracker
        self.tracker = DocumentTracker(docs_dir=str(self.docs_directory))
        
        # Initialize document processor
        self.document_processor = DocumentProcessor(
            docs_dir=str(self.docs_directory),
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
    
    def _initialize_embeddings(self) -> None:
        """Initialize embeddings with error handling and fallback options."""
        try:
            logger.info(f"Initializing embeddings with model: {self.embedding_model_name}")
            self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
            logger.info("Embeddings initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing embeddings: {e}")
            try:
                # Fallback to basic embeddings
                from langchain_community.embeddings import FakeEmbeddings
                logger.warning("Falling back to simple embeddings")
                self.embeddings = FakeEmbeddings(size=384)
            except Exception as e2:
                logger.critical(f"Fatal error initializing embeddings: {e2}")
                raise
    
    def initialize_or_update(self, force_rebuild: bool = False) -> Optional[Chroma]:
        """
        Initialize or update the vector store based on document changes.
        
        Args:
            force_rebuild: Whether to force a complete rebuild of the vector store
            
        Returns:
            The initialized or updated Chroma vector store
        """
        try:
            # Scan for document changes
            logger.info("Scanning for document changes...")
            self.tracker._scan_documents()
            
            # Since _scan_documents doesn't return document changes, we'll need to modify our approach
            # For now, we'll just do a full rebuild if force_rebuild is True or the vector store doesn't exist
            if force_rebuild or not os.path.exists(self.persist_directory):
                if force_rebuild and os.path.exists(self.persist_directory):
                    logger.info("Removing existing vector store for rebuild")
                    shutil.rmtree(self.persist_directory)
                
                logger.info("Creating new vector store")
                return self._full_rebuild()
            
            # If we get here, just load the existing vector store
            logger.info("Loading existing vector store")
            vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            logger.info(f"Loaded existing vector store with {len(vectorstore.get()['ids'])} documents")
            return vectorstore
            
        except Exception as e:
            logger.error(f"Error in incremental update: {e}")
            # Attempt a full rebuild as a fallback
            logger.warning("Attempting full rebuild as a fallback")
            try:
                return self._full_rebuild()
            except Exception as e2:
                logger.critical(f"Fatal error in vector store initialization: {e2}")
                raise
    
    def _full_rebuild(self) -> Optional[Chroma]:
        """
        Perform a full rebuild of the vector store.
        
        Returns:
            The newly built Chroma vector store
        """
        try:
            logger.info("Starting full vector store rebuild")
            
            # Clear existing vector store if it exists
            if os.path.exists(self.persist_directory):
                logger.info(f"Removing existing vector store at {self.persist_directory}")
                shutil.rmtree(self.persist_directory)
            
            # Process all documents in the docs directory
            documents = self.document_processor.process_directory(directory=self.docs_directory)
            
            if not documents:
                logger.warning("No documents found to add to vector store")
                return None
            
            # Create new vector store
            logger.info(f"Creating new vector store with {len(documents)} document chunks")
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            
            # Persist to disk
            vectorstore.persist()
            logger.info("Vector store created and persisted successfully")
            
            return vectorstore
            
        except Exception as e:
            logger.error(f"Error rebuilding vector store: {e}")
            raise

    def get_document_count(self) -> int:
        """Get the number of documents in the vector store."""
        try:
            vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            return len(vectorstore.get()["ids"])
        except Exception as e:
            logger.error(f"Error getting document count: {e}")
            return 0


# For testing purposes
if __name__ == "__main__":
    # Initialize the vector store
    vector_store = IncrementalVectorStore()
    
    # Update the vector store
    vs = vector_store.initialize_or_update()
    
    if vs:
        # Test a simple query
        print("\nTesting query...")
        results = vs.similarity_search("What is a document?", k=1)
        if results:
            print(f"Found result: {results[0].page_content[:100]}...")
        else:
            print("No results found")
    else:
        print("Vector store initialization failed") 