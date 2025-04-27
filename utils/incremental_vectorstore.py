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

from utils.config import config
from utils.logger import get_logger
from utils.document_tracker import DocumentTracker
from utils.document_processor import DocumentProcessor

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
            document_changes = self.tracker.scan_documents()
            has_changes = any(len(changes) > 0 for changes in document_changes.values())
            
            # Force rebuild if requested or if vector store doesn't exist
            if force_rebuild or not os.path.exists(self.persist_directory):
                if force_rebuild and os.path.exists(self.persist_directory):
                    logger.info("Removing existing vector store for rebuild")
                    shutil.rmtree(self.persist_directory)
                
                logger.info("Creating new vector store")
                return self._full_rebuild()
            
            # If no changes, load existing vector store
            if not has_changes:
                logger.info("No document changes detected. Loading existing vector store")
                vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                logger.info(f"Loaded existing vector store with {len(vectorstore.get()['ids'])} documents")
                return vectorstore
            
            # Process incremental updates
            logger.info("Processing incremental updates")
            new_files = document_changes["new"]
            modified_files = document_changes["modified"]
            deleted_files = document_changes["deleted"]
            
            # If we have a lot of changes, it might be more efficient to rebuild
            total_changes = len(new_files) + len(modified_files) + len(deleted_files)
            total_docs = self.tracker.get_document_count()
            
            # If more than 30% of documents changed, full rebuild is more efficient
            if total_changes > 0 and total_docs > 0 and (total_changes / total_docs) > 0.3:
                logger.info(f"Large number of changes detected ({total_changes}/{total_docs}). Performing full rebuild")
                return self._full_rebuild()
            
            # Load the existing vector store
            logger.info("Loading existing vector store for incremental update")
            vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            
            # Process new and modified files
            if new_files or modified_files:
                files_to_process = new_files + modified_files
                logger.info(f"Processing {len(files_to_process)} new or modified files")
                
                # Process documents using document processor
                documents = self.document_processor.process_files(files_to_process)
                
                if documents:
                    logger.info(f"Adding {len(documents)} document chunks to vector store")
                    vectorstore.add_documents(documents)
                    vectorstore.persist()
                    logger.info("Documents added to vector store")
                else:
                    logger.warning("No valid documents found in new or modified files")
            
            # Process deleted files
            if deleted_files:
                logger.info(f"Processing {len(deleted_files)} deleted files")
                
                # Get all document IDs to check which ones to delete
                all_ids = vectorstore.get()["ids"]
                all_metadatas = vectorstore.get()["metadatas"]
                
                # Find IDs of documents from deleted files
                ids_to_delete = []
                for i, metadata in enumerate(all_metadatas):
                    source = metadata.get("source")
                    if source and any(source == deleted_file for deleted_file in deleted_files):
                        ids_to_delete.append(all_ids[i])
                
                if ids_to_delete:
                    logger.info(f"Deleting {len(ids_to_delete)} document chunks from vector store")
                    vectorstore.delete(ids_to_delete)
                    vectorstore.persist()
                    logger.info("Documents deleted from vector store")
                else:
                    logger.info("No documents to delete from vector store")
            
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
            documents = self.document_processor.process_directory()
            
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