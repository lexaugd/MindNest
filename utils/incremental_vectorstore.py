import os
import shutil
from typing import List, Dict, Optional, Any
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredMarkdownLoader,
    PythonLoader
)
from langchain.docstore.document import Document

from utils.document_tracker import DocumentTracker

class IncrementalVectorStore:
    """
    Provides incremental updates to the vector store based on document changes.
    This avoids rebuilding the entire vector store when only a few documents have changed.
    """
    
    def __init__(
        self, 
        persist_directory: str = "chroma_db",
        docs_directory: str = "docs",
        embedding_model_name: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize the incremental vector store.
        
        Args:
            persist_directory: Directory to store the vector database
            docs_directory: Directory containing the documents
            embedding_model_name: Name of the embedding model to use
        """
        self.persist_directory = persist_directory
        self.docs_directory = docs_directory
        self.embedding_model_name = embedding_model_name
        
        # Initialize embeddings
        try:
            print("Initializing embeddings...")
            self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
            print("Embeddings initialized successfully")
        except Exception as e:
            print(f"Error initializing embeddings: {e}")
            try:
                # Fallback to basic embeddings
                from langchain_community.embeddings import FakeEmbeddings
                print("Falling back to simple embeddings")
                self.embeddings = FakeEmbeddings(size=384)
            except Exception as e2:
                print(f"Fatal error initializing embeddings: {e2}")
                raise
        
        # Initialize document tracker
        self.tracker = DocumentTracker(docs_dir=docs_directory)
        
        # Initialize text splitter with default values
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        
        # Initialize loaders for different file types
        self.loaders = {
            ".txt": TextLoader,
            ".md": UnstructuredMarkdownLoader,
            ".py": PythonLoader,
            ".groovy": TextLoader,
            ".java": TextLoader,
            ".js": TextLoader,
            ".ts": TextLoader,
        }
    
    def _get_documents_from_files(self, file_paths: List[str]) -> List[Document]:
        """
        Load documents from specific file paths.
        
        Args:
            file_paths: List of file paths to load documents from
            
        Returns:
            List of loaded documents
        """
        documents = []
        
        for file_path in file_paths:
            try:
                # Get the file extension
                _, ext = os.path.splitext(file_path)
                ext = ext.lower()
                
                # Skip if we don't have a loader for this file type
                if ext not in self.loaders:
                    print(f"No loader available for {ext} files. Skipping {file_path}")
                    continue
                
                # Load the document
                loader_class = self.loaders[ext]
                loader = loader_class(file_path)
                file_docs = loader.load()
                
                if file_docs:
                    print(f"Loaded {len(file_docs)} documents from {file_path}")
                    documents.extend(file_docs)
                else:
                    print(f"No content loaded from {file_path}")
            except Exception as e:
                print(f"Error loading document {file_path}: {e}")
        
        return documents
    
    def _chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks for better vector search.
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of document chunks
        """
        if not documents:
            return []
            
        print(f"Splitting {len(documents)} documents into chunks")
        try:
            split_docs = self.text_splitter.split_documents(documents)
            print(f"Document splitting complete. Created {len(split_docs)} chunks")
            return split_docs
        except Exception as e:
            print(f"Error splitting documents: {e}")
            return documents
    
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
            print("Scanning for document changes...")
            document_changes = self.tracker.scan_documents()
            has_changes = any(len(changes) > 0 for changes in document_changes.values())
            
            # Force rebuild if requested or if vector store doesn't exist
            if force_rebuild or not os.path.exists(self.persist_directory):
                if force_rebuild and os.path.exists(self.persist_directory):
                    print("Removing existing vector store for rebuild")
                    shutil.rmtree(self.persist_directory)
                
                print("Creating new vector store")
                return self._full_rebuild()
            
            # If no changes, load existing vector store
            if not has_changes:
                print("No document changes detected. Loading existing vector store")
                vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                print(f"Loaded existing vector store with {len(vectorstore.get()['ids'])} documents")
                return vectorstore
            
            # Process incremental updates
            print("Processing incremental updates")
            new_files = document_changes["new"]
            modified_files = document_changes["modified"]
            deleted_files = document_changes["deleted"]
            
            # If we have a lot of changes, it might be more efficient to rebuild
            total_changes = len(new_files) + len(modified_files) + len(deleted_files)
            total_docs = self.tracker.get_document_count()
            
            # If more than 30% of documents changed, full rebuild is more efficient
            if total_changes > 0 and total_docs > 0 and (total_changes / total_docs) > 0.3:
                print(f"Large number of changes detected ({total_changes}/{total_docs}). Performing full rebuild")
                return self._full_rebuild()
            
            # Load the existing vector store
            print("Loading existing vector store for incremental update")
            vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            
            # Process new and modified files
            if new_files or modified_files:
                files_to_process = new_files + modified_files
                print(f"Processing {len(files_to_process)} new or modified files")
                
                # Load documents from files
                documents = self._get_documents_from_files(files_to_process)
                
                # Split documents into chunks
                chunks = self._chunk_documents(documents)
                
                if chunks:
                    # Add documents to vector store
                    print(f"Adding {len(chunks)} document chunks to vector store")
                    vectorstore.add_documents(chunks)
            
            # Process deleted files (would require document IDs linked to file paths)
            # This is more complex and would require tracking document IDs by file path
            if deleted_files:
                print(f"Found {len(deleted_files)} deleted files. Note: Document deletion not implemented yet")
                # In a more complex implementation, we would:
                # 1. Track which document IDs belong to which file paths
                # 2. Remove those document IDs from the vector store
            
            # Persist the vector store
            print("Persisting vector store")
            vectorstore.persist()
            print("Vector store updated and persisted successfully")
            
            return vectorstore
            
        except Exception as e:
            print(f"Error initializing or updating vector store: {e}")
            return None
    
    def _full_rebuild(self) -> Optional[Chroma]:
        """
        Perform a full rebuild of the vector store.
        
        Returns:
            The rebuilt Chroma vector store
        """
        try:
            print("Performing full vector store rebuild")
            
            # Scan all documents to ensure tracker is up to date
            self.tracker.scan_documents()
            
            # Get all files from tracker
            all_files = list(self.tracker.tracking_data["files"].keys())
            print(f"Loading {len(all_files)} documents")
            
            documents = self._get_documents_from_files(all_files)
            
            if not documents:
                print("No documents to add to vector store!")
                return None
            
            # Split documents into chunks
            chunks = self._chunk_documents(documents)
            
            if not chunks:
                print("No document chunks created!")
                return None
            
            # Create new vector store
            print(f"Adding {len(chunks)} document chunks to vector store")
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            
            # Persist the vector store
            print("Persisting vector store")
            vectorstore.persist()
            print("Vector store created and persisted successfully")
            
            return vectorstore
        
        except Exception as e:
            print(f"Error rebuilding vector store: {e}")
            return None


# For testing purposes
if __name__ == "__main__":
    vector_store = IncrementalVectorStore()
    vs = vector_store.initialize_or_update()
    
    if vs is not None:
        print(f"Vector store ready with {len(vs.get()['ids'])} documents") 