"""
Document processing functionality for MindNest.
Provides unified document loading, processing, and chunking.
"""

import os
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredMarkdownLoader,
    PythonLoader,
    DirectoryLoader,
    PDFMinerLoader
)

from utils.config import config
from utils.logger import get_logger

# Get module logger
logger = get_logger(__name__)

class DocumentProcessor:
    """Centralized document processing for the application."""
    
    def __init__(
        self,
        docs_dir: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ):
        """
        Initialize document processor with configuration.
        
        Args:
            docs_dir: Directory containing documents (default: from config)
            chunk_size: Size of document chunks (default: from config)
            chunk_overlap: Overlap between chunks (default: from config)
        """
        # Use provided values or defaults from config
        self.docs_dir = docs_dir if docs_dir is not None else config.docs_dir
        self.chunk_size = chunk_size if chunk_size is not None else config.chunk_size
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else config.chunk_overlap
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        
        # Configure loaders for different file types
        self.loaders = {
            ".txt": TextLoader,
            ".md": UnstructuredMarkdownLoader,
            ".py": PythonLoader,
            ".groovy": TextLoader,
            ".java": TextLoader,
            ".js": TextLoader,
            ".ts": TextLoader,
            ".pdf": PDFMinerLoader,
        }
    
    def get_loader_for_file(self, file_path: Union[str, Path]) -> Optional[Any]:
        """
        Get the appropriate loader for a file based on its extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            A loader instance for the file type or None if unsupported
        """
        file_path = str(file_path)
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if ext not in self.loaders:
            logger.warning(f"No loader available for {ext} files. Skipping {file_path}")
            return None
            
        loader_class = self.loaders[ext]
        return loader_class(file_path)
    
    def load_file(self, file_path: Union[str, Path]) -> List[Document]:
        """
        Load a document from a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of loaded documents
        """
        try:
            loader = self.get_loader_for_file(file_path)
            if loader is None:
                return []
                
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} documents from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            return []
    
    def load_directory(self, directory: Optional[Union[str, Path]] = None) -> Dict[str, List[Document]]:
        """
        Load all documents from a directory.
        
        Args:
            directory: Directory to load documents from (default: self.docs_dir)
            
        Returns:
            Dictionary mapping file extensions to lists of documents
        """
        directory = directory if directory is not None else self.docs_dir
        
        result = {}
        for ext, loader_class in self.loaders.items():
            try:
                # Create glob pattern for this extension
                glob_pattern = f"**/*{ext}"
                
                # Create directory loader
                loader = DirectoryLoader(
                    directory,
                    glob=glob_pattern,
                    loader_cls=loader_class
                )
                
                # Load documents
                docs = loader.load()
                
                if docs:
                    logger.info(f"Loaded {len(docs)} {ext} files from {directory}")
                    result[ext] = docs
                else:
                    logger.info(f"No {ext} files found in {directory}")
                
            except Exception as e:
                logger.error(f"Error loading {ext} files from {directory}: {e}")
        
        return result
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks for better processing.
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of document chunks
        """
        if not documents:
            logger.warning("No documents to chunk")
            return []
            
        try:
            logger.info(f"Splitting {len(documents)} documents into chunks")
            split_docs = self.text_splitter.split_documents(documents)
            logger.info(f"Document splitting complete. Created {len(split_docs)} chunks")
            return split_docs
        except Exception as e:
            logger.error(f"Error splitting documents: {e}")
            return documents
    
    def process_files(self, file_paths: List[Union[str, Path]]) -> List[Document]:
        """
        Process multiple files and return chunked documents.
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            List of chunked document objects
        """
        # Load documents from each file
        all_docs = []
        for file_path in file_paths:
            docs = self.load_file(file_path)
            all_docs.extend(docs)
        
        # Chunk the documents
        if all_docs:
            return self.chunk_documents(all_docs)
        else:
            return []
    
    def process_directory(self, directory: Optional[Union[str, Path]] = None) -> List[Document]:
        """
        Process all documents in a directory and return chunked documents.
        
        Args:
            directory: Directory to process (default: self.docs_dir)
            
        Returns:
            List of chunked document objects
        """
        # Load documents from directory
        docs_by_type = self.load_directory(directory)
        
        # Combine all documents
        all_docs = []
        for docs in docs_by_type.values():
            all_docs.extend(docs)
        
        # Chunk the documents
        if all_docs:
            return self.chunk_documents(all_docs)
        else:
            return []


# For testing purposes
if __name__ == "__main__":
    processor = DocumentProcessor()
    document_chunks = processor.process_directory()
    print(f"Processed {len(document_chunks)} document chunks")
    
    if document_chunks:
        # Print first document details
        doc = document_chunks[0]
        print("\nSample document:")
        print(f"Content (first 100 chars): {doc.page_content[:100]}...")
        print(f"Metadata: {doc.metadata}") 