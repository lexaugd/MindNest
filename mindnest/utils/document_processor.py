"""
Document processing utilities for MindNest.

This module contains utilities for processing various document types
and preparing them for inclusion in the vector store.
"""

import os
import glob
from typing import List, Dict, Optional, Any, Union
from pathlib import Path

from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
    UnstructuredMarkdownLoader,
    PythonLoader,
    CSVLoader,
    PDFMinerLoader,
    JSONLoader
)
from langchain.document_loaders.base import BaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from mindnest.core.config import config
from mindnest.utils.logger import get_logger

# Get module logger
logger = get_logger(__name__)

class DocumentProcessor:
    """
    Handles loading and processing of documents from various sources.
    Supports multiple file types and document chunking.
    """
    
    def __init__(self, docs_dir=None, chunk_size=None, chunk_overlap=None):
        """
        Initialize document processor with configuration.
        
        Args:
            docs_dir: Directory containing documents to process
            chunk_size: Size of text chunks for splitting documents
            chunk_overlap: Overlap between chunks
        """
        # Use provided values or defaults from config
        self.docs_dir = Path(docs_dir) if docs_dir else Path(config.docs_dir)
        self.chunk_size = chunk_size if chunk_size else config.chunk_size
        self.chunk_overlap = chunk_overlap if chunk_overlap else config.chunk_overlap
        
        # Define loaders for different file types
        self.loaders = {
            ".txt": TextLoader,
            ".md": UnstructuredMarkdownLoader,
            ".py": PythonLoader,
            ".js": TextLoader,
            ".ts": TextLoader,
            ".html": TextLoader,
            ".css": TextLoader,
            ".json": JSONLoader,
            ".csv": CSVLoader,
            ".pdf": PDFMinerLoader,
            ".java": TextLoader,
            ".groovy": TextLoader,
            # Add more loaders as needed
        }
    
    def get_loader_for_file(self, file_path: Union[str, Path]) -> Optional[BaseLoader]:
        """
        Get the appropriate loader for a file based on its extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            BaseLoader: The appropriate loader or None if not supported
        """
        file_path = Path(file_path)
        file_extension = file_path.suffix.lower()
        
        if file_extension in self.loaders:
            try:
                if file_extension == ".json":
                    # JSONLoader requires a jq-style path to the text field
                    return self.loaders[file_extension](file_path=str(file_path), jq_schema='.', text_content=False)
                else:
                    return self.loaders[file_extension](file_path=str(file_path))
            except Exception as e:
                logger.error(f"Error creating loader for {file_path}: {e}")
                return None
        else:
            logger.warning(f"No loader available for {file_extension} files. Skipping {file_path}")
            return None
    
    def load_file(self, file_path: Union[str, Path]) -> List[Document]:
        """
        Load a single file into documents.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List[Document]: Loaded documents
        """
        loader = self.get_loader_for_file(file_path)
        if loader:
            try:
                docs = loader.load()
                logger.info(f"Loaded {len(docs)} documents from {file_path}")
                return docs
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        return []
    
    def load_directory(self, path: Optional[Union[str, Path]] = None, 
                      file_types: Optional[List[str]] = None) -> List[Document]:
        """
        Load all supported documents from a directory.
        
        Args:
            path: Directory path (default: self.docs_dir)
            file_types: List of file extensions to include
            
        Returns:
            List[Document]: Loaded documents
        """
        path = Path(path) if path else self.docs_dir
        file_types = file_types or list(self.loaders.keys())
        
        if not path.exists():
            logger.warning(f"Directory {path} does not exist")
            return []
        
        all_docs = []
        
        # Process each supported file type
        for file_type in file_types:
            # Skip if the file type doesn't have a loader
            if file_type not in self.loaders:
                logger.warning(f"No loader configured for {file_type} files. Skipping.")
                continue
            
            # Use glob pattern to find all files of this type
            pattern = str(path / f"**/*{file_type}")
            matching_files = glob.glob(pattern, recursive=True)
            
            # Log what we found
            if matching_files:
                logger.info(f"Loaded {len(matching_files)} {file_type} files from {path}")
            else:
                logger.info(f"No {file_type} files found in {path}")
            
            # Load each file
            for file_path in matching_files:
                docs = self.load_file(file_path)
                all_docs.extend(docs)
        
        return all_docs
    
    def load_documents(self, directory: Optional[Union[str, Path]] = None,
                      file_types: Optional[List[str]] = None) -> List[Document]:
        """
        Load documents from a directory with custom filtering.
        
        Args:
            directory: Directory to load from
            file_types: List of file extensions to load
            
        Returns:
            List[Document]: Loaded documents
        """
        return self.load_directory(path=directory, file_types=file_types)
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of documents to split
            
        Returns:
            List[Document]: Chunked documents
        """
        if not documents:
            return []
        
        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
        
        # Split documents
        logger.info(f"Splitting {len(documents)} documents into chunks")
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Document splitting complete. Created {len(chunks)} chunks")
        
        return chunks
    
    def process_documents(self, directory: Optional[Union[str, Path]] = None,
                         file_types: Optional[List[str]] = None) -> List[Document]:
        """
        Process documents from a directory, including loading and chunking.
        
        Args:
            directory: Directory to process
            file_types: List of file extensions to process
            
        Returns:
            List[Document]: Processed document chunks
        """
        # Load documents
        documents = self.load_documents(directory=directory, file_types=file_types)
        
        # Split documents and return
        return self.split_documents(documents)
    
    def process_files(self, file_paths: List[Union[str, Path]]) -> List[Document]:
        """
        Process a list of files, including loading and chunking.
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            List[Document]: Processed document chunks
        """
        # Load documents
        documents = []
        for file_path in file_paths:
            docs = self.load_file(file_path)
            documents.extend(docs)
        
        # Split documents and return
        return self.split_documents(documents)
    
    def process_directory(self, directory: Union[str, Path]) -> List[Document]:
        """
        Process a single directory, including loading and chunking.
        
        Args:
            directory: Directory to process
            
        Returns:
            List[Document]: Processed document chunks
        """
        return self.process_documents(directory=directory) 