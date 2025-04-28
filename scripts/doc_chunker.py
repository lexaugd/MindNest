#!/usr/bin/env python3
"""
Semantic Document Chunker for MindNest.

This script processes documentation files and chunks them into semantically
coherent sections for better retrieval and context optimization when answering queries.

Usage:
    python doc_chunker.py
"""

import os
import re
import json
from pathlib import Path
import numpy as np
from typing import List, Dict, Any

# Try to import langchain components
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("Warning: LangChain not available. Using basic chunking only.")

class SemanticDocumentChunker:
    """
    Splits documents into semantically meaningful chunks for better retrieval.
    """
    
    def __init__(self, chunk_size=1000, chunk_overlap=100):
        """
        Initialize the document chunker.
        
        Args:
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Use LangChain's advanced text splitter if available
        if LANGCHAIN_AVAILABLE:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
                keep_separator=True
            )
        
    def chunk_document(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split a document into semantically coherent chunks.
        
        Args:
            content: The document content to split
            metadata: Metadata for the document
            
        Returns:
            List of chunk dicts with content and metadata
        """
        # Use LangChain splitter if available
        if LANGCHAIN_AVAILABLE:
            # Create LangChain document
            doc = Document(page_content=content, metadata=metadata)
            
            # Split into chunks
            chunks = self.text_splitter.split_documents([doc])
            
            # Convert to dictionaries
            return [
                {
                    "content": chunk.page_content,
                    "metadata": {
                        **chunk.metadata,
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    }
                }
                for i, chunk in enumerate(chunks)
            ]
        
        # Fallback: Simple paragraph-based chunking
        return self._simple_chunk(content, metadata)
    
    def _simple_chunk(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Simple paragraph-based document chunking.
        
        Args:
            content: The document content
            metadata: Document metadata
            
        Returns:
            List of chunks
        """
        # Split on paragraph boundaries
        paragraphs = re.split(r'\n\s*\n', content)
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            # If adding this paragraph exceeds chunk size and we already have content,
            # save the current chunk and start a new one
            if len(current_chunk) + len(para) > self.chunk_size and current_chunk:
                chunks.append(current_chunk)
                # Start new chunk with overlap
                words = current_chunk.split()
                if len(words) > 20:  # Ensure we have enough to overlap
                    overlap_text = " ".join(words[-20:])
                    current_chunk = overlap_text + "\n\n"
                else:
                    current_chunk = ""
            
            # Add paragraph to current chunk
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
        
        # Add the final chunk if not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        # Convert to dictionaries with metadata
        return [
            {
                "content": chunk,
                "metadata": {
                    **metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            }
            for i, chunk in enumerate(chunks)
        ]
    
    def process_file(self, file_path: str, output_dir: str = None) -> List[Dict[str, Any]]:
        """
        Process a single file into chunks.
        
        Args:
            file_path: Path to the file to process
            output_dir: Directory to save individual chunk files (if None, won't save files)
            
        Returns:
            List of chunk dictionaries
        """
        try:
            # Read the file
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Create metadata
            rel_path = os.path.relpath(file_path)
            filename = os.path.basename(file_path)
            file_type = os.path.splitext(filename)[1].lstrip('.')
            
            metadata = {
                "source": file_path,
                "relative_path": rel_path,
                "filename": filename,
                "filetype": file_type
            }
            
            # Chunk the document
            chunks = self.chunk_document(content, metadata)
            
            # Save individual chunks if output_dir is specified
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                base_name = os.path.splitext(filename)[0]
                
                for i, chunk in enumerate(chunks):
                    chunk_filename = f"{base_name}_chunk{i+1}.json"
                    chunk_path = os.path.join(output_dir, chunk_filename)
                    
                    with open(chunk_path, 'w', encoding='utf-8') as f:
                        json.dump(chunk, f, indent=2)
            
            return chunks
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return []
    
    def process_directory(self, directory: str, output_dir: str = None, 
                         file_types: List[str] = None) -> List[Dict[str, Any]]:
        """
        Process all files in a directory.
        
        Args:
            directory: Directory to process
            output_dir: Directory to save chunk files (if None, won't save)
            file_types: List of file extensions to process (e.g., ['md', 'txt'])
            
        Returns:
            List of all chunks
        """
        if file_types is None:
            file_types = ['md', 'txt', 'html', 'text']
        
        all_chunks = []
        
        # Walk through directory
        for root, dirs, files in os.walk(directory):
            for file in files:
                ext = os.path.splitext(file)[1].lstrip('.')
                
                # Skip files that don't match our file types
                if ext.lower() not in file_types:
                    continue
                
                file_path = os.path.join(root, file)
                print(f"Processing {file_path}")
                
                # Process the file
                chunks = self.process_file(file_path, output_dir)
                all_chunks.extend(chunks)
        
        return all_chunks

def main():
    """Main function to run the document chunker."""
    # Set parameters
    chunk_size = 1000
    chunk_overlap = 100
    docs_dir = "docs"
    chunks_dir = "doc_chunks"
    
    print(f"Starting semantic document chunking with size={chunk_size}, overlap={chunk_overlap}")
    
    # Create chunker
    chunker = SemanticDocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # Process all documents
    all_chunks = chunker.process_directory(docs_dir, chunks_dir)
    
    print(f"Processed {len(all_chunks)} chunks from documents in {docs_dir}")
    print(f"Chunk files saved to {chunks_dir}")

if __name__ == "__main__":
    main() 