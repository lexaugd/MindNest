"""
Tests for the document processor module.
"""

import os
import unittest
import tempfile
from pathlib import Path
import shutil

# Import from utils module
from mindnest.utils.document_processor import DocumentProcessor

class TestDocumentProcessor(unittest.TestCase):
    """Test the document processor functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Create test files
        with open(os.path.join(self.test_dir, "test.txt"), "w") as f:
            f.write("This is a test document.\nIt has multiple lines.\nIt is used for testing.")
        
        with open(os.path.join(self.test_dir, "test.py"), "w") as f:
            f.write("def test_function():\n    print('This is a test function')\n    return True")
        
        # Create processor with test directory
        self.processor = DocumentProcessor(docs_dir=self.test_dir)
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_load_documents(self):
        """Test loading documents from directory."""
        # Load documents
        docs = self.processor.load_documents()
        
        # Check that documents were loaded
        self.assertGreater(len(docs), 0)
        
        # Check that document content was extracted
        self.assertTrue(any("test document" in doc.page_content for doc in docs))
        
        # Check that Python file was loaded
        self.assertTrue(any("test_function" in doc.page_content for doc in docs))
    
    def test_load_documents_filtered(self):
        """Test loading documents with filtering."""
        # Load only text files
        docs = self.processor.load_documents(file_types=[".txt"])
        
        # Check that only text files were loaded
        self.assertGreater(len(docs), 0)
        self.assertTrue(all(".txt" in doc.metadata.get("source", "") for doc in docs))
        
        # Check that all text files contain expected content
        self.assertTrue(all("test document" in doc.page_content for doc in docs))
    
    def test_split_documents(self):
        """Test splitting documents into chunks."""
        # Load documents
        docs = self.processor.load_documents()
        
        # Split documents
        chunks = self.processor.split_documents(docs)
        
        # Check that chunks were created
        self.assertGreaterEqual(len(chunks), len(docs))
        
        # Check that chunks have metadata
        self.assertTrue(all("source" in chunk.metadata for chunk in chunks))
    
    def test_process_documents(self):
        """Test the full document processing pipeline."""
        # Process documents
        docs = self.processor.process_documents()
        
        # Check that documents were processed
        self.assertGreater(len(docs), 0)
        
        # Check that chunks have content and metadata
        self.assertTrue(all(len(doc.page_content) > 0 for doc in docs))
        self.assertTrue(all("source" in doc.metadata for doc in docs))

if __name__ == "__main__":
    unittest.main() 