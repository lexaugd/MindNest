"""
Tests for the document processor module.
"""

import os
import unittest
from unittest.mock import patch, MagicMock
from tempfile import TemporaryDirectory
from pathlib import Path

from utils.document_processor import DocumentProcessor

class TestDocumentProcessor(unittest.TestCase):
    """Test cases for the DocumentProcessor class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test files
        self.temp_dir = TemporaryDirectory()
        self.docs_dir = Path(self.temp_dir.name)
        
        # Create a test document processor
        self.processor = DocumentProcessor(
            docs_dir=self.docs_dir,
            chunk_size=500,
            chunk_overlap=100
        )
        
        # Create some test files
        self.test_files = []
        self._create_test_files()
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def _create_test_files(self):
        """Create test files in the temporary directory."""
        # Create a text file
        text_file = self.docs_dir / "test.txt"
        with open(text_file, "w") as f:
            f.write("This is a test document.\nIt has multiple lines.\nThis is for testing purposes.")
        self.test_files.append(text_file)
        
        # Create a Python file
        py_file = self.docs_dir / "test.py"
        with open(py_file, "w") as f:
            f.write("""def test_function():
    \"\"\"This is a test function.\"\"\"
    return "Hello, world!"
""")
        self.test_files.append(py_file)
    
    def test_initialization(self):
        """Test that the processor initializes correctly."""
        self.assertEqual(self.processor.docs_dir, self.docs_dir)
        self.assertEqual(self.processor.chunk_size, 500)
        self.assertEqual(self.processor.chunk_overlap, 100)
    
    def test_get_loader_for_file(self):
        """Test the get_loader_for_file method."""
        # Test with a text file
        loader = self.processor.get_loader_for_file(self.test_files[0])
        self.assertIsNotNone(loader)
        
        # Test with a Python file
        loader = self.processor.get_loader_for_file(self.test_files[1])
        self.assertIsNotNone(loader)
        
        # Test with an unsupported file type
        unsupported_file = self.docs_dir / "test.xyz"
        with open(unsupported_file, "w") as f:
            f.write("Unsupported file type")
        
        loader = self.processor.get_loader_for_file(unsupported_file)
        self.assertIsNone(loader)
    
    def test_load_file(self):
        """Test the load_file method."""
        # Test loading a text file
        docs = self.processor.load_file(self.test_files[0])
        self.assertTrue(len(docs) > 0)
        self.assertIn("This is a test document.", docs[0].page_content)
    
    @patch("utils.document_processor.DirectoryLoader")
    def test_load_directory(self, mock_loader):
        """Test the load_directory method."""
        # Mock the DirectoryLoader
        mock_instance = MagicMock()
        mock_loader.return_value = mock_instance
        mock_instance.load.return_value = ["test document"]
        
        # Call the method
        result = self.processor.load_directory()
        
        # Check that the loader was called at least once
        self.assertTrue(mock_loader.called)
        
        # Check that the result contains the expected extension
        for ext in self.processor.loaders.keys():
            file_path = self.docs_dir / f"test{ext}"
            with open(file_path, "w") as f:
                f.write(f"Test file with extension {ext}")
        
        # Test with real files
        with patch("utils.document_processor.logger"):  # Suppress logging
            result = self.processor.load_directory()
            # Check that we got some results
            self.assertTrue(len(result) > 0)
    
    def test_chunk_documents(self):
        """Test the chunk_documents method."""
        # Create test documents
        from langchain.docstore.document import Document
        docs = [
            Document(page_content="This is the first test document with enough text to be split into multiple chunks based on the test configuration. " * 10),
            Document(page_content="This is the second test document with enough text to be split into multiple chunks based on the test configuration. " * 10)
        ]
        
        # Chunk the documents
        chunks = self.processor.chunk_documents(docs)
        
        # Check that we got more chunks than original documents
        self.assertTrue(len(chunks) > len(docs))
    
    def test_process_files(self):
        """Test the process_files method."""
        # Process the test files
        chunks = self.processor.process_files(self.test_files)
        
        # Check that we got some chunks
        self.assertTrue(len(chunks) > 0)
    
    def test_process_directory(self):
        """Test the process_directory method."""
        # Process the test directory
        chunks = self.processor.process_directory(self.docs_dir)
        
        # Check that we got some chunks
        self.assertTrue(len(chunks) > 0)

if __name__ == "__main__":
    unittest.main() 