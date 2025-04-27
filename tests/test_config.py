"""
Tests for the configuration module.
"""

import os
import unittest
from unittest.mock import patch

from utils.config import Config

class TestConfig(unittest.TestCase):
    """Test cases for the Config class."""
    
    def test_default_values(self):
        """Test that default values are set correctly."""
        config = Config()
        
        # Check some default values
        self.assertEqual(config.persist_directory, "chroma_db")
        self.assertEqual(config.embedding_model_name, "all-MiniLM-L6-v2")
        self.assertEqual(config.chunk_size, 1000)
        self.assertEqual(config.chunk_overlap, 200)
        self.assertEqual(config.port, 8000)
    
    @patch.dict(os.environ, {"CHUNK_SIZE": "2000", "CHUNK_OVERLAP": "300"})
    def test_environment_override(self):
        """Test that environment variables override default values."""
        config = Config()
        
        # Check that environment variables are applied
        self.assertEqual(config.chunk_size, 2000)
        self.assertEqual(config.chunk_overlap, 300)
    
    def test_model_path(self):
        """Test the get_model_path method."""
        config = Config()
        
        # Test with default setting
        expected_path = str(config.models_dir / config.model_name)
        self.assertEqual(config.get_model_path(), expected_path)
        
        # Test with small model setting
        expected_small_path = str(config.models_dir / config.small_model_name)
        self.assertEqual(config.get_model_path(use_small_model=True), expected_small_path)
    
    def test_to_dict(self):
        """Test the to_dict method."""
        config = Config()
        config_dict = config.to_dict()
        
        # Check that the dictionary contains expected keys
        self.assertIn("chunk_size", config_dict)
        self.assertIn("chunk_overlap", config_dict)
        self.assertIn("port", config_dict)
        
        # Check that the values match
        self.assertEqual(config_dict["chunk_size"], config.chunk_size)
        self.assertEqual(config_dict["port"], config.port)

if __name__ == "__main__":
    unittest.main() 