"""
Tests for the configuration module.
"""

import os
import unittest
import tempfile
from pathlib import Path

from mindnest.core.config import Config

class TestConfig(unittest.TestCase):
    """Test the configuration system."""
    
    def test_default_values(self):
        """Test that default values are set correctly."""
        # Create a config with default values
        config = Config()
        
        # Check that required attributes exist with defaults
        self.assertTrue(hasattr(config, 'base_dir'))
        self.assertTrue(hasattr(config, 'models_dir'))
        self.assertTrue(hasattr(config, 'docs_dir'))
        
    def test_environment_override(self):
        """Test that environment variables override defaults."""
        # Set environment variables
        os.environ["VECTORSTORE_DIR"] = "test_chroma_db"
        os.environ["CHUNK_SIZE"] = "500"
        
        # Create a new config which should pick up these values
        config = Config()
        
        # Check that the values were overridden
        self.assertEqual(config.persist_directory, "test_chroma_db")
        self.assertEqual(config.chunk_size, 500)
        
        # Clean up
        del os.environ["VECTORSTORE_DIR"]
        del os.environ["CHUNK_SIZE"]
    
    def test_model_path(self):
        """Test the get_model_path method."""
        config = Config()
        
        # Test with default (use config setting)
        model_path = config.get_model_path()
        self.assertIn(config.model_name if not config.use_small_model else config.small_model_name, 
                      model_path)
        
        # Test with explicit small model
        small_model_path = config.get_model_path(use_small=True)
        self.assertIn(config.small_model_name, small_model_path)
        
        # Test with explicit large model
        large_model_path = config.get_model_path(use_small=False)
        self.assertIn(config.model_name, large_model_path)
    
    def test_to_dict(self):
        """Test the to_dict method."""
        config = Config()
        
        # Convert to dict
        config_dict = config.to_dict()
        
        # Check it's a dict and contains expected keys
        self.assertIsInstance(config_dict, dict)
        self.assertIn('base_dir', config_dict)
        self.assertIn('models_dir', config_dict)
        self.assertIn('docs_dir', config_dict)

if __name__ == "__main__":
    unittest.main() 