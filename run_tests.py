#!/usr/bin/env python
"""
Test runner for MindNest.
Discovers and runs all tests in the project.
"""

import unittest
import sys
import os
from pathlib import Path

def run_tests():
    """Discover and run all tests in the project."""
    # Make sure we're in the correct directory
    os.chdir(Path(__file__).parent)
    
    # Discover tests
    loader = unittest.TestLoader()
    suite = loader.discover('tests')
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return exit code based on test results
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    sys.exit(run_tests()) 