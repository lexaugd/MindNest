#!/usr/bin/env python
"""
Server script for running MindNest application.
This file provides a convenient way to run the MindNest server.
"""

import sys
import traceback

def exception_handler(exctype, value, tb):
    """Custom exception handler to print detailed tracebacks"""
    print("=" * 80)
    print("UNCAUGHT EXCEPTION:")
    traceback.print_exception(exctype, value, tb)
    print("=" * 80)
    sys.__excepthook__(exctype, value, tb)

# Install the custom exception handler
sys.excepthook = exception_handler

# Set up debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Import the server runner
from mindnest.app import run_server

if __name__ == "__main__":
    try:
        print("Starting server with debug logging enabled...")
        run_server() 
    except Exception as e:
        print(f"Error starting server: {e}")
        traceback.print_exc() 