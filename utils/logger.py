"""
Logging configuration for MindNest.
Provides standardized logging throughout the application.
"""

import os
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

class Logger:
    """Centralized logging configuration for MindNest."""
    
    def __init__(self, name="mindnest", log_level=None, log_file=None):
        """
        Initialize logger with name and optional level/file.
        
        Args:
            name: Logger name
            log_level: Logging level (default: from env or INFO)
            log_file: Path to log file (default: from env or logs/mindnest.log)
        """
        self.name = name
        
        # Get log level from environment or use default
        if log_level is None:
            env_level = os.environ.get("LOG_LEVEL", "INFO").upper()
            self.log_level = getattr(logging, env_level, logging.INFO)
        else:
            self.log_level = log_level
            
        # Get log file path from environment or use default
        if log_file is None:
            self.log_file = os.environ.get("LOG_FILE", "logs/mindnest.log")
        else:
            self.log_file = log_file
            
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(self.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        # Configure root logger
        self._configure_root_logger()
        
        # Create and configure module logger
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(self.log_level)
        
    def _configure_root_logger(self):
        """Configure the root logger with console and file handlers."""
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        
        # Remove existing handlers to avoid duplicates
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            
        # Create formatters
        console_format = logging.Formatter('%(levelname)s: %(message)s')
        file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_format)
        console_handler.setLevel(self.log_level)
        root_logger.addHandler(console_handler)
        
        # Create file handler
        try:
            file_handler = RotatingFileHandler(
                self.log_file,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
            file_handler.setFormatter(file_format)
            file_handler.setLevel(self.log_level)
            root_logger.addHandler(file_handler)
        except (PermissionError, FileNotFoundError) as e:
            console_handler.setLevel(logging.WARNING)
            root_logger.warning(f"Could not create log file at {self.log_file}: {e}")
            root_logger.warning("Continuing with console logging only")
    
    def get_logger(self):
        """Get the configured logger instance."""
        return self.logger


# Create singleton instance with default configuration
app_logger = Logger().get_logger()

# For direct imports
def get_logger(name):
    """Get a named logger that inherits the main configuration."""
    return logging.getLogger(name)


# For testing
if __name__ == "__main__":
    app_logger.debug("This is a debug message")
    app_logger.info("This is an info message")
    app_logger.warning("This is a warning message")
    app_logger.error("This is an error message")
    app_logger.critical("This is a critical message")
    
    # Test module logger
    test_logger = get_logger("test")
    test_logger.info("Test module logger") 