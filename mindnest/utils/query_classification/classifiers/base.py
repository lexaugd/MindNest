"""
Base classifier interface for query classification.

This module defines the base interface that all query classifiers must implement.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional


class BaseClassifier(ABC):
    """
    Abstract base class for all query classifiers.
    
    All classifiers must implement the initialize and classify methods.
    """
    
    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the classifier and prepare it for use.
        
        This might involve loading models, compiling patterns, or preparing resources.
        """
        pass
    
    @abstractmethod
    def classify(self, query: str) -> Tuple[str, float]:
        """
        Classify a query into one of the predefined categories.
        
        Args:
            query: The query text to classify
            
        Returns:
            Tuple[str, float]: A tuple containing the category and confidence score.
                The category should be one of:
                - "CONVERSATION" - General conversational queries
                - "DOCUMENT_QUERY" - Queries about document content
                - "CONCISE_QUERY" - Requests for brief responses
                - "DOCUMENT_SEARCH" - Explicit search requests
        """
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about classifier usage and performance.
        
        Returns:
            Dict[str, Any]: A dictionary of statistics
        """
        return {}
    
    def clear_cache(self) -> None:
        """Clear any caches used by the classifier."""
        pass 