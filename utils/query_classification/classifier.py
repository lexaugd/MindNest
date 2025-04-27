"""
AI-based query classifier for determining query intent.
This is used to distinguish between conversation queries, document queries, etc.
"""

import time
import logging
from functools import lru_cache
from typing import Tuple, Dict, Any, Optional

from .model_loader import create_classifier

# Set up logging
logger = logging.getLogger(__name__)

class QueryClassifier:
    """
    AI-powered query classifier that distinguishes between different types of queries.
    Uses embedding similarity or neural models for classification.
    """
    
    def __init__(self, embeddings_model, classifier_type: str = "embeddings"):
        """
        Initialize the query classifier.
        
        Args:
            embeddings_model: The embeddings model to use
            classifier_type: The type of classifier to use ("embeddings", "neural", or "hybrid")
        """
        self.classifier_type = classifier_type
        self.embeddings_model = embeddings_model
        self.classifier = None
        self.stats = {
            "calls": 0,
            "conversation_count": 0,
            "document_query_count": 0,
            "concise_query_count": 0,
            "document_search_count": 0,
            "average_time_ms": 0,
            "cache_hits": 0
        }
    
    def initialize(self):
        """Initialize the classifier based on the selected type."""
        if self.classifier is not None:
            return
            
        # Create the appropriate classifier
        self.classifier = create_classifier(self.classifier_type, self.embeddings_model)
        
        # Initialize the classifier
        self.classifier.initialize()
            
    @lru_cache(maxsize=1024)
    def _classify_cached(self, query: str) -> Tuple[str, float]:
        """
        Cached version of the classification function.
        
        Args:
            query: The query to classify
            
        Returns:
            Tuple[str, float]: Classification category and confidence score
        """
        if self.classifier is None:
            self.initialize()
            
        return self.classifier.classify(query)
    
    def classify(self, query: str) -> Tuple[str, str]:
        """
        Classify a query to determine its intent.
        
        Args:
            query: The query to classify
            
        Returns:
            Tuple[str, str]: Classification result and processed query
                First value is one of: "CONVERSATION", "DOCUMENT_QUERY", 
                "CONCISE_QUERY", or "DOCUMENT_SEARCH"
        """
        # Quick check for search queries (no AI needed for these obvious cases)
        if query.lower().startswith("find "):
            self.stats["document_search_count"] += 1
            return "DOCUMENT_SEARCH", query[5:]
            
        start_time = time.time()
        
        # Check if query is in cache
        cached_calls_before = self._classify_cached.cache_info().hits
        
        # Perform classification
        category, score = self._classify_cached(query.lower())
        
        # Update stats
        self.stats["calls"] += 1
        if self._classify_cached.cache_info().hits > cached_calls_before:
            self.stats["cache_hits"] += 1
            
        # Update category counts
        if category == "CONVERSATION":
            self.stats["conversation_count"] += 1
        elif category == "DOCUMENT_QUERY":
            self.stats["document_query_count"] += 1
        elif category == "CONCISE_QUERY":
            self.stats["concise_query_count"] += 1
        elif category == "DOCUMENT_SEARCH":
            self.stats["document_search_count"] += 1
            
        # Update timing stats
        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        self.stats["average_time_ms"] = (
            (self.stats["average_time_ms"] * (self.stats["calls"] - 1)) + elapsed_ms
        ) / self.stats["calls"]
        
        logger.debug(
            f"Query classified as {category} with score {float(score):.4f} in {elapsed_ms:.2f}ms"
        )
        
        return category, query
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the classifier usage.
        
        Returns:
            Dict[str, Any]: Classification statistics
        """
        cache_info = self._classify_cached.cache_info()
        
        stats = {
            **self.stats,
            "cache_size": cache_info.currsize,
            "cache_max_size": cache_info.maxsize,
            "cache_hit_ratio": cache_info.hits / max(1, (cache_info.hits + cache_info.misses))
        }
        
        return stats
        
    def clear_cache(self):
        """Clear the classification cache."""
        self._classify_cached.cache_clear()
        print("Query classification cache cleared") 