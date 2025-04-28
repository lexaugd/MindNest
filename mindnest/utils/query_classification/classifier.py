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
    
    def __init__(self, embeddings_model, classifier_type: str = "embeddings", enable_feedback_learning: bool = False):
        """
        Initialize the query classifier.
        
        Args:
            embeddings_model: The embeddings model to use
            classifier_type: The type of classifier to use ("embeddings", "neural", "hybrid", or "zero-shot")
            enable_feedback_learning: Whether to enable automatic learning from feedback
        """
        self.classifier_type = classifier_type
        self.embeddings_model = embeddings_model
        self.classifier = None
        self._last_confidence = 0.0
        self.feedback_learner = None
        self.enable_feedback_learning = enable_feedback_learning
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
        
        # Initialize feedback learner if enabled
        if self.enable_feedback_learning:
            self._initialize_feedback_learning()
            
    def _initialize_feedback_learning(self):
        """Initialize the feedback learning system."""
        try:
            from .feedback_learning import FeedbackLearner
            self.feedback_learner = FeedbackLearner(self.classifier)
            logger.info("Feedback learning system initialized")
        except ImportError as e:
            logger.warning(f"Could not initialize feedback learning: {e}")
            self.enable_feedback_learning = False
            
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
        # Check if we should update the classifier based on feedback
        if self.enable_feedback_learning and self.feedback_learner:
            try:
                if self.feedback_learner.check_and_update():
                    logger.info("Classifier updated based on feedback data")
                    # Clear cache when classifier examples are updated
                    self._classify_cached.cache_clear()
            except Exception as e:
                logger.error(f"Error updating classifier from feedback: {e}")
        
        start_time = time.time()
        
        # Check if query is in cache
        cached_calls_before = self._classify_cached.cache_info().hits
        
        # Perform classification using the neural classifier
        # No hardcoded patterns here - let the classifier do its job
        category, score = self._classify_cached(query.lower())
        
        # Store the confidence score for external access
        self._last_confidence = score
        
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
            "cache_hit_ratio": cache_info.hits / max(1, (cache_info.hits + cache_info.misses)),
            "last_confidence": self._last_confidence
        }
        
        # Add feedback learning stats if available
        if self.enable_feedback_learning and self.feedback_learner:
            stats["feedback_learning"] = self.feedback_learner.get_stats()
        
        return stats
        
    def clear_cache(self):
        """Clear the classification cache."""
        self._classify_cached.cache_clear()
        print("Query classification cache cleared")
        
    def update_from_feedback(self, force: bool = True) -> bool:
        """
        Manually update the classifier based on collected feedback.
        
        Args:
            force: Whether to force an update even if the update interval hasn't been reached
            
        Returns:
            bool: True if an update was performed, False otherwise
        """
        if not self.enable_feedback_learning or not self.feedback_learner:
            logger.warning("Feedback learning is not enabled")
            return False
            
        if force:
            examples_added = self.feedback_learner.force_update()
            logger.info(f"Forced update added {examples_added} new examples")
            return examples_added > 0
        else:
            return self.feedback_learner.check_and_update()
            
    def set_feedback_learning(self, enabled: bool) -> None:
        """
        Enable or disable feedback learning.
        
        Args:
            enabled: Whether to enable feedback learning
        """
        if enabled and not self.enable_feedback_learning:
            self.enable_feedback_learning = True
            if self.feedback_learner is None:
                self._initialize_feedback_learning()
            logger.info("Feedback learning enabled")
        elif not enabled and self.enable_feedback_learning:
            self.enable_feedback_learning = False
            logger.info("Feedback learning disabled") 