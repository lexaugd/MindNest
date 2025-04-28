"""
Feedback-based learning system for query classification.

This module implements a system that learns from user feedback to improve
query classification over time. It periodically updates the classifiers
based on the feedback collected.
"""

import time
import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path

from .feedback import FeedbackCollector

# Set up logging
logger = logging.getLogger(__name__)

class FeedbackLearner:
    """
    A system that improves classifier performance based on user feedback.
    
    This class learns from user feedback by periodically updating the
    classifier examples with high-confidence misclassified queries.
    """
    
    def __init__(
        self,
        classifier,
        feedback_collector: Optional[FeedbackCollector] = None,
        confidence_threshold: float = 0.75,
        update_interval: int = 50,
        max_examples_per_category: int = 30
    ):
        """
        Initialize the feedback learner.
        
        Args:
            classifier: The classifier to improve
            feedback_collector: The feedback collector instance to use
            confidence_threshold: Minimum confidence to consider feedback
            update_interval: How many feedback entries before updating
            max_examples_per_category: Maximum number of examples per category
        """
        self.classifier = classifier
        self.feedback_collector = feedback_collector or FeedbackCollector()
        self.confidence_threshold = confidence_threshold
        self.update_interval = update_interval
        self.max_examples_per_category = max_examples_per_category
        self.last_update_count = 0
        self.stats = {
            "updates": 0,
            "new_examples_added": 0,
            "last_update_time": None,
            "examples_by_category": {}
        }
        
    def check_and_update(self) -> bool:
        """
        Check if an update is needed based on feedback and update if necessary.
        
        Returns:
            bool: True if an update was performed, False otherwise
        """
        # Get current feedback stats
        stats = self.feedback_collector.get_stats()
        current_count = stats["total_feedback"]
        
        # Check if we have enough new feedback to trigger an update
        if current_count - self.last_update_count >= self.update_interval:
            self._update_classifier()
            self.last_update_count = current_count
            return True
            
        return False
        
    def _update_classifier(self) -> None:
        """Update the classifier based on feedback data."""
        logger.info("Updating classifier based on feedback data...")
        start_time = time.time()
        
        # Get misclassified entries with high confidence
        misclassified = self.feedback_collector.get_feedback_entries(
            is_correct=False,
            min_confidence=self.confidence_threshold,
            limit=100  # Get the most recent 100 misclassified entries
        )
        
        if not misclassified:
            logger.info("No high-confidence misclassified entries found for learning")
            return
            
        # Check if the classifier supports example updates
        if not hasattr(self.classifier, 'examples') or not hasattr(self.classifier, 'category_embeddings'):
            logger.warning("Classifier does not support example-based learning")
            return
            
        # Track examples added
        examples_added = 0
        category_counts = {}
        
        # Update classifier examples with misclassified queries
        for entry in misclassified:
            correct_category = entry["correct_category"]
            query = entry["query"]
            
            # Initialize category count if needed
            if correct_category not in category_counts:
                category_counts[correct_category] = 0
                
            # Limit the number of examples per category
            if category_counts[correct_category] >= self.max_examples_per_category:
                continue
                
            # Skip if this query is already in the examples
            if correct_category in self.classifier.examples and query in self.classifier.examples[correct_category]:
                continue
                
            # Add to examples
            if correct_category not in self.classifier.examples:
                self.classifier.examples[correct_category] = []
                
            self.classifier.examples[correct_category].append(query)
            
            # Update embeddings
            if hasattr(self.classifier, 'embeddings'):
                query_embedding = self.classifier.embeddings.embed_query(query)
                if correct_category not in self.classifier.category_embeddings:
                    self.classifier.category_embeddings[correct_category] = []
                self.classifier.category_embeddings[correct_category].append(query_embedding)
                
            examples_added += 1
            category_counts[correct_category] = category_counts.get(correct_category, 0) + 1
        
        # Update stats
        self.stats["updates"] += 1
        self.stats["new_examples_added"] += examples_added
        self.stats["last_update_time"] = time.time()
        self.stats["examples_by_category"] = {
            category: len(examples) 
            for category, examples in self.classifier.examples.items()
        }
        
        end_time = time.time()
        logger.info(
            f"Classifier updated with {examples_added} new examples in "
            f"{end_time - start_time:.2f} seconds"
        )
        
        # If using a caching classifier, clear the cache
        if hasattr(self.classifier, '_classify_cached') and hasattr(self.classifier._classify_cached, 'cache_clear'):
            self.classifier._classify_cached.cache_clear()
            logger.info("Classification cache cleared after updating examples")
            
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the learning system.
        
        Returns:
            Dict[str, Any]: Dictionary of statistics
        """
        return self.stats
        
    def force_update(self) -> int:
        """
        Force an immediate update of the classifier based on feedback.
        
        Returns:
            int: Number of new examples added
        """
        old_examples_count = self.stats.get("new_examples_added", 0)
        self._update_classifier()
        return self.stats.get("new_examples_added", 0) - old_examples_count 