"""
Ensemble-based query classifier.

This module implements a classifier that combines multiple classification approaches
using a weighted voting system.
"""

import time
from typing import Dict, List, Tuple, Any, Optional

from .base import BaseClassifier
from .pattern_classifier import PatternClassifier
from .embedding_classifier import EmbeddingClassifier


class OptimizedEnsembleClassifier(BaseClassifier):
    """
    Classifier that combines multiple classifiers using weighted voting.
    
    This classifier delegates classification to specialized classifiers and
    combines their results using a weighted voting system. It provides better
    accuracy than any individual classifier.
    """
    
    def __init__(self, embeddings_model):
        """
        Initialize the ensemble classifier.
        
        Args:
            embeddings_model: The embeddings model to use for the embedding classifier
        """
        self.classifiers = {
            "pattern": PatternClassifier(extended_patterns=True),
            "embedding": EmbeddingClassifier(embeddings_model, threshold=0.65)
        }
        # Default weights based on reliability
        self.weights = {"pattern": 0.6, "embedding": 0.4}
        self.initialized = False
        self.stats = {
            "calls": 0,
            "pattern_used": 0,
            "embedding_used": 0,
            "consensus": 0,
            "disagreement": 0,
            "avg_time_ms": 0
        }
    
    def initialize(self) -> None:
        """Initialize all classifiers in the ensemble."""
        if self.initialized:
            return
            
        print("Initializing ensemble classifier...")
        start_time = time.time()
        
        # Initialize all classifiers
        for name, classifier in self.classifiers.items():
            classifier.initialize()
            
        end_time = time.time()
        print(f"Ensemble classifier initialized in {end_time - start_time:.2f} seconds")
        self.initialized = True
            
    def classify(self, query: str, fallback: str = "DOCUMENT_QUERY") -> Tuple[str, float]:
        """
        Classify using weighted ensemble approach.
        
        Args:
            query: The query text to classify
            fallback: The category to use if no result can be determined
            
        Returns:
            Tuple[str, float]: Category and confidence score
        """
        if not self.initialized:
            self.initialize()
            
        start_time = time.time()
        self.stats["calls"] += 1
            
        # Fast path: check for obvious patterns first
        if query.lower().startswith("find "):
            self.stats["pattern_used"] += 1
            end_time = time.time()
            self._update_timing_stats(start_time, end_time)
            return "DOCUMENT_SEARCH", 1.0
            
        results = {}
        confidences = {}
        classifier_results = {}
        
        # Get classifications from all classifiers
        for name, classifier in self.classifiers.items():
            category, confidence = classifier.classify(query)
            classifier_results[name] = (category, confidence)
            
            # Weight the confidence
            weighted_confidence = confidence * self.weights[name]
            
            # Add to results
            if category not in results:
                results[category] = 0
                confidences[category] = []
                
            results[category] += weighted_confidence
            confidences[category].append(confidence)
        
        # No results
        if not results:
            end_time = time.time()
            self._update_timing_stats(start_time, end_time)
            return fallback, 0.0
            
        # Find highest scoring category
        best_category = max(results.items(), key=lambda x: x[1])
        category = best_category[0]
        
        # Update stats based on classifier agreement
        if all(result[0] == category for result in classifier_results.values()):
            self.stats["consensus"] += 1
        else:
            self.stats["disagreement"] += 1
            
        if classifier_results["pattern"][0] == category:
            self.stats["pattern_used"] += 1
            
        if classifier_results["embedding"][0] == category:
            self.stats["embedding_used"] += 1
        
        # Calculate a blended confidence score
        avg_confidence = sum(confidences[category]) / len(confidences[category])
        weighted_sum = best_category[1]
        weight_sum = sum(self.weights.values())
        
        # Blend confidence metric (balance of weighted sum and average confidence)
        final_confidence = 0.7 * (weighted_sum / weight_sum) + 0.3 * avg_confidence
        
        end_time = time.time()
        self._update_timing_stats(start_time, end_time)
        return category, final_confidence
    
    def _update_timing_stats(self, start_time: float, end_time: float) -> None:
        """Update timing statistics for performance monitoring."""
        elapsed_ms = (end_time - start_time) * 1000
        self.stats["avg_time_ms"] = (
            (self.stats["avg_time_ms"] * (self.stats["calls"] - 1)) + elapsed_ms
        ) / self.stats["calls"]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the classifier's performance.
        
        Returns:
            Dict[str, Any]: Dictionary of statistics
        """
        # Combine stats from this classifier and component classifiers
        stats = self.stats.copy()
        
        # Add component classifier stats
        for name, classifier in self.classifiers.items():
            classifier_stats = classifier.get_stats()
            for key, value in classifier_stats.items():
                stats[f"{name}_{key}"] = value
                
        return stats 