"""
Embedding-based query classifier.

This module implements a classifier that uses vector embeddings to classify queries.
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from .base import BaseClassifier


class EmbeddingClassifier(BaseClassifier):
    """
    Classifier that uses vector embeddings to categorize queries.
    
    This classifier calculates the semantic similarity between a query and a set
    of example queries for each category, assigning the query to the category
    with the highest similarity.
    """
    
    def __init__(self, embeddings_model, threshold: float = 0.65):
        """
        Initialize the embedding classifier.
        
        Args:
            embeddings_model: The embeddings model to use for encoding queries
            threshold: Similarity threshold for classification
        """
        self.embeddings = embeddings_model
        self.threshold = threshold
        self.examples = {}
        self.category_embeddings = {}
        self.initialized = False
        self.stats = {
            "calls": 0,
            "above_threshold": 0,
            "below_threshold": 0,
            "avg_time_ms": 0,
            "avg_max_similarity": 0.0
        }
    
    def initialize(self) -> None:
        """Initialize by computing embeddings for all example queries."""
        if self.initialized:
            return
            
        print("Initializing embedding-based query classifier...")
        start_time = time.time()
        
        # Load example queries
        self.examples = self._load_examples()
        
        # Compute embeddings for each category's examples
        for category, queries in self.examples.items():
            # Compute embeddings for each example query
            query_embeddings = [self.embeddings.embed_query(query) for query in queries]
            self.category_embeddings[category] = query_embeddings
            
        end_time = time.time()
        print(f"Embedding classifier initialized in {end_time - start_time:.2f} seconds")
        print(f"Loaded {sum(len(examples) for examples in self.examples.values())} example queries")
        self.initialized = True
    
    def _load_examples(self) -> Dict[str, List[str]]:
        """
        Load example queries for each category.
        
        Returns:
            Dict[str, List[str]]: Dictionary of example queries by category
        """
        return {
            "CONVERSATION": [
                "How are you?",
                "What's your name?",
                "Thanks for the help",
                "Hello there",
                "What can you do?",
                "Tell me about yourself",
                "Nice to meet you",
                "Do you enjoy working?",
                "What is your favorite color?",
                "Goodbye for now"
            ],
            "DOCUMENT_QUERY": [
                "How does the authentication system work?",
                "What is the purpose of the wrapper class?",
                "Explain the object hierarchy",
                "What features does the API support?",
                "How do I use the configuration module?",
                "What are the main components of the system?",
                "How is data stored in the application?",
                "Explain how error handling works",
                "What design patterns are used in the codebase?",
                "How does the logging system work?"
            ],
            "CONCISE_QUERY": [
                "Summarize how authentication works",
                "Give me a short explanation of the wrapper class",
                "Brief overview of the object hierarchy",
                "Explain the API features concisely",
                "Describe the configuration module briefly",
                "Short summary of the system architecture",
                "Key points about error handling",
                "One paragraph about logging system",
                "Concise explanation of design patterns used",
                "TL;DR of the data storage approach"
            ],
            "DOCUMENT_SEARCH": [
                "Find documentation about authentication",
                "Search for wrapper class information",
                "Find all references to object hierarchy",
                "Locate API features documentation",
                "Find configuration examples",
                "Show me documents about error handling",
                "Find information about logging",
                "Search for design pattern examples",
                "Where can I find data storage documentation",
                "Locate examples of component usage"
            ]
        }
    
    def classify(self, query: str) -> Tuple[str, float]:
        """
        Classify a query based on similarity to example queries.
        
        Args:
            query: The query text to classify
            
        Returns:
            Tuple[str, float]: The predicted category and confidence score
        """
        if not self.initialized:
            self.initialize()
            
        start_time = time.time()
        self.stats["calls"] += 1
            
        # Compute query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Find most similar category
        max_score = -1.0
        predicted_category = "DOCUMENT_QUERY"  # Default
        
        for category, embeddings in self.category_embeddings.items():
            # Calculate cosine similarity with each example in this category
            similarities = [
                self._cosine_similarity(query_embedding, ref_embedding) 
                for ref_embedding in embeddings
            ]
            
            # Get the maximum similarity score for this category
            category_score = max(similarities)
            
            if category_score > max_score:
                max_score = category_score
                predicted_category = category
        
        # Update stats
        self.stats["avg_max_similarity"] = (
            (self.stats["avg_max_similarity"] * (self.stats["calls"] - 1)) + max_score
        ) / self.stats["calls"]
        
        # Check if the score is below the threshold (uncertain classification)
        if max_score < self.threshold:
            self.stats["below_threshold"] += 1
            end_time = time.time()
            self._update_timing_stats(start_time, end_time)
            # If unsure, default to document query to be safe
            return "DOCUMENT_QUERY", max_score
           
        self.stats["above_threshold"] += 1
        end_time = time.time()
        self._update_timing_stats(start_time, end_time)
        return predicted_category, max_score
    
    def _cosine_similarity(self, vec_a, vec_b) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec_a: First vector
            vec_b: Second vector
            
        Returns:
            float: Cosine similarity between the vectors
        """
        # Check if inputs are lists or numpy arrays
        if isinstance(vec_a, list):
            vec_a = np.array(vec_a)
        if isinstance(vec_b, list):
            vec_b = np.array(vec_b)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return dot_product / (norm_a * norm_b)
    
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
        return self.stats 