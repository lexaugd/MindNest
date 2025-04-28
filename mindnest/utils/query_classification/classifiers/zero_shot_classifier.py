"""
Zero-shot query classifier.

This module implements a classifier that uses category descriptions rather than examples,
allowing for more flexible and adaptable classification without hardcoded examples.
"""

import time
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from .base import BaseClassifier


class ZeroShotClassifier(BaseClassifier):
    """
    Classifier that uses category descriptions instead of examples.
    
    This classifier calculates similarity between a query and category descriptions,
    making it more flexible and less dependent on hardcoded examples.
    """
    
    def __init__(self, embeddings_model, threshold: float = 0.65):
        """
        Initialize the zero-shot classifier.
        
        Args:
            embeddings_model: The embeddings model to use for encoding queries and descriptions
            threshold: Similarity threshold for classification
        """
        self.embeddings = embeddings_model
        self.threshold = threshold
        self.category_descriptions = {}
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
        """Initialize by computing embeddings for all category descriptions."""
        if self.initialized:
            return
            
        print("Initializing zero-shot classifier...")
        start_time = time.time()
        
        # Load category descriptions
        self.category_descriptions = self._load_category_descriptions()
        
        # Compute embeddings for each category description
        for category, descriptions in self.category_descriptions.items():
            # Compute embeddings for primary description and variations
            description_embeddings = [self.embeddings.embed_query(desc) for desc in descriptions]
            self.category_embeddings[category] = description_embeddings
            
        end_time = time.time()
        print(f"Zero-shot classifier initialized in {end_time - start_time:.2f} seconds")
        self.initialized = True
    
    def _load_category_descriptions(self) -> Dict[str, List[str]]:
        """
        Load descriptions for each category.
        
        Instead of examples, we use detailed descriptions of what each category means.
        
        Returns:
            Dict[str, List[str]]: Dictionary of descriptions by category
        """
        return {
            "CONVERSATION": [
                "General conversation, chitchat, greetings, or casual interaction with the system",
                "Personal questions about the system or casual dialogue not related to documents",
                "Greetings, thanks, acknowledgments, or personal questions about capabilities",
                "Small talk, pleasantries, or questions about opinions and preferences"
            ],
            "DOCUMENT_QUERY": [
                "Questions about specific information contained in documents",
                "Requests for explanations or details about topics covered in documentation",
                "Questions seeking to understand concepts, systems, or components described in docs",
                "Queries asking how something works or what something means in the documentation"
            ],
            "CONCISE_QUERY": [
                "Requests for brief or summarized information from documents",
                "Queries explicitly asking for concise, short, or summarized responses",
                "Requests to explain something briefly or provide a quick overview",
                "Queries using terms like 'summarize', 'brief', 'short', 'tldr', or 'concise'"
            ],
            "DOCUMENT_SEARCH": [
                "Explicit requests to find or locate specific information in documents",
                "Search-like queries asking where to find information on a topic",
                "Queries starting with 'find', 'search', 'locate', or 'where can I find'",
                "Requests to show or list documents related to a specific topic"
            ]
        }
    
    def classify(self, query: str) -> Tuple[str, float]:
        """
        Classify a query based on similarity to category descriptions.
        
        Args:
            query: The query text to classify
            
        Returns:
            Tuple[str, float]: The predicted category and confidence score
        """
        if not self.initialized:
            self.initialize()
            
        start_time = time.time()
        self.stats["calls"] += 1
            
        # Fast path for obvious search queries
        lower_query = query.lower()
        if (lower_query.startswith("find ") or 
            lower_query.startswith("search ") or 
            lower_query.startswith("locate ")):
            self.stats["above_threshold"] += 1
            end_time = time.time()
            self._update_timing_stats(start_time, end_time)
            return "DOCUMENT_SEARCH", 0.95
            
        # Compute query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Find most similar category
        max_score = -1.0
        predicted_category = "DOCUMENT_QUERY"  # Default
        
        for category, embeddings in self.category_embeddings.items():
            # Calculate cosine similarity with each description in this category
            similarities = [
                self._cosine_similarity(query_embedding, desc_embedding) 
                for desc_embedding in embeddings
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
    
    def add_descriptions(self, category: str, queries: List[str], max_descriptions: int = 4) -> int:
        """
        Add new descriptions to a category based on actual user queries.
        
        This allows the classifier to learn from user feedback and actual usage patterns.
        
        Args:
            category: The category to add descriptions for
            queries: List of user queries that belong to this category
            max_descriptions: Maximum number of descriptions to maintain per category
            
        Returns:
            int: Number of descriptions added
        """
        if not self.initialized:
            self.initialize()
            
        if category not in self.category_descriptions:
            self.category_descriptions[category] = []
            
        # Keep track of how many we've added
        added = 0
            
        # Preserve the original descriptions
        current_descriptions = self.category_descriptions[category].copy()
        
        # Add new descriptions up to the max (leaving room for original descriptions)
        for query in queries:
            # Only use queries with a minimum length - shorter queries are often too ambiguous
            if len(query.split()) < 3:
                continue
                
            # Create a better description from the query
            new_description = self.generate_description_from_query(query)
            
            # Check if this is too similar to existing descriptions
            query_embedding = self.embeddings.embed_query(query)
            existing_embeddings = [
                self.embeddings.embed_query(desc) for desc in current_descriptions
            ]
            
            # Skip if too similar to existing descriptions
            if any(self._cosine_similarity(query_embedding, existing) > 0.85 
                  for existing in existing_embeddings):
                continue
                
            # Add the new description
            current_descriptions.append(new_description)
            added += 1
            
            # Stop if we've reached the maximum
            if len(current_descriptions) >= max_descriptions + len(self.category_descriptions[category]):
                break
                
        # Update the descriptions
        self.category_descriptions[category] = current_descriptions
        
        # If we've added new descriptions, we need to recalculate embeddings
        if added > 0:
            # Update embeddings for this category
            description_embeddings = [
                self.embeddings.embed_query(desc) for desc in current_descriptions
            ]
            self.category_embeddings[category] = description_embeddings
            
        return added
        
    def train_from_feedback(self, feedback_entries: List[Dict[str, Any]]) -> int:
        """
        Train the classifier using feedback data.
        
        Args:
            feedback_entries: List of feedback entries with user queries and correct categories
            
        Returns:
            int: Total number of entries used for training
        """
        category_queries = {}
        
        # Group queries by category
        for entry in feedback_entries:
            if not entry.get("is_correct", False):
                continue
                
            category = entry.get("predicted_category")
            query = entry.get("query")
            
            if not category or not query:
                continue
                
            if category not in category_queries:
                category_queries[category] = []
                
            category_queries[category].append(query)
            
        # Add descriptions for each category
        total_added = 0
        for category, queries in category_queries.items():
            added = self.add_descriptions(category, queries)
            total_added += added
            
        return total_added
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the classifier's performance.
        
        Returns:
            Dict[str, Any]: Dictionary of statistics
        """
        stats = self.stats.copy()
        
        # Add category description counts
        stats["category_descriptions"] = {
            category: len(descriptions) for category, descriptions in self.category_descriptions.items()
        }
        
        return stats
    
    def generate_description_from_query(self, query: str) -> str:
        """
        Generate a category description from a user query.
        
        This method takes a user query and generates a description that can
        be used as a category descriptor for future classifications.
        
        Args:
            query: User query to generate a description from
            
        Returns:
            str: Generated description
        """
        # For longer queries, extract key patterns
        words = query.split()
        
        if len(words) <= 3:
            # For very short queries, just use the query directly
            return f"Queries like '{query}'"
            
        # Check for common patterns in queries
        lower_query = query.lower()
        
        # Check for document search patterns
        if any(lower_query.startswith(term) for term in ["find", "search", "locate", "where"]):
            topic = " ".join(words[1:])
            return f"Requests to find or locate information about {topic}"
            
        # Check for concise query patterns
        if any(term in lower_query for term in ["summarize", "brief", "concise", "short", "tldr"]):
            topic = lower_query.replace("summarize", "").replace("brief", "").replace("concise", "").replace("short", "").replace("tldr", "").strip()
            return f"Requests for brief or summarized information about {topic}"
            
        # Check for document query patterns
        if any(lower_query.startswith(term) for term in ["how", "what", "why", "when", "explain", "describe"]):
            topic = " ".join(words[1:])
            return f"Questions seeking information about {topic}"
            
        # Default case for conversation
        return f"Conversational queries similar to '{query}'" 