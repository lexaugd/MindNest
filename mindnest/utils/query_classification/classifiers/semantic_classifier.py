"""
Semantic query classifier that uses embeddings to classify queries.
Provides more accurate classification than regex-based approaches.
"""

import logging
import numpy as np
import os
import json
import time
from typing import Dict, Any, List, Tuple, Optional
from functools import lru_cache
from pathlib import Path

# Setup logging
logger = logging.getLogger(__name__)

class SemanticClassifier:
    """
    Classifies queries using semantic similarity to example queries.
    Uses embeddings to compare queries to examples of different categories.
    Features adaptive learning from feedback rather than hardcoded patterns.
    """
    
    def __init__(self, embedding_model, confidence_threshold: float = 0.75):
        """
        Initialize the semantic classifier.
        
        Args:
            embedding_model: Model to use for creating embeddings
            confidence_threshold: Minimum confidence threshold for classification
        """
        self.embedding_model = embedding_model
        self.confidence_threshold = confidence_threshold
        
        # Initialize examples with minimal seed data
        # This will be updated through feedback learning
        self.examples = {
            "CONVERSATION": [],
            "DOCUMENT_QUERY": [],
            "CONCISE_QUERY": [],
            "DOCUMENT_SEARCH": []
        }
        
        # Category embeddings cache
        self._category_embeddings = {}
        
        # Track feedback for learning
        self.feedback_data = []
        self.max_feedback_items = 500
        
        # Path for saving/loading trained data
        self.data_path = os.environ.get(
            "CLASSIFIER_DATA_PATH", 
            os.path.join(os.path.dirname(__file__), "../data")
        )
        Path(self.data_path).mkdir(parents=True, exist_ok=True)
        
        # Path for specific classifier data
        self.examples_path = os.path.join(self.data_path, "semantic_classifier_examples.json")
        
    def initialize(self):
        """Initialize the classifier by loading examples and pre-computing embeddings."""
        logger.info("Initializing semantic classifier...")
        
        # Load saved examples if available
        self._load_examples()
        
        # Add default examples only if we have no data at all for a category
        self._add_default_examples_if_needed()
        
        # Pre-compute category embeddings for each example
        self._update_category_embeddings()
        
        logger.info("Semantic classifier initialization complete")
    
    def _add_default_examples_if_needed(self):
        """Add minimal default examples only if we have no examples for a category."""
        # These are only used when no learned examples exist
        # The system will primarily learn from user feedback
        default_examples = {
            "CONVERSATION": ["hello", "how are you", "thanks"],
            "DOCUMENT_QUERY": ["how does this work", "explain the system"],
            "CONCISE_QUERY": ["summarize this", "brief explanation"],
            "DOCUMENT_SEARCH": ["find documentation", "search for"]
        }
        
        # Only add defaults where we have no examples
        for category, examples in default_examples.items():
            if not self.examples.get(category):
                logger.info(f"Adding default examples for category: {category}")
                self.examples[category] = examples
    
    def _load_examples(self):
        """Load saved examples from disk if available."""
        try:
            if os.path.exists(self.examples_path):
                with open(self.examples_path, 'r') as f:
                    loaded_examples = json.load(f)
                    logger.info(f"Loaded examples from {self.examples_path}")
                    
                    # Update our examples with loaded data
                    for category, examples in loaded_examples.items():
                        if category in self.examples:
                            self.examples[category] = examples
                        
                    # Log the number of examples loaded
                    total_examples = sum(len(examples) for examples in self.examples.values())
                    logger.info(f"Loaded {total_examples} examples across {len(self.examples)} categories")
                    
                    return True
            else:
                logger.info(f"No saved examples found at {self.examples_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading examples: {e}")
            return False
    
    def _save_examples(self):
        """Save examples to disk for persistence."""
        try:
            with open(self.examples_path, 'w') as f:
                json.dump(self.examples, f)
            logger.info(f"Saved examples to {self.examples_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving examples: {e}")
            return False
    
    def _update_category_embeddings(self):
        """Update category embeddings based on current examples."""
        for category, examples in self.examples.items():
            if not examples:
                logger.warning(f"No examples for category: {category}")
                continue
                
            try:
                # Get embeddings for all examples in this category
                embeddings = self._get_embeddings(examples)
                # Store the mean embedding as the category centroid
                self._category_embeddings[category] = np.mean(embeddings, axis=0)
                logger.info(f"Computed embeddings for category: {category} with {len(examples)} examples")
            except Exception as e:
                logger.error(f"Error computing embeddings for category {category}: {e}")
    
    def add_examples(self, category: str, examples: List[str]):
        """
        Add new examples for a category.
        
        Args:
            category: The category to add examples for
            examples: List of example queries
        """
        if category not in self.examples:
            self.examples[category] = []
            
        # Add new examples, avoiding duplicates
        for example in examples:
            if example not in self.examples[category]:
                self.examples[category].append(example)
        
        # Save updated examples
        self._save_examples()
        
        # Recompute category embeddings
        try:
            embeddings = self._get_embeddings(self.examples[category])
            self._category_embeddings[category] = np.mean(embeddings, axis=0)
            logger.info(f"Updated embeddings for category: {category} with {len(examples)} new examples")
        except Exception as e:
            logger.error(f"Error updating embeddings for category {category}: {e}")
    
    def add_feedback(self, query: str, predicted_category: str, correct_category: str):
        """
        Add feedback about a classification to improve future results.
        
        Args:
            query: The query that was classified
            predicted_category: The category that was predicted
            correct_category: The correct category from feedback
        """
        if predicted_category == correct_category:
            # If prediction was correct, add to examples to reinforce
            self.add_examples(correct_category, [query])
        else:
            # If prediction was wrong, add to correct category
            self.add_examples(correct_category, [query])
            
            # Store feedback for analysis
            self.feedback_data.append({
                "query": query,
                "predicted": predicted_category,
                "correct": correct_category,
                "timestamp": time.time()
            })
            
            # Trim feedback data if needed
            if len(self.feedback_data) > self.max_feedback_items:
                self.feedback_data = self.feedback_data[-self.max_feedback_items:]
                
        # Return success indicator
        return True
    
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Array of embeddings
        """
        if not texts:
            return np.array([])
            
        # Handle embedding model specific interface
        if hasattr(self.embedding_model, "embed_documents"):
            # LangChain style embedding model
            return np.array(self.embedding_model.embed_documents(texts))
        elif hasattr(self.embedding_model, "encode"):
            # Sentence-transformers style
            return np.array(self.embedding_model.encode(texts))
        else:
            # Try generic call
            return np.array(self.embedding_model(texts))
    
    @lru_cache(maxsize=1024)
    def _get_query_embedding(self, query: str) -> np.ndarray:
        """
        Get embedding for a single query with caching.
        
        Args:
            query: Query text
            
        Returns:
            Query embedding vector
        """
        # Handle embedding model specific interface
        if hasattr(self.embedding_model, "embed_query"):
            # LangChain style embedding model for single query
            return np.array(self.embedding_model.embed_query(query))
        elif hasattr(self.embedding_model, "encode"):
            # Sentence-transformers style
            return np.array(self.embedding_model.encode(query))
        else:
            # Try generic call
            embeddings = self._get_embeddings([query])
            return embeddings[0]
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def classify(self, query: str) -> Tuple[str, float]:
        """
        Classify a query into one of the defined categories.
        
        Args:
            query: The query to classify
            
        Returns:
            Tuple of (category, confidence_score)
        """
        try:
            # Special handling for empty categories
            if not any(self._category_embeddings.values()):
                # No embeddings available yet, use fallback based on query length
                if len(query.split()) <= 3:
                    return "CONVERSATION", 0.6
                else:
                    return "DOCUMENT_QUERY", 0.6
            
            # Get query embedding
            query_embedding = self._get_query_embedding(query)
            
            # Calculate similarity to each category
            similarities = {}
            for category, category_embedding in self._category_embeddings.items():
                if category_embedding is not None and len(category_embedding) > 0:
                    similarity = self._cosine_similarity(query_embedding, category_embedding)
                    similarities[category] = similarity
            
            # If no similarities calculated, fall back to default
            if not similarities:
                return "DOCUMENT_QUERY", 0.5
            
            # Find the most similar category
            best_category = max(similarities, key=similarities.get)
            confidence = similarities[best_category]
            
            # Apply confidence threshold
            if confidence < self.confidence_threshold:
                # For low confidence, use query length as a heuristic
                # This is a temporary fallback until enough examples are learned
                if len(query.split()) <= 3:
                    logger.debug(f"Low confidence ({confidence:.4f}), falling back to CONVERSATION")
                    return "CONVERSATION", 0.6
                else:
                    logger.debug(f"Low confidence ({confidence:.4f}), falling back to DOCUMENT_QUERY")
                    return "DOCUMENT_QUERY", 0.6
            
            logger.debug(f"Classified query as {best_category} with confidence {confidence:.4f}")
            return best_category, float(confidence)
            
        except Exception as e:
            logger.error(f"Error classifying query: {e}")
            # Default to DOCUMENT_QUERY on error for longer queries
            if len(query.split()) > 3:
                return "DOCUMENT_QUERY", 0.5
            else:
                return "CONVERSATION", 0.5
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the classifier.
        
        Returns:
            Dictionary of classifier statistics
        """
        return {
            "example_counts": {category: len(examples) for category, examples in self.examples.items()},
            "confidence_threshold": self.confidence_threshold,
            "cache_info": self._get_query_embedding.cache_info()._asdict(),
            "feedback_count": len(self.feedback_data)
        }
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self._get_query_embedding.cache_clear()
        logger.info("Semantic classifier embedding cache cleared") 