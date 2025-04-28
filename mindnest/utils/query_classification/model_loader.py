"""
Model loaders for query classification.
Handles loading and initialization of various classification models.
"""

import os
import time
import numpy as np
from functools import lru_cache
from typing import List, Dict, Tuple, Any, Optional, Union

# Try to import util from sentence_transformers, if it fails, create a simple fallback
try:
    from sentence_transformers import util
except ImportError:
    # Simple fallback for cosine similarity
    print("Warning: sentence_transformers.util not available, using fallback cosine similarity")
    class UtilFallback:
        @staticmethod
        def cos_sim(a, b):
            """Compute cosine similarity between two vectors."""
            if isinstance(a, list):
                a = np.array(a)
            if isinstance(b, list):
                b = np.array(b)
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    util = UtilFallback()

# Import classifiers directly - no need for hardcoded examples
from .classifiers.pattern_classifier import PatternClassifier
from .classifiers.embedding_classifier import EmbeddingClassifier
from .classifiers.ensemble_classifier import OptimizedEnsembleClassifier
from .classifiers.zero_shot_classifier import ZeroShotClassifier

def create_classifier(classifier_type: str, embeddings_model):
    """
    Create a classifier based on the specified type.
    
    Args:
        classifier_type: Type of classifier to create
            ('pattern', 'embedding', 'ensemble', 'hybrid', 'zero-shot')
        embeddings_model: Model to use for embeddings (required for 
            embedding-based classifiers)
    
    Returns:
        The created classifier instance
    """
    if classifier_type == "pattern":
        return PatternClassifier(extended_patterns=True)
    elif classifier_type == "embedding":
        return EmbeddingClassifier(embeddings_model, threshold=0.65)
    elif classifier_type == "ensemble" or classifier_type == "hybrid":
        return OptimizedEnsembleClassifier(embeddings_model)
    elif classifier_type == "zero-shot":
        return ZeroShotClassifier(embeddings_model, threshold=0.65)
    else:
        print(f"Unknown classifier type: {classifier_type}, using embedding classifier")
        return EmbeddingClassifier(embeddings_model, threshold=0.65)


class TinyBERTClassifier:
    """Classification using a small BERT model specifically for query classification."""
    
    def __init__(self, model_name: str = "prajjwal1/bert-tiny", threshold: float = 0.6):
        """
        Initialize the TinyBERT classifier.
        
        Args:
            model_name: The name of the model to use
            threshold: Confidence threshold for classification
        """
        self.model_name = model_name
        self.threshold = threshold
        self.model = None
        self.initialized = False
        
    def initialize(self):
        """Initialize the TinyBERT model."""
        if self.initialized:
            return
            
        print(f"Initializing TinyBERT classifier with model {self.model_name}...")
        start_time = time.time()
        
        try:
            from transformers import pipeline
            
            # Load classification pipeline
            self.model = pipeline(
                "text-classification",
                model=self.model_name,
                return_all_scores=True
            )
            
            end_time = time.time()
            print(f"TinyBERT classifier initialized in {end_time - start_time:.2f} seconds")
            self.initialized = True
            
        except Exception as e:
            print(f"Error initializing TinyBERT classifier: {e}")
            print("Falling back to embedding-based classification.")
            self.initialized = False
    
    def classify(self, query: str) -> Tuple[str, float]:
        """
        Classify a query using the TinyBERT model.
        
        Args:
            query: The query to classify
            
        Returns:
            Tuple[str, float]: The predicted category and confidence score
        """
        if not self.initialized:
            self.initialize()
            
        if not self.initialized or self.model is None:
            # Return a default if initialization failed
            return "DOCUMENT_QUERY", 0.0
            
        try:
            # Get classification scores
            result = self.model(query)
            
            # Extract the highest scoring category
            best_match = max(result[0], key=lambda x: x['score'])
            category = best_match['label']
            score = best_match['score']
            
            # Check if the score is below the threshold (uncertain classification)
            if score < self.threshold:
                # If unsure, default to document query to be safe
                return "DOCUMENT_QUERY", score
                
            return category, score
            
        except Exception as e:
            print(f"Error during TinyBERT classification: {e}")
            # Default to document query on error
            return "DOCUMENT_QUERY", 0.0


class HybridClassifier:
    """
    Hybrid approach that combines embedding-based classification with TinyBERT.
    Uses embedding for faster classification of obvious cases, 
    and falls back to TinyBERT for ambiguous cases.
    """
    
    def __init__(self, embeddings_model, embedding_threshold: float = 0.7):
        """
        Initialize the hybrid classifier.
        
        Args:
            embeddings_model: The embeddings model to use
            embedding_threshold: Threshold for the embedding classifier
        """
        self.embedding_classifier = EmbeddingClassifier(
            embeddings_model, 
            threshold=embedding_threshold
        )
        self.tinybert_classifier = TinyBERTClassifier()
        self.initialized = False
        
    def initialize(self):
        """Initialize both classifiers."""
        if self.initialized:
            return
            
        print("Initializing hybrid query classifier...")
        
        # Initialize embedding classifier (this should always succeed)
        self.embedding_classifier.initialize()
        
        # Try to initialize TinyBERT (this might fail)
        try:
            self.tinybert_classifier.initialize()
        except Exception as e:
            print(f"Warning: Failed to initialize TinyBERT classifier: {e}")
            print("Hybrid classifier will rely only on embeddings.")
            
        self.initialized = True
    
    def classify(self, query: str) -> Tuple[str, float]:
        """
        Classify a query using the hybrid approach.
        
        Args:
            query: The query to classify
            
        Returns:
            Tuple[str, float]: The predicted category and confidence score
        """
        if not self.initialized:
            self.initialize()
            
        # First try embedding classification
        category, score = self.embedding_classifier.classify(query)
        
        # If score is high enough, trust the embedding classification
        if score >= self.embedding_classifier.threshold:
            return category, score
            
        # For ambiguous cases, try TinyBERT if available
        if self.tinybert_classifier.initialized:
            tiny_category, tiny_score = self.tinybert_classifier.classify(query)
            
            # If TinyBERT is confident, use its prediction
            if tiny_score >= self.tinybert_classifier.threshold:
                return tiny_category, tiny_score
                
        # Fall back to embedding prediction if TinyBERT failed or was not confident
        return category, score
 