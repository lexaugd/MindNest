#!/usr/bin/env python
"""
Script to test the query classification system.
"""

import sys
import os
import time
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import necessary modules
from mindnest.utils.query_classification.classifiers.zero_shot_classifier import ZeroShotClassifier
from mindnest.utils.query_classification.feedback import FeedbackCollector

# Simple embedding model for testing
class SimpleEmbeddings:
    def embed_query(self, text):
        """Simple embedding function that maps text to vectors based on keywords."""
        import numpy as np
        
        # Create a vector with slight randomness for testing
        vec = np.random.normal(0.5, 0.1, size=768)
        
        # Modify specific dimensions based on content
        text = text.lower()
        
        # Conversation indicators
        if any(word in text for word in ["hello", "hi", "hey", "thanks", "bye", "chat", "talk", "name"]):
            vec[0:100] += np.random.normal(0.5, 0.1, size=100)
            
        # Document query indicators  
        if any(word in text for word in ["how", "what", "why", "explain", "does", "work", "mean"]):
            vec[100:200] += np.random.normal(0.5, 0.1, size=100)
            
        # Concise query indicators
        if any(word in text for word in ["brief", "concise", "summarize", "short", "tldr", "quick"]):
            vec[200:300] += np.random.normal(0.5, 0.1, size=100)
            
        # Document search indicators
        if any(word in text for word in ["find", "search", "locate", "where", "show me"]):
            vec[300:400] += np.random.normal(0.5, 0.1, size=100)
            
        # Normalize the vector
        return vec / np.linalg.norm(vec)


def test_classification():
    """Test the classification system with different types of queries."""
    
    # Initialize the embedding model
    embeddings = SimpleEmbeddings()
    
    # Initialize the classifier
    classifier = ZeroShotClassifier(embeddings)
    classifier.initialize()
    
    # Initialize the feedback collector
    feedback_collector = FeedbackCollector()
    
    # Test queries
    test_queries = [
        # Conversation queries
        "Hello there",
        "What's your name?",
        "Thanks for your help",
        "How are you doing today?",
        
        # Document queries
        "How does the authentication system work?",
        "What is the purpose of the wrapper class?",
        "Explain the object hierarchy",
        "Can you tell me about error handling?",
        
        # Concise queries
        "Summarize how authentication works",
        "Give me a brief explanation of the wrapper class",
        "TLDR of the object hierarchy",
        "Explain the API features concisely",
        
        # Document search queries
        "Find documentation about authentication",
        "Search for wrapper class information",
        "Where can I find information about error handling?",
        "Locate examples of component usage"
    ]
    
    # Print header
    print("\nTesting query classification:")
    print("-" * 80)
    print(f"{'Query':<40} | {'Category':<15} | {'Confidence':<10}")
    print("-" * 80)
    
    # Test each query
    for query in test_queries:
        # Classify the query
        category, confidence = classifier.classify(query)
        
        # Print the results
        print(f"{query[:40]:<40} | {category:<15} | {confidence:.4f}")
        
        # Record feedback (assume it's correct for simplicity)
        feedback_collector.add_feedback(
            query=query,
            predicted_category=category,
            confidence=confidence,
            is_correct=True
        )
    
    # Print stats
    print("\nClassification statistics:")
    print("-" * 80)
    stats = classifier.get_stats()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for subkey, subvalue in value.items():
                print(f"  {subkey}: {subvalue}")
        else:
            print(f"{key}: {value}")
    
    # Print feedback stats
    print("\nFeedback statistics:")
    print("-" * 80)
    feedback_stats = feedback_collector.get_stats()
    for key, value in feedback_stats.items():
        print(f"{key}: {value}")
    
    # Now test learning from feedback
    print("\nTraining classifier with feedback...")
    trained = classifier.train_from_feedback(
        feedback_collector.get_feedback_entries(limit=1000)
    )
    print(f"Added {trained} new descriptions from feedback")
    
    # Get updated category descriptions
    print("\nUpdated category descriptions:")
    print("-" * 80)
    for category, descriptions in classifier.category_descriptions.items():
        print(f"{category}:")
        for desc in descriptions:
            print(f"  - {desc}")


if __name__ == "__main__":
    # Create the output directory if it doesn't exist
    os.makedirs("data/feedback", exist_ok=True)
    
    # Run the test
    test_classification() 