"""
Query classification feedback collection system.

This module implements a system for collecting and storing feedback on
query classification results to improve future classifications.
"""

import json
import os
import time
from typing import Dict, List, Optional, Any
from datetime import datetime


class FeedbackCollector:
    """
    Collects and stores feedback on query classification results.
    
    This class provides methods for recording feedback on classification
    accuracy, storing the feedback in a persistent format, and retrieving
    feedback data for analysis.
    """
    
    def __init__(self, 
                 storage_dir: str = "data/feedback", 
                 max_entries: int = 10000):
        """
        Initialize the feedback collector.
        
        Args:
            storage_dir: Directory to store feedback data
            max_entries: Maximum number of feedback entries to store in memory
        """
        self.storage_dir = storage_dir
        self.max_entries = max_entries
        self.feedback_data = []
        self.stats = {
            "total_feedback": 0,
            "correct_classifications": 0,
            "incorrect_classifications": 0,
            "avg_confidence_correct": 0,
            "avg_confidence_incorrect": 0
        }
        
        # Ensure storage directory exists
        os.makedirs(storage_dir, exist_ok=True)
        
        # Try to load existing feedback data
        self._load_feedback()
    
    def _load_feedback(self) -> None:
        """Load existing feedback data from storage."""
        feedback_file = os.path.join(self.storage_dir, "classification_feedback.json")
        if os.path.exists(feedback_file):
            try:
                with open(feedback_file, "r") as f:
                    data = json.load(f)
                    # Limit entries to max_entries, keeping most recent
                    self.feedback_data = data[-self.max_entries:]
                    # Recalculate stats
                    self._recalculate_stats()
                    print(f"Loaded {len(self.feedback_data)} feedback entries")
            except Exception as e:
                print(f"Error loading feedback data: {e}")
                self.feedback_data = []
    
    def _save_feedback(self) -> None:
        """Save current feedback data to storage."""
        feedback_file = os.path.join(self.storage_dir, "classification_feedback.json")
        try:
            with open(feedback_file, "w") as f:
                json.dump(self.feedback_data, f, indent=2)
        except Exception as e:
            print(f"Error saving feedback data: {e}")
    
    def add_feedback(self, 
                     query: str, 
                     predicted_category: str, 
                     confidence: float,
                     correct_category: Optional[str] = None,
                     is_correct: Optional[bool] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add feedback on a classification result.
        
        Args:
            query: The original query text
            predicted_category: The category predicted by the classifier
            confidence: The confidence score of the prediction
            correct_category: The correct category (if known)
            is_correct: Whether the prediction was correct (if known)
            metadata: Additional metadata about the classification
        """
        if is_correct is None and correct_category is not None:
            is_correct = predicted_category == correct_category
            
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "predicted_category": predicted_category,
            "confidence": confidence,
            "is_correct": is_correct,
            "metadata": metadata or {}
        }
        
        if correct_category is not None:
            feedback_entry["correct_category"] = correct_category
            
        # Add feedback to collection
        self.feedback_data.append(feedback_entry)
        
        # Trim collection if needed
        if len(self.feedback_data) > self.max_entries:
            self.feedback_data = self.feedback_data[-self.max_entries:]
            
        # Update statistics
        self._update_stats(feedback_entry)
        
        # Save feedback data periodically (every 10 entries)
        if self.stats["total_feedback"] % 10 == 0:
            self._save_feedback()
    
    def _update_stats(self, entry: Dict[str, Any]) -> None:
        """Update statistics with new feedback entry."""
        self.stats["total_feedback"] += 1
        
        if entry["is_correct"]:
            self.stats["correct_classifications"] += 1
            # Update running average of confidence for correct classifications
            prev_avg = self.stats["avg_confidence_correct"]
            prev_count = self.stats["correct_classifications"] - 1
            if prev_count > 0:
                self.stats["avg_confidence_correct"] = (
                    (prev_avg * prev_count + entry["confidence"]) / 
                    self.stats["correct_classifications"]
                )
            else:
                self.stats["avg_confidence_correct"] = entry["confidence"]
        elif entry["is_correct"] is False:  # explicitly False, not None
            self.stats["incorrect_classifications"] += 1
            # Update running average of confidence for incorrect classifications
            prev_avg = self.stats["avg_confidence_incorrect"]
            prev_count = self.stats["incorrect_classifications"] - 1
            if prev_count > 0:
                self.stats["avg_confidence_incorrect"] = (
                    (prev_avg * prev_count + entry["confidence"]) / 
                    self.stats["incorrect_classifications"]
                )
            else:
                self.stats["avg_confidence_incorrect"] = entry["confidence"]
    
    def _recalculate_stats(self) -> None:
        """Recalculate all statistics from feedback data."""
        self.stats = {
            "total_feedback": 0,
            "correct_classifications": 0,
            "incorrect_classifications": 0,
            "avg_confidence_correct": 0,
            "avg_confidence_incorrect": 0
        }
        
        correct_confidences = []
        incorrect_confidences = []
        
        for entry in self.feedback_data:
            self.stats["total_feedback"] += 1
            
            if entry.get("is_correct"):
                self.stats["correct_classifications"] += 1
                correct_confidences.append(entry["confidence"])
            elif entry.get("is_correct") is False:  # explicitly False, not None
                self.stats["incorrect_classifications"] += 1
                incorrect_confidences.append(entry["confidence"])
        
        # Calculate averages
        if correct_confidences:
            self.stats["avg_confidence_correct"] = sum(correct_confidences) / len(correct_confidences)
        if incorrect_confidences:
            self.stats["avg_confidence_incorrect"] = sum(incorrect_confidences) / len(incorrect_confidences)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about collected feedback.
        
        Returns:
            Dict[str, Any]: Dictionary of statistics
        """
        stats = self.stats.copy()
        
        # Calculate accuracy if possible
        total_classified = stats["correct_classifications"] + stats["incorrect_classifications"]
        if total_classified > 0:
            stats["accuracy"] = stats["correct_classifications"] / total_classified
            
        return stats
    
    def get_feedback_entries(self, 
                            limit: int = 100, 
                            category: Optional[str] = None,
                            only_incorrect: bool = False) -> List[Dict[str, Any]]:
        """
        Get feedback entries matching specified criteria.
        
        Args:
            limit: Maximum number of entries to return
            category: Only return entries for this category
            only_incorrect: Only return entries where classification was incorrect
            
        Returns:
            List[Dict[str, Any]]: List of feedback entries
        """
        filtered_entries = self.feedback_data
        
        if category is not None:
            filtered_entries = [e for e in filtered_entries 
                               if e["predicted_category"] == category]
                               
        if only_incorrect:
            filtered_entries = [e for e in filtered_entries 
                               if e.get("is_correct") is False]
                               
        # Return most recent entries first, up to limit
        return filtered_entries[-limit:][::-1] 