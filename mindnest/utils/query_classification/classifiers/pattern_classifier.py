"""
Pattern-based query classifier.

This module implements a classifier that uses regex patterns to classify queries.
"""

import re
import time
from typing import Dict, List, Tuple, Any

from .base import BaseClassifier


class PatternClassifier(BaseClassifier):
    """
    Classifier that uses regex patterns to categorize queries.
    
    This classifier is fast and lightweight, making it ideal for initial filtering
    or obvious cases. It works by matching the query against a set of predefined
    patterns for each category.
    """
    
    def __init__(self, extended_patterns: bool = True):
        """
        Initialize the pattern classifier.
        
        Args:
            extended_patterns: Whether to use the extended set of patterns for
                more comprehensive but potentially slower matching.
        """
        self.extended_patterns = extended_patterns
        self.patterns = {}
        self.compiled_patterns = {}
        self.initialized = False
        self.stats = {
            "calls": 0,
            "direct_matches": 0,
            "indirect_matches": 0,
            "default_cases": 0,
            "avg_time_ms": 0
        }
    
    def initialize(self) -> None:
        """Initialize the classifier by loading and compiling patterns."""
        if self.initialized:
            return
        
        self.patterns = self._load_patterns(self.extended_patterns)
        
        # Compile patterns for efficiency
        self.compiled_patterns = {}
        for category, pattern_list in self.patterns.items():
            self.compiled_patterns[category] = [
                re.compile(p, re.IGNORECASE) for p in pattern_list
            ]
        
        self.initialized = True
        print(f"Pattern classifier initialized with {sum(len(p) for p in self.patterns.values())} patterns")
    
    def _load_patterns(self, extended: bool = True) -> Dict[str, List[str]]:
        """
        Load classification patterns with optional extended set.
        
        Args:
            extended: Whether to include extended patterns
            
        Returns:
            Dict[str, List[str]]: Dictionary of patterns by category
        """
        # Basic patterns for core query types
        patterns = {
            "DOCUMENT_SEARCH": [
                r"^find\s+.+",
                r"^search\s+for\s+.+",
                r"^where\s+can\s+I\s+find\s+.+",
                r"^locate\s+.+"
            ],
            "CONCISE_QUERY": [
                r"^(summarize|summarise|brief|concise|short|tl;dr|tldr|summary)\s+.+",
                r".+\s+in\s+(one|1|a|few)\s+(sentence|sentences|line|lines|paragraph|paragraphs|words).*$",
                r"^(explain|describe|define|what\s+is)\s+.+\s+(briefly|concisely|in short).*$"
            ],
            "CONVERSATION": [
                r"^(hi|hello|hey|thanks|thank you|ok|okay|good|nice|cool|great|bye)$",
                r"^how\s+are\s+you.*$",
                r"^what(\s+is|\s*'s)\s+your\s+name.*$"
            ]
        }
        
        # Add extended patterns if requested
        if extended:
            patterns["DOCUMENT_SEARCH"].extend([
                r"^show\s+me\s+.+\s+(about|related\s+to)\s+.+",
                r"^list\s+all\s+.+",
                r"^get\s+information\s+(about|on)\s+.+"
            ])
            patterns["CONCISE_QUERY"].extend([
                r"^(give|provide)\s+(me\s+)?(a\s+)?(brief|short|concise|quick|simple)\s+(summary|description|explanation|overview|answer)\s+.+$",
                r"^(can\s+you\s+)?(quickly|briefly|concisely)\s+(tell|explain|describe|summarize)\s+.+$",
                r"^short\s+(answer|explanation|description)\s+.+$"
            ])
            patterns["CONVERSATION"].extend([
                r".*\b(how do you|how you doing|how is it going)\b.*",
                r".*\b(nice to meet you|pleasure to meet you)\b.*",
                r".*\b(what do you think|your opinion|you like|you enjoy)\b.*",
                r".*\b(thanks|thank you|appreciate it)\b.*",
                r".*\b(goodbye|bye|see you|talk to you later)\b.*"
            ])
        
        return patterns
    
    def classify(self, query: str) -> Tuple[str, float]:
        """
        Classify a query based on regex pattern matching.
        
        Args:
            query: The query text to classify
            
        Returns:
            Tuple[str, float]: Category and confidence score
        """
        if not self.initialized:
            self.initialize()
        
        start_time = time.time()
        self.stats["calls"] += 1
        
        # Default category and confidence
        best_category = "DOCUMENT_QUERY"  # Default when no patterns match
        best_confidence = 0.5
        
        # Fast path for search queries
        if query.lower().startswith("find "):
            self.stats["direct_matches"] += 1
            end_time = time.time()
            self._update_timing_stats(start_time, end_time)
            return "DOCUMENT_SEARCH", 0.98
        
        # Check each category's patterns
        for category, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(query.lower()):
                    self.stats["direct_matches"] += 1
                    end_time = time.time()
                    self._update_timing_stats(start_time, end_time)
                    return category, 0.95  # High confidence for direct pattern matches
        
        # Check if it meets general document query criteria
        doc_indicators = ["what", "how", "why", "when", "where", "which", "explain", "describe", "tell me about"]
        doc_keywords = ["class", "method", "function", "object", "system", "component", "documentation", 
                       "code", "api", "interface", "library", "module", "framework"]
        
        has_indicator = any(query.lower().startswith(ind) for ind in doc_indicators)
        has_keyword = any(kw in query.lower() for kw in doc_keywords)
        
        if has_indicator and has_keyword:
            self.stats["indirect_matches"] += 1
            end_time = time.time()
            self._update_timing_stats(start_time, end_time)
            return "DOCUMENT_QUERY", 0.85
        elif has_indicator or has_keyword:
            self.stats["indirect_matches"] += 1
            end_time = time.time()
            self._update_timing_stats(start_time, end_time)
            return "DOCUMENT_QUERY", 0.7
        
        # Default case - longer queries are likely document queries
        self.stats["default_cases"] += 1
        if len(query.split()) >= 4:
            confidence = 0.6
        else:
            confidence = 0.5
        
        end_time = time.time()
        self._update_timing_stats(start_time, end_time)
        return best_category, confidence
    
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