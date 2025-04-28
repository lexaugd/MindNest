"""
Tests for the response quality evaluation utilities.
"""

import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add the parent directory to path to allow importing the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mindnest.utils.response_evaluation import (
    evaluate_response_quality,
    evaluate_relevance,
    detect_hallucination,
    check_formatting, 
    is_length_appropriate,
    check_question_answered,
    calculate_overall_quality,
    should_regenerate_response
)


class TestResponseEvaluation(unittest.TestCase):
    """Test class for response evaluation utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock document for context testing
        self.doc1 = MagicMock()
        self.doc1.page_content = "The Roman Empire fell in 476 CE. Julius Caesar was assassinated in 44 BCE."
        
        self.doc2 = MagicMock()
        self.doc2.page_content = "Python was created by Guido van Rossum and released in 1991."
        
        self.context_docs = [self.doc1, self.doc2]
        
        # Sample model capabilities
        self.model_capabilities = {
            "size": "large",
            "token_limit": 4096,
            "strengths": ["factual_recall", "code_generation"]
        }
    
    def test_evaluate_relevance(self):
        """Test the relevance evaluation function."""
        # Test high relevance
        query = "When did the Roman Empire fall?"
        response = "The Roman Empire fell in 476 CE when Romulus Augustus was deposed."
        score = evaluate_relevance(query, response)
        self.assertGreater(score, 0.7, "High relevance score expected")
        
        # Test low relevance
        query = "When did the Roman Empire fall?"
        response = "Python is a popular programming language."
        score = evaluate_relevance(query, response)
        self.assertLess(score, 0.3, "Low relevance score expected")
        
        # Test empty query
        query = ""
        response = "This is a response."
        score = evaluate_relevance(query, response)
        self.assertEqual(score, 0.5, "Default score for empty query expected")
        
        # Test length penalties
        query = "Roman Empire"
        response = "Yes."  # Too short
        score = evaluate_relevance(query, response)
        self.assertLess(score, 0.5, "Penalty for very short response expected")
    
    def test_detect_hallucination(self):
        """Test the hallucination detection function."""
        # Test valid response (no hallucination)
        response = "The Roman Empire fell in 476 CE."
        result = detect_hallucination(response, self.context_docs)
        self.assertFalse(result, "No hallucination expected for factual statement")
        
        # Test hallucination
        response = "I am certain that the Roman Empire fell in 375 CE and was conquered by Vikings."
        result = detect_hallucination(response, self.context_docs)
        self.assertTrue(result, "Hallucination expected for incorrect statement with certainty")
        
        # Test another hallucination example
        response = "The document explicitly states that Julius Caesar built the Colosseum in 50 BCE."
        result = detect_hallucination(response, self.context_docs)
        self.assertTrue(result, "Hallucination expected for unsubstantiated claim")
    
    def test_check_formatting(self):
        """Test the formatting check function."""
        # Test well-formatted response
        response = "This is a well-formatted response. It has multiple sentences and reasonable length."
        issues = check_formatting(response)
        self.assertEqual(len(issues), 0, "No issues expected for well-formatted text")
        
        # Test incomplete ending
        response = "This response ends with an incomplete sentence, which should be flagged as..."
        issues = check_formatting(response)
        self.assertEqual(len(issues), 1, "One issue expected for incomplete ending")
        self.assertIn("incomplete sentence", issues[0].lower())
        
        # Test excessive bullet points
        response = "Too many bullets:\n- Point 1\n- Point 2\n- Point 3\n- Point 4\n- Point 5\n- Point 6\n- Point 7\n- Point 8\n- Point 9\n- Point 10\n- Point 11"
        issues = check_formatting(response)
        self.assertEqual(len(issues), 1, "One issue expected for excessive bullets")
        self.assertIn("excessive use", issues[0].lower())
        
        # Test inconsistent bullet styles
        response = "Inconsistent bullets:\n- Point 1\n* Point 2\n• Point 3"
        issues = check_formatting(response)
        self.assertEqual(len(issues), 1, "One issue expected for inconsistent bullets")
        self.assertIn("inconsistent bullet", issues[0].lower())
        
        # Test code block mismatch
        response = "Bad code block:\n```python\ndef hello():\n    print('hello')\n"  # Missing closing ```
        issues = check_formatting(response)
        self.assertEqual(len(issues), 1, "One issue expected for mismatched code blocks")
        self.assertIn("code block", issues[0].lower())
    
    def test_is_length_appropriate(self):
        """Test the length appropriateness function."""
        # Test short query with concise response
        query = "What is Python?"
        response = "Python is a popular programming language created by Guido van Rossum."
        result = is_length_appropriate(query, response, self.model_capabilities)
        self.assertTrue(result, "Appropriate length expected")
        
        # Test short query with verbose response
        long_response = "Python is " + "very " * 200 + "popular."
        result = is_length_appropriate(query, long_response, self.model_capabilities)
        self.assertFalse(result, "Inappropriate length expected for verbose response to simple query")
        
        # Test complex query with appropriate response
        complex_query = "Can you explain the differences between functional and object-oriented programming paradigms, including their respective strengths and weaknesses?"
        medium_response = "The differences are " + "significant. " * 100
        result = is_length_appropriate(complex_query, medium_response, self.model_capabilities)
        self.assertTrue(result, "Appropriate length expected for complex query")
        
        # Test extremely short response
        result = is_length_appropriate(query, "Yes.", self.model_capabilities)
        self.assertFalse(result, "Inappropriate length expected for extremely short response")
    
    def test_check_question_answered(self):
        """Test the question answered check function."""
        # Test direct question with answer
        query = "When was Python created?"
        response = "Python was created by Guido van Rossum and released in 1991."
        result = check_question_answered(query, response)
        self.assertTrue(result, "Question should be marked as answered")
        
        # Test direct question with irrelevant response
        query = "When was Python created?"
        response = "Programming languages are used for software development."
        result = check_question_answered(query, response)
        self.assertFalse(result, "Question should be marked as not answered")
        
        # Test direct yes/no question
        query = "Is Python a compiled language?"
        response = "No, Python is typically an interpreted language."
        result = check_question_answered(query, response)
        self.assertTrue(result, "Yes/no question should be marked as answered")
        
        # Test non-question
        query = "Tell me about Python."
        response = "Python is a high-level programming language."
        result = check_question_answered(query, response)
        self.assertTrue(result, "Non-question should be marked as answered with relevant response")
    
    def test_calculate_overall_quality(self):
        """Test the overall quality calculation function."""
        # Test high quality response
        evaluation = {
            "relevance_score": 0.9,
            "potential_hallucination": False,
            "formatting_issues": [],
            "length_appropriate": True,
            "question_answered": True
        }
        score = calculate_overall_quality(evaluation)
        self.assertGreater(score, 0.8, "High overall quality expected")
        
        # Test low quality response
        evaluation = {
            "relevance_score": 0.3,
            "potential_hallucination": True,
            "formatting_issues": ["Issue 1", "Issue 2", "Issue 3"],
            "length_appropriate": False,
            "question_answered": False
        }
        score = calculate_overall_quality(evaluation)
        self.assertLess(score, 0.3, "Low overall quality expected")
    
    def test_should_regenerate_response(self):
        """Test the regeneration decision function."""
        # Test case that should not regenerate
        evaluation = {
            "relevance_score": 0.9,
            "potential_hallucination": False,
            "formatting_issues": [],
            "length_appropriate": True,
            "question_answered": True,
            "overall_quality": 0.9
        }
        regenerate, reason = should_regenerate_response(evaluation)
        self.assertFalse(regenerate, "Should not regenerate high quality response")
        
        # Test case with hallucination
        evaluation = {
            "relevance_score": 0.9,
            "potential_hallucination": True,
            "formatting_issues": [],
            "length_appropriate": True,
            "question_answered": True,
            "overall_quality": 0.6
        }
        regenerate, reason = should_regenerate_response(evaluation)
        self.assertTrue(regenerate, "Should regenerate response with hallucination")
        self.assertIn("hallucination", reason.lower())
        
        # Test case with low quality score
        evaluation = {
            "relevance_score": 0.3,
            "potential_hallucination": False,
            "formatting_issues": ["Issue 1"],
            "length_appropriate": False,
            "question_answered": True,
            "overall_quality": 0.4
        }
        regenerate, reason = should_regenerate_response(evaluation)
        self.assertTrue(regenerate, "Should regenerate response with low quality")
        self.assertIn("low quality", reason.lower())
    
    def test_evaluate_response_quality_integration(self):
        """Integration test for the main evaluation function."""
        query = "When did the Roman Empire fall?"
        good_response = "The Roman Empire fell in 476 CE when the last Roman emperor, Romulus Augustus, was deposed."
        
        # Test good response
        result = evaluate_response_quality(query, good_response, self.context_docs, self.model_capabilities)
        self.assertGreater(result["relevance_score"], 0.7)
        self.assertFalse(result["potential_hallucination"])
        self.assertTrue(result["question_answered"])
        self.assertTrue(result["length_appropriate"])
        self.assertEqual(len(result["formatting_issues"]), 0)
        self.assertGreater(result["overall_quality"], 0.7)
        
        # Test bad response
        bad_response = "The Roman Empire... "  # Incomplete
        result = evaluate_response_quality(query, bad_response, self.context_docs, self.model_capabilities)
        self.assertLess(result["overall_quality"], 0.7)


if __name__ == "__main__":
    unittest.main() 