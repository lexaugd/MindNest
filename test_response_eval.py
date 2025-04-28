#!/usr/bin/env python
"""
Script to test the response evaluation function directly.
"""

import sys
import traceback
from langchain.schema import Document
from mindnest.utils.response_evaluation import check_question_answered, check_formatting

def test_check_question_answered():
    print("Testing check_question_answered function:")
    
    # Test case 1: Direct question with answer
    query = "When was Python created?"
    response = "Python was created by Guido van Rossum and released in 1991."
    result = check_question_answered(query, response)
    print(f"Query: '{query}'")
    print(f"Response: '{response}'")
    print(f"Result: {result}\n")
    
    # Test case 2: "Tell me about" query
    query = "Tell me about Python"
    response = "Python is a high-level programming language created by Guido van Rossum."
    result = check_question_answered(query, response)
    print(f"Query: '{query}'")
    print(f"Response: '{response}'")
    print(f"Result: {result}\n")
    
    # Test case 3: Another "Tell me about" query
    query = "Tell me about JavaScript"
    response = "JavaScript is a programming language used for web development."
    result = check_question_answered(query, response)
    print(f"Query: '{query}'")
    print(f"Response: '{response}'")
    print(f"Result: {result}\n")

def test_check_formatting():
    print("Testing check_formatting function:")
    
    # Test case 1: Well-formatted response
    response = "This is a well-formatted response. It has multiple sentences and reasonable length."
    issues = check_formatting(response)
    print(f"Response: '{response}'")
    print(f"Issues: {issues}\n")
    
    # Test case 2: Code block mismatch
    response = "Bad code block:\n```python\ndef hello():\n    print('hello')\n"  # Missing closing ```
    issues = check_formatting(response)
    print(f"Response: '{response}'")
    print(f"Issues: {issues}\n")

if __name__ == "__main__":
    try:
        print("=" * 80)
        print("RESPONSE EVALUATION TEST")
        print("=" * 80)
        
        test_check_question_answered()
        test_check_formatting()
        
        print("All tests completed.")
    except Exception as e:
        print(f"Error during testing: {e}")
        traceback.print_exc() 