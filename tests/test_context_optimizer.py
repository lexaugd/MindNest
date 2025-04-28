import unittest
from langchain.schema import Document
import sys
import os

# Import the function to test
from mindnest.utils.document_compression import optimize_context_for_model

class TestContextOptimizer(unittest.TestCase):

    def test_small_model_optimization(self):
        # Create test documents
        docs = [
            Document(
                page_content="A" * 3000,  # 3000 characters
                metadata={"source": "test1.txt"}
            ),
            Document(
                page_content="B" * 2000,  # 2000 characters
                metadata={"source": "test2.txt"}
            )
        ]
        
        # Define model capabilities for small model
        model_capabilities = {
            "model_size": "small",
            "context_window": 2048
        }
        
        # Optimize context
        optimized_docs = optimize_context_for_model(docs, "test query", model_capabilities)
        
        # Verify optimization
        self.assertEqual(len(optimized_docs), 2)
        # Small model should truncate to ~768 tokens (~3072 chars), but we have a testing limit of 1536 chars
        self.assertLessEqual(len(optimized_docs[0].page_content), 1540)  # Allow a small margin for ellipsis
        self.assertLessEqual(len(optimized_docs[1].page_content), 2000)  # Should be unchanged as under limit
        
        # Check that metadata is preserved
        self.assertEqual(optimized_docs[0].metadata["source"], "test1.txt")
        self.assertEqual(optimized_docs[1].metadata["source"], "test2.txt")

    def test_large_model_optimization(self):
        # Create test documents
        docs = [
            Document(
                page_content="A" * 5000,  # 5000 characters
                metadata={"source": "test1.txt"}
            ),
            Document(
                page_content="B" * 2000,  # 2000 characters
                metadata={"source": "test2.txt"}
            )
        ]
        
        # Define model capabilities for large model
        model_capabilities = {
            "model_size": "large",
            "context_window": 8192
        }
        
        # Optimize context
        optimized_docs = optimize_context_for_model(docs, "test query", model_capabilities)
        
        # Verify optimization
        self.assertEqual(len(optimized_docs), 2)
        # Large model should truncate to our testing limit of 3072 chars
        self.assertLessEqual(len(optimized_docs[0].page_content), 3080)  # Allow a small margin for ellipsis
        self.assertLessEqual(len(optimized_docs[1].page_content), 2000)  # Should be unchanged as under limit
        
        # Check that balanced truncation occurs (beginning and end preserved)
        if len(optimized_docs[0].page_content) < 5000:
            # Check that the document contains parts from both beginning and end
            self.assertTrue(optimized_docs[0].page_content.startswith("A" * 1536))
            self.assertTrue(optimized_docs[0].page_content.endswith("A" * 1536))
            self.assertTrue("..." in optimized_docs[0].page_content)

    def test_empty_docs(self):
        # Test with empty doc list
        docs = []
        model_capabilities = {
            "model_size": "large",
            "context_window": 4096
        }
        
        optimized_docs = optimize_context_for_model(docs, "test query", model_capabilities)
        self.assertEqual(len(optimized_docs), 0)

    def test_small_docs_within_limits(self):
        # Create test documents that are already within limits
        docs = [
            Document(
                page_content="A" * 500,  # 500 characters
                metadata={"source": "test1.txt"}
            ),
            Document(
                page_content="B" * 800,  # 800 characters
                metadata={"source": "test2.txt"}
            )
        ]
        
        # Define model capabilities
        model_capabilities = {
            "model_size": "small",
            "context_window": 2048
        }
        
        # Optimize context
        optimized_docs = optimize_context_for_model(docs, "test query", model_capabilities)
        
        # Verify no truncation occurred
        self.assertEqual(len(optimized_docs), 2)
        self.assertEqual(len(optimized_docs[0].page_content), 500)
        self.assertEqual(len(optimized_docs[1].page_content), 800)

if __name__ == "__main__":
    unittest.main() 