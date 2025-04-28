"""
BM25 retriever for keyword-based document retrieval.

Implements the BM25 algorithm for keyword-based retrieval
to complement embedding-based vector retrieval.
"""

import logging
import math
import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from collections import Counter, defaultdict
from langchain.schema import Document, BaseRetriever

# Setup logging
logger = logging.getLogger(__name__)

class BM25Retriever(BaseRetriever):
    """
    BM25 retriever for keyword-based document retrieval.
    Provides sparse retrieval to complement dense vector retrieval.
    """
    
    def __init__(
        self,
        documents: Optional[List[Document]] = None,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25
    ):
        """
        Initialize the BM25 retriever.
        
        Args:
            documents: List of documents to index
            k1: Term saturation parameter (1.2-2.0 typical)
            b: Document length normalization (0.5-0.8 typical)
            epsilon: Additive smoothing parameter
        """
        super().__init__()
        
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        
        # Inverted index and document data
        self.inverted_index = defaultdict(list)
        self.doc_lengths = {}
        self.avg_doc_length = 0
        self.doc_freqs = Counter()
        self.documents = {}
        self.total_docs = 0
        
        # Index documents if provided
        if documents:
            self.index_documents(documents)
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of lowercase tokens
        """
        # Simple tokenization with regex to extract words
        return re.findall(r'\b\w+\b', text.lower())
    
    def _calculate_tf(self, term: str, doc_tokens: List[str]) -> float:
        """
        Calculate term frequency.
        
        Args:
            term: The term
            doc_tokens: List of tokens in the document
            
        Returns:
            Term frequency value
        """
        # Raw term frequency
        return doc_tokens.count(term) / len(doc_tokens) if doc_tokens else 0
    
    def _preprocess_document(self, doc: Document) -> Tuple[str, List[str]]:
        """
        Preprocess a document for indexing.
        
        Args:
            doc: The document to preprocess
            
        Returns:
            Tuple of (doc_id, tokens)
        """
        # Use document source as ID if available, otherwise use content hash
        doc_id = doc.metadata.get("source", None)
        if not doc_id:
            # Create a semi-unique identifier based on content 
            doc_id = f"doc_{hash(doc.page_content) % 10000000:07d}"
            
        # Tokenize content
        tokens = self._tokenize(doc.page_content)
        
        return doc_id, tokens
    
    def index_documents(self, documents: List[Document]) -> None:
        """
        Index documents for BM25 retrieval.
        
        Args:
            documents: List of documents to index
        """
        logger.info(f"Indexing {len(documents)} documents for BM25 retrieval")
        
        # Reset index if needed
        if self.total_docs > 0:
            self.inverted_index = defaultdict(list)
            self.doc_lengths = {}
            self.doc_freqs = Counter()
            self.documents = {}
            self.total_docs = 0
        
        # Process all documents
        total_tokens = 0
        for doc in documents:
            doc_id, tokens = self._preprocess_document(doc)
            
            # Skip empty documents
            if not tokens:
                continue
            
            # Store document and length
            self.documents[doc_id] = doc
            self.doc_lengths[doc_id] = len(tokens)
            total_tokens += len(tokens)
            self.total_docs += 1
            
            # Calculate term frequencies for the document
            term_freqs = Counter(tokens)
            
            # Update inverted index with term frequencies
            for term, freq in term_freqs.items():
                self.inverted_index[term].append((doc_id, freq))
                self.doc_freqs[term] += 1
        
        # Calculate average document length
        self.avg_doc_length = total_tokens / self.total_docs if self.total_docs > 0 else 0
        
        logger.info(f"BM25 indexing complete: {self.total_docs} documents, {len(self.inverted_index)} terms")
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add new documents to the index.
        
        Args:
            documents: List of new documents to add
        """
        logger.info(f"Adding {len(documents)} documents to BM25 index")
        
        # Process all new documents
        total_tokens = self.avg_doc_length * self.total_docs
        for doc in documents:
            doc_id, tokens = self._preprocess_document(doc)
            
            # Skip empty documents or already indexed
            if not tokens or doc_id in self.documents:
                continue
            
            # Store document and length
            self.documents[doc_id] = doc
            self.doc_lengths[doc_id] = len(tokens)
            total_tokens += len(tokens)
            self.total_docs += 1
            
            # Calculate term frequencies for the document
            term_freqs = Counter(tokens)
            
            # Update inverted index with term frequencies
            for term, freq in term_freqs.items():
                self.inverted_index[term].append((doc_id, freq))
                self.doc_freqs[term] += 1
        
        # Recalculate average document length
        self.avg_doc_length = total_tokens / self.total_docs if self.total_docs > 0 else 0
    
    def _calculate_bm25_score(self, query_tokens: List[str], doc_id: str) -> float:
        """
        Calculate BM25 score for a document-query pair.
        
        Args:
            query_tokens: List of tokens in the query
            doc_id: Document ID
            
        Returns:
            BM25 score
        """
        # Skip if document doesn't exist
        if doc_id not in self.doc_lengths:
            return 0.0
        
        # Get document length
        doc_length = self.doc_lengths[doc_id]
        
        # Calculate document length normalization term
        len_norm = self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
        
        # Accumulate score for all query terms
        score = 0.0
        for token in query_tokens:
            # Skip if term not in index
            if token not in self.inverted_index:
                continue
            
            # Get document frequency and term frequency
            df = self.doc_freqs[token]
            tf = 0.0
            
            # Find term frequency in this document
            for did, freq in self.inverted_index[token]:
                if did == doc_id:
                    tf = freq
                    break
            
            # Skip if term not in document
            if tf == 0:
                continue
            
            # Calculate IDF component with smoothing
            idf = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1.0)
            
            # Calculate BM25 term score
            term_score = idf * ((tf * (self.k1 + 1)) / (tf + len_norm))
            
            # Add to total score
            score += term_score
        
        return score
    
    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager=None, 
        k: int = 4
    ) -> List[Document]:
        """
        Get relevant documents for a query using BM25.
        
        Args:
            query: The query string
            run_manager: Run manager for callbacks
            k: Number of documents to return
            
        Returns:
            List of relevant documents
        """
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        # Skip if query is empty
        if not query_tokens:
            return []
        
        # Calculate scores for all documents that contain at least one query term
        # This is more efficient than scoring all documents
        scored_docs = {}
        for token in query_tokens:
            if token in self.inverted_index:
                # For each document that contains this term
                for doc_id, _ in self.inverted_index[token]:
                    # Only calculate score once per document
                    if doc_id not in scored_docs:
                        scored_docs[doc_id] = self._calculate_bm25_score(query_tokens, doc_id)
        
        # Sort documents by score
        sorted_docs = sorted(
            [(doc_id, score) for doc_id, score in scored_docs.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Retrieve documents and normalize scores
        results = []
        max_score = sorted_docs[0][1] if sorted_docs else 1.0
        for doc_id, score in sorted_docs[:k]:
            # Get document
            doc = self.documents[doc_id]
            
            # Normalize score to 0-1 range
            normalized_score = score / max_score if max_score > 0 else 0.0
            
            # Add score to metadata
            doc.metadata = doc.metadata or {}
            doc.metadata["bm25_score"] = normalized_score
            
            # Add to results
            results.append(doc)
            
            # Break once we have enough documents
            if len(results) >= k:
                break
        
        logger.debug(f"BM25 retrieval: {len(results)} docs for query '{query}'")
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the retriever.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "total_docs": self.total_docs,
            "avg_doc_length": self.avg_doc_length,
            "vocabulary_size": len(self.inverted_index),
            "k1": self.k1,
            "b": self.b,
            "epsilon": self.epsilon
        }

# Factory function to create a BM25 retriever
def create_bm25_retriever(
    documents: Optional[List[Document]] = None,
    k1: float = 1.5,
    b: float = 0.75,
    epsilon: float = 0.25
) -> BM25Retriever:
    """
    Create a BM25 retriever with the given parameters.
    
    Args:
        documents: List of documents to index
        k1: Term saturation parameter
        b: Document length normalization
        epsilon: Additive smoothing parameter
        
    Returns:
        BM25Retriever instance
    """
    return BM25Retriever(
        documents=documents,
        k1=k1,
        b=b,
        epsilon=epsilon
    ) 