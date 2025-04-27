"""
Document compression utilities for context optimization.

This module provides functions to compress and prioritize documents based on
query relevance, ensuring the most important information is preserved when
fitting documents into a model's context window.
"""

from langchain.schema import Document
from typing import List, Dict, Any, Optional, Callable
import re
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from utils.token_counter import count_tokens, truncate_to_token_limit

class RelevanceScorer:
    """
    Scores document chunks based on relevance to a query.
    
    Uses embeddings to determine semantic similarity between query and document chunks.
    """
    
    def __init__(self, embeddings=None):
        """
        Initialize the relevance scorer.
        
        Args:
            embeddings: Embedding model to use (defaults to creating a new one if None)
        """
        self.embeddings = embeddings
        if self.embeddings is None:
            try:
                self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            except Exception as e:
                print(f"Warning: Failed to initialize embeddings for relevance scoring: {e}")
                self.embeddings = None
    
    def _get_embeddings(self, texts: List[str]) -> Optional[List[List[float]]]:
        """
        Get embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors or None if embedding fails
        """
        if not self.embeddings:
            return None
            
        try:
            return self.embeddings.embed_documents(texts)
        except Exception as e:
            print(f"Warning: Error generating embeddings: {e}")
            return None
    
    def score_by_embeddings(self, query: str, documents: List[Document]) -> List[float]:
        """
        Score documents by embedding similarity to query.
        
        Args:
            query: The query string
            documents: List of documents to score
            
        Returns:
            List of relevance scores (higher is more relevant)
        """
        if not self.embeddings:
            # Fallback to keyword scoring if embeddings unavailable
            return self.score_by_keywords(query, documents)
        
        try:
            # Extract document content
            doc_texts = [doc.page_content for doc in documents]
            
            # Get embeddings for query and documents
            query_embedding = self.embeddings.embed_query(query)
            doc_embeddings = self._get_embeddings(doc_texts)
            
            if not doc_embeddings:
                return self.score_by_keywords(query, documents)
            
            # Calculate cosine similarity
            scores = []
            for doc_embedding in doc_embeddings:
                # Compute cosine similarity
                dot_product = sum(q * d for q, d in zip(query_embedding, doc_embedding))
                query_norm = sum(q * q for q in query_embedding) ** 0.5
                doc_norm = sum(d * d for d in doc_embedding) ** 0.5
                
                if query_norm == 0 or doc_norm == 0:
                    scores.append(0.0)
                else:
                    scores.append(dot_product / (query_norm * doc_norm))
            
            return scores
        except Exception as e:
            print(f"Warning: Error in embedding-based scoring: {e}")
            return self.score_by_keywords(query, documents)
    
    def score_by_keywords(self, query: str, documents: List[Document]) -> List[float]:
        """
        Score documents by keyword overlap with query.
        
        Args:
            query: The query string
            documents: List of documents to score
            
        Returns:
            List of relevance scores (higher is more relevant)
        """
        # Extract important words from query (words with 3+ chars)
        query_words = set(re.findall(r'\b\w{3,}\b', query.lower()))
        if not query_words:
            # If no substantial words found, use all words
            query_words = set(query.lower().split())
        
        scores = []
        for doc in documents:
            content = doc.page_content.lower()
            
            # Count keyword occurrences
            matches = sum(content.count(word) for word in query_words)
            
            # Calculate density (matches per 100 chars)
            density = matches * 100 / max(1, len(content))
            
            # Check for exact phrase matches
            exact_match_bonus = 0
            if len(query) > 5:  # Only for substantial queries
                exact_match_count = content.count(query.lower())
                exact_match_bonus = exact_match_count * 2  # Bonus for exact matches
            
            # Combine signals
            scores.append(density + exact_match_bonus)
        
        # Normalize scores to 0-1 range if we have more than one document
        if scores and max(scores) > 0:
            scores = [score / max(scores) for score in scores]
        
        return scores

def compress_documents(
    documents: List[Document], 
    query: str, 
    model_capabilities: Dict[str, Any],
    embeddings=None
) -> List[Document]:
    """
    Compress documents to fit within context constraints while prioritizing relevance.
    
    Args:
        documents: List of documents to compress
        query: The query string
        model_capabilities: Dictionary containing model capabilities
        embeddings: Optional embeddings model to use for relevance scoring
        
    Returns:
        Compressed list of documents optimized for the query and model
    """
    if not documents:
        return []
    
    # Initialize relevance scorer
    scorer = RelevanceScorer(embeddings)
    
    # Score documents by relevance to query
    relevance_scores = scorer.score_by_embeddings(query, documents)
    
    # Create document-score pairs and sort by relevance (highest first)
    doc_score_pairs = list(zip(documents, relevance_scores))
    doc_score_pairs.sort(key=lambda pair: pair[1], reverse=True)
    
    # Determine max tokens based on model size
    from utils.token_counter import get_max_tokens_for_model
    max_tokens = get_max_tokens_for_model(model_capabilities)
    
    # Start with most relevant documents and add until we reach token limit
    compressed_docs = []
    current_tokens = 0
    
    # Always include at least one document even if it exceeds the limit
    must_include_one = True
    
    for doc, score in doc_score_pairs:
        # Count tokens in this document
        doc_tokens = count_tokens(doc.page_content)
        
        if current_tokens + doc_tokens <= max_tokens or must_include_one:
            # If fits within limit or it's the first document, add complete
            compressed_docs.append(doc)
            current_tokens += doc_tokens
            must_include_one = False
        else:
            # If we're on a document with low relevance, just skip it
            if score < 0.5 and len(compressed_docs) > 0:
                continue
                
            # Try to include partial document (it's relevant but doesn't fully fit)
            remaining_tokens = max_tokens - current_tokens
            if remaining_tokens > 50:  # Only if we have meaningful space left
                # Truncate document to fit
                truncated_content = truncate_to_token_limit(doc.page_content, remaining_tokens)
                
                # Create new document with truncated content but same metadata
                truncated_doc = Document(
                    page_content=truncated_content,
                    metadata=doc.metadata.copy() if doc.metadata else {}
                )
                
                compressed_docs.append(truncated_doc)
                break  # We've reached our token limit
    
    return compressed_docs

def balanced_document_truncation(
    documents: List[Document], 
    model_capabilities: Dict[str, Any]
) -> List[Document]:
    """
    Apply balanced truncation to documents when query relevance is not available.
    
    This is a fallback method that provides better truncation than simple
    character-based truncation by keeping both the start and end of documents.
    
    Args:
        documents: List of documents to truncate
        model_capabilities: Dictionary containing model capabilities
        
    Returns:
        List of truncated documents
    """
    if not documents:
        return []
    
    # Determine max tokens per document
    from utils.token_counter import get_max_tokens_for_model
    max_total_tokens = get_max_tokens_for_model(model_capabilities)
    
    # Assign token budget to each document
    tokens_per_doc = max_total_tokens // len(documents)
    
    truncated_docs = []
    for doc in documents:
        # Get token count for this document
        doc_tokens = count_tokens(doc.page_content)
        
        if doc_tokens <= tokens_per_doc:
            # Document already fits in budget, keep as is
            truncated_docs.append(doc)
        else:
            # Document needs truncation, keep beginning and end
            # For important documents, keep both beginning and end
            half_token_budget = tokens_per_doc // 2
            
            # Create new document with truncated content
            truncated_content = truncate_to_token_limit(
                doc.page_content[:len(doc.page_content)//2], 
                half_token_budget
            ) + "..." + doc.page_content[-len(doc.page_content)//4:]
            
            truncated_doc = Document(
                page_content=truncated_content,
                metadata=doc.metadata.copy() if doc.metadata else {}
            )
            
            truncated_docs.append(truncated_doc)
    
    return truncated_docs 