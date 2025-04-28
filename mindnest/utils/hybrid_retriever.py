"""
Hybrid retrieval module for improved document retrieval.

Combines vector similarity search with BM25 sparse retrieval
and adds similarity thresholds to improve retrieval quality.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from langchain.schema import Document, BaseRetriever

# Setup logging
logger = logging.getLogger(__name__)

class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever that combines multiple retrieval methods.
    Features configurable similarity thresholds and relevance-based ranking.
    """
    
    def __init__(
        self,
        vector_retriever=None,
        bm25_retriever=None,
        min_similarity_threshold: float = 0.25,
        use_reciprocal_rank_fusion: bool = True,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3
    ):
        """
        Initialize the hybrid retriever.
        
        Args:
            vector_retriever: Vector store retriever for semantic search
            bm25_retriever: BM25 retriever for keyword search
            min_similarity_threshold: Minimum similarity threshold (0-1)
            use_reciprocal_rank_fusion: Whether to use RRF for combining results
            vector_weight: Weight for vector similarity scores (0-1)
            bm25_weight: Weight for BM25 scores (0-1)
        """
        super().__init__()
        
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.min_similarity_threshold = min_similarity_threshold
        self.use_reciprocal_rank_fusion = use_reciprocal_rank_fusion
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        
        # Ensure weights sum to 1
        total_weight = self.vector_weight + self.bm25_weight
        if total_weight != 1.0:
            self.vector_weight = self.vector_weight / total_weight
            self.bm25_weight = self.bm25_weight / total_weight
            
        # Tracking stats
        self.stats = {
            "total_queries": 0,
            "filtered_docs": 0,
            "avg_retrieved_docs": 0,
            "avg_similarity_score": 0
        }
    
    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager=None,
        k: int = 4,
        fetch_k: int = 20,
        filter_threshold: Optional[float] = None
    ) -> List[Document]:
        """
        Get relevant documents for a query using hybrid retrieval.
        
        Args:
            query: The query string
            run_manager: Run manager for callbacks
            k: Number of documents to return
            fetch_k: Number of documents to initially fetch before filtering
            filter_threshold: Override the default similarity threshold
            
        Returns:
            List of relevant documents
        """
        # Track stats
        self.stats["total_queries"] += 1
        
        # Use instance threshold if not overridden
        threshold = filter_threshold if filter_threshold is not None else self.min_similarity_threshold
        
        # Retrieve documents from available retrievers
        vector_docs = self._get_vector_docs(query, fetch_k)
        bm25_docs = self._get_bm25_docs(query, fetch_k)
        
        # Combine results
        if self.use_reciprocal_rank_fusion:
            merged_docs = self._reciprocal_rank_fusion(vector_docs, bm25_docs, k=k)
        else:
            merged_docs = self._weighted_score_fusion(vector_docs, bm25_docs, k=k)
        
        # Filter by threshold
        filtered_docs = []
        for doc in merged_docs:
            # Ensure document has score metadata
            score = doc.metadata.get("score", 0.0)
            
            # Skip documents below threshold
            if score < threshold:
                self.stats["filtered_docs"] += 1
                continue
                
            filtered_docs.append(doc)
            
            # Break if we have enough documents
            if len(filtered_docs) >= k:
                break
        
        # Update stats
        total_queries = max(1, self.stats["total_queries"])
        self.stats["avg_retrieved_docs"] = (
            (self.stats["avg_retrieved_docs"] * (total_queries - 1)) + len(filtered_docs)
        ) / total_queries
        
        if filtered_docs:
            avg_score = sum(doc.metadata.get("score", 0.0) for doc in filtered_docs) / len(filtered_docs)
            self.stats["avg_similarity_score"] = (
                (self.stats["avg_similarity_score"] * (total_queries - 1)) + avg_score
            ) / total_queries
        
        logger.info(f"Hybrid retrieval: {len(filtered_docs)} docs with avg score {self.stats['avg_similarity_score']:.4f}")
        return filtered_docs
    
    def _get_vector_docs(self, query: str, k: int) -> List[Document]:
        """
        Get documents from vector retriever.
        
        Args:
            query: The query string
            k: Number of documents to retrieve
            
        Returns:
            List of documents with scores
        """
        if not self.vector_retriever:
            return []
            
        try:
            # Try to use similarity_search_with_score if available
            if hasattr(self.vector_retriever, "similarity_search_with_score"):
                docs_and_scores = self.vector_retriever.similarity_search_with_score(query, k=k)
                
                # Convert to documents with scores in metadata
                docs = []
                for doc, score in docs_and_scores:
                    # Normalize score to 0-1 if needed (scores might be distances)
                    normalized_score = self._normalize_vector_score(score)
                    
                    # Add score to metadata
                    doc.metadata = doc.metadata or {}
                    doc.metadata["vector_score"] = normalized_score
                    doc.metadata["score"] = normalized_score  # For combined scoring
                    docs.append(doc)
                
                return docs
            else:
                # Fall back to regular similarity search
                docs = self.vector_retriever.get_relevant_documents(query, k=k)
                
                # Assign estimated scores since we don't have real scores
                for i, doc in enumerate(docs):
                    # Estimate score from position (1.0 for first, decreasing for later docs)
                    estimated_score = max(0.2, 1.0 - (i * 0.1))
                    
                    doc.metadata = doc.metadata or {}
                    doc.metadata["vector_score"] = estimated_score
                    doc.metadata["score"] = estimated_score
                
                return docs
        except Exception as e:
            logger.error(f"Error retrieving vector documents: {e}")
            return []
    
    def _get_bm25_docs(self, query: str, k: int) -> List[Document]:
        """
        Get documents from BM25 retriever.
        
        Args:
            query: The query string
            k: Number of documents to retrieve
            
        Returns:
            List of documents with scores
        """
        if not self.bm25_retriever:
            return []
            
        try:
            # Get documents from BM25
            docs = self.bm25_retriever.get_relevant_documents(query, k=k)
            
            # Assign estimated scores based on position
            for i, doc in enumerate(docs):
                # Estimate score from position
                estimated_score = max(0.2, 1.0 - (i * 0.1))
                
                doc.metadata = doc.metadata or {}
                doc.metadata["bm25_score"] = estimated_score
                
                # Only set combined score if no vector score exists
                if "score" not in doc.metadata:
                    doc.metadata["score"] = estimated_score
            
            return docs
        except Exception as e:
            logger.error(f"Error retrieving BM25 documents: {e}")
            return []
    
    def _normalize_vector_score(self, score: float) -> float:
        """
        Normalize vector scores to 0-1 range.
        
        Args:
            score: The raw score or distance
            
        Returns:
            Normalized score
        """
        # Check if score is a similarity (higher is better) or distance (lower is better)
        if score > 1.0:
            # Likely a distance (e.g., L2), convert to similarity
            # Using a heuristic to normalize distances - adjust as needed
            return max(0.0, 1.0 - min(1.0, score / 10.0))
        elif score < 0.0:
            # Some similarity measures might give negative values
            return max(0.0, (score + 1.0) / 2.0)
        else:
            # Already a similarity score
            return score
    
    def _weighted_score_fusion(
        self, 
        vector_docs: List[Document], 
        bm25_docs: List[Document],
        k: int
    ) -> List[Document]:
        """
        Combine results using weighted score fusion.
        
        Args:
            vector_docs: Documents from vector retrieval
            bm25_docs: Documents from BM25 retrieval
            k: Number of documents to return
            
        Returns:
            Combined and scored documents
        """
        # Create a unified dict to merge docs with the same source
        doc_map = {}
        
        # Process vector docs
        for doc in vector_docs:
            source = doc.metadata.get("source", id(doc))
            vector_score = doc.metadata.get("vector_score", 0.0)
            
            if source in doc_map:
                # Update existing document score
                doc_map[source]["vector_score"] = vector_score
            else:
                # Add new document
                doc_map[source] = {
                    "doc": doc,
                    "vector_score": vector_score,
                    "bm25_score": 0.0
                }
        
        # Process BM25 docs
        for doc in bm25_docs:
            source = doc.metadata.get("source", id(doc))
            bm25_score = doc.metadata.get("bm25_score", 0.0)
            
            if source in doc_map:
                # Update existing document score
                doc_map[source]["bm25_score"] = bm25_score
            else:
                # Add new document
                doc_map[source] = {
                    "doc": doc,
                    "vector_score": 0.0,
                    "bm25_score": bm25_score
                }
        
        # Calculate combined scores
        for source, data in doc_map.items():
            combined_score = (
                (data["vector_score"] * self.vector_weight) +
                (data["bm25_score"] * self.bm25_weight)
            )
            
            # Update metadata with scores
            doc = data["doc"]
            doc.metadata["vector_score"] = data["vector_score"]
            doc.metadata["bm25_score"] = data["bm25_score"]
            doc.metadata["score"] = combined_score
        
        # Sort by combined score and return top k
        sorted_docs = sorted(
            [data["doc"] for data in doc_map.values()],
            key=lambda x: x.metadata.get("score", 0.0),
            reverse=True
        )
        
        return sorted_docs[:k]
    
    def _reciprocal_rank_fusion(
        self, 
        vector_docs: List[Document], 
        bm25_docs: List[Document],
        k: int,
        rrf_k: int = 60
    ) -> List[Document]:
        """
        Combine results using Reciprocal Rank Fusion.
        
        Args:
            vector_docs: Documents from vector retrieval
            bm25_docs: Documents from BM25 retrieval
            k: Number of documents to return
            rrf_k: RRF constant for smoothing
            
        Returns:
            Combined and ranked documents
        """
        # Create unified doc map
        doc_map = {}
        
        # Process vector docs
        for i, doc in enumerate(vector_docs):
            source = doc.metadata.get("source", id(doc))
            vector_score = doc.metadata.get("vector_score", 0.0)
            vector_rank = i + 1  # 1-based rank
            
            if source in doc_map:
                # Update existing document
                doc_map[source]["vector_rank"] = vector_rank
                doc_map[source]["vector_score"] = vector_score
            else:
                # Add new document
                doc_map[source] = {
                    "doc": doc,
                    "vector_rank": vector_rank,
                    "vector_score": vector_score,
                    "bm25_rank": float('inf'),
                    "bm25_score": 0.0
                }
        
        # Process BM25 docs
        for i, doc in enumerate(bm25_docs):
            source = doc.metadata.get("source", id(doc))
            bm25_score = doc.metadata.get("bm25_score", 0.0) 
            bm25_rank = i + 1  # 1-based rank
            
            if source in doc_map:
                # Update existing document
                doc_map[source]["bm25_rank"] = bm25_rank
                doc_map[source]["bm25_score"] = bm25_score
            else:
                # Add new document
                doc_map[source] = {
                    "doc": doc,
                    "vector_rank": float('inf'),
                    "vector_score": 0.0,
                    "bm25_rank": bm25_rank,
                    "bm25_score": bm25_score
                }
        
        # Calculate RRF scores
        for source, data in doc_map.items():
            vector_rrf = 1.0 / (data["vector_rank"] + rrf_k) if data["vector_rank"] != float('inf') else 0.0
            bm25_rrf = 1.0 / (data["bm25_rank"] + rrf_k) if data["bm25_rank"] != float('inf') else 0.0
            
            # Combine RRF scores with linear weighting
            rrf_score = (vector_rrf * self.vector_weight) + (bm25_rrf * self.bm25_weight)
            
            # Calculate a weighted combination of raw scores too (as a secondary signal)
            raw_score = (data["vector_score"] * self.vector_weight) + (data["bm25_score"] * self.bm25_weight)
            
            # Update metadata with scores
            doc = data["doc"]
            doc.metadata["vector_score"] = data["vector_score"]
            doc.metadata["bm25_score"] = data["bm25_score"]
            doc.metadata["rrf_score"] = rrf_score
            doc.metadata["score"] = raw_score  # Keep the raw score as the main score
        
        # Sort by RRF score and return top k
        sorted_docs = sorted(
            [data["doc"] for data in doc_map.values()],
            key=lambda x: (x.metadata.get("rrf_score", 0.0), x.metadata.get("score", 0.0)),
            reverse=True
        )
        
        return sorted_docs[:k]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get retriever statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "min_similarity_threshold": self.min_similarity_threshold,
            "total_queries": self.stats["total_queries"],
            "filtered_docs": self.stats["filtered_docs"],
            "avg_retrieved_docs": self.stats["avg_retrieved_docs"],
            "avg_similarity_score": self.stats["avg_similarity_score"],
            "vector_weight": self.vector_weight,
            "bm25_weight": self.bm25_weight
        }

# Factory function to create a hybrid retriever
def create_hybrid_retriever(
    vector_retriever=None,
    bm25_retriever=None,
    min_similarity_threshold: float = 0.25,
    use_reciprocal_rank_fusion: bool = True,
    vector_weight: float = 0.7,
    bm25_weight: float = 0.3
) -> HybridRetriever:
    """
    Create a hybrid retriever with the given parameters.
    
    Args:
        vector_retriever: Vector store retriever for semantic search
        bm25_retriever: BM25 retriever for keyword search
        min_similarity_threshold: Minimum similarity threshold (0-1)
        use_reciprocal_rank_fusion: Whether to use RRF for combining results
        vector_weight: Weight for vector similarity scores (0-1)
        bm25_weight: Weight for BM25 scores (0-1)
        
    Returns:
        HybridRetriever instance
    """
    return HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        min_similarity_threshold=min_similarity_threshold,
        use_reciprocal_rank_fusion=use_reciprocal_rank_fusion,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight
    ) 