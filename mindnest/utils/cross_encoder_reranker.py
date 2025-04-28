"""
Cross-encoder reranker for improved retrieval precision.

Implements a cross-encoder model to re-rank retrieval results
for higher precision document retrieval.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from langchain.schema import Document, BaseRetriever

# Setup logging
logger = logging.getLogger(__name__)

class CrossEncoderReranker:
    """
    Cross-encoder reranker for more accurate document-query relevance.
    Uses a transformer cross-encoder model to score document-query pairs.
    """
    
    def __init__(
        self, 
        cross_encoder_model=None,
        use_batched_scoring: bool = True,
        batch_size: int = 8,
        normalize_scores: bool = True,
        cache_scores: bool = True,
        cache_size: int = 1024
    ):
        """
        Initialize the cross-encoder reranker.
        
        Args:
            cross_encoder_model: Cross-encoder model for scoring (should have a predict method)
            use_batched_scoring: Whether to use batched scoring for efficiency
            batch_size: Batch size for scoring
            normalize_scores: Whether to normalize scores to 0-1 range
            cache_scores: Whether to cache scores for repeated queries
            cache_size: Size of the score cache
        """
        self.cross_encoder = cross_encoder_model
        self.use_batched_scoring = use_batched_scoring
        self.batch_size = batch_size
        self.normalize_scores = normalize_scores
        self.cache_scores = cache_scores
        
        # Score cache
        self._score_cache = {}
        self._cache_size = cache_size
        
        # Indicate if initialized with model
        self.is_initialized = cross_encoder_model is not None
        
        # Track usage stats
        self.stats = {
            "total_rerankings": 0,
            "total_documents_reranked": 0,
            "cache_hits": 0,
            "batch_count": 0
        }
    
    def initialize(self, model_name_or_path: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the cross-encoder model.
        
        Args:
            model_name_or_path: Cross-encoder model name or path
        """
        if self.is_initialized:
            logger.info("Cross-encoder already initialized")
            return
            
        try:
            # Import only if needed to avoid dependencies
            from sentence_transformers.cross_encoder import CrossEncoder
            
            logger.info(f"Initializing cross-encoder model: {model_name_or_path}")
            self.cross_encoder = CrossEncoder(model_name_or_path)
            self.is_initialized = True
            logger.info("Cross-encoder initialization complete")
        except Exception as e:
            logger.error(f"Error initializing cross-encoder: {e}")
            # Create a fallback scoring function
            logger.warning("Using fallback scoring function (simple position-based)")
            self._fallback_scoring()
    
    def _fallback_scoring(self):
        """Create a fallback scoring function when model isn't available."""
        def fallback_predict(pairs):
            # Simple fallback using document position as score
            if isinstance(pairs, list):
                # For batched input
                return [1.0 - (i * 0.1) for i in range(len(pairs))]
            else:
                # For single input
                return 0.8
                
        # Create a mock cross-encoder with predict method
        class FallbackEncoder:
            def predict(self, pairs):
                return fallback_predict(pairs)
                
        self.cross_encoder = FallbackEncoder()
        self.is_initialized = True
        logger.warning("Using fallback scoring (no model)")
    
    def _get_cache_key(self, query: str, doc_id: str) -> str:
        """Create a cache key for a query-document pair."""
        return f"{hash(query)}_{hash(doc_id)}"
    
    def _check_cache(self, query: str, doc_id: str) -> Optional[float]:
        """Check if score is in cache."""
        if not self.cache_scores:
            return None
            
        cache_key = self._get_cache_key(query, doc_id)
        score = self._score_cache.get(cache_key)
        
        if score is not None:
            self.stats["cache_hits"] += 1
            
        return score
    
    def _update_cache(self, query: str, doc_id: str, score: float) -> None:
        """Update score cache."""
        if not self.cache_scores:
            return
            
        # Manage cache size
        if len(self._score_cache) >= self._cache_size:
            # Simple strategy: clear oldest 10% of cache
            num_to_remove = self._cache_size // 10
            for _ in range(num_to_remove):
                if self._score_cache:
                    self._score_cache.pop(next(iter(self._score_cache)))
        
        # Add to cache
        cache_key = self._get_cache_key(query, doc_id)
        self._score_cache[cache_key] = score
    
    def rerank(
        self, 
        query: str, 
        documents: List[Document], 
        top_n: Optional[int] = None
    ) -> List[Document]:
        """
        Rerank documents using the cross-encoder model.
        
        Args:
            query: Query string
            documents: List of documents to rerank
            top_n: Number of top documents to return (None for all)
            
        Returns:
            Reranked list of documents
        """
        if not documents:
            return []
            
        # Ensure model is initialized
        if not self.is_initialized:
            self.initialize()
        
        # Use all documents if top_n not specified
        top_n = top_n or len(documents)
        
        # Update stats
        self.stats["total_rerankings"] += 1
        self.stats["total_documents_reranked"] += len(documents)
        
        # Check cache for scores first
        scored_docs = []
        documents_to_score = []
        doc_indices = []
        
        for i, doc in enumerate(documents):
            # Try to get document ID for cache lookup
            doc_id = doc.metadata.get("source", str(id(doc)))
            
            # Check if score is in cache
            cached_score = self._check_cache(query, doc_id)
            
            if cached_score is not None:
                # Use cached score
                doc.metadata["cross_encoder_score"] = cached_score
                scored_docs.append((i, doc, cached_score))
            else:
                # Add to list for scoring
                documents_to_score.append(doc)
                doc_indices.append(i)
        
        # Score documents that need scoring
        if documents_to_score:
            if self.use_batched_scoring:
                # Batch score documents
                batch_scores = self._batch_score(query, documents_to_score)
                
                # Process scores
                for i, (doc_idx, doc, score) in enumerate(zip(doc_indices, documents_to_score, batch_scores)):
                    # Get document ID for cache
                    doc_id = doc.metadata.get("source", str(id(doc)))
                    
                    # Update cache
                    self._update_cache(query, doc_id, score)
                    
                    # Add score to metadata
                    doc.metadata["cross_encoder_score"] = score
                    
                    # Add to scored documents
                    scored_docs.append((doc_idx, doc, score))
            else:
                # Score documents individually
                for doc_idx, doc in zip(doc_indices, documents_to_score):
                    # Get document ID for cache
                    doc_id = doc.metadata.get("source", str(id(doc)))
                    
                    # Score document
                    score = self._score_document(query, doc)
                    
                    # Update cache
                    self._update_cache(query, doc_id, score)
                    
                    # Add score to metadata
                    doc.metadata["cross_encoder_score"] = score
                    
                    # Add to scored documents
                    scored_docs.append((doc_idx, doc, score))
        
        # Sort by score (descending)
        scored_docs.sort(key=lambda x: x[2], reverse=True)
        
        # Update original scores if available
        for _, doc, cross_encoder_score in scored_docs:
            # Update combined score if present
            if "score" in doc.metadata:
                # Calculate a weighted average with original score
                original_score = doc.metadata["score"]
                # Higher weight to cross-encoder score as it's more accurate
                doc.metadata["score"] = (cross_encoder_score * 0.7) + (original_score * 0.3)
        
        # Return sorted documents (up to top_n)
        return [doc for _, doc, _ in scored_docs[:top_n]]
    
    def _score_document(self, query: str, document: Document) -> float:
        """
        Score a single document with the cross-encoder.
        
        Args:
            query: Query string
            document: Document to score
            
        Returns:
            Relevance score
        """
        try:
            # Get text content from document
            text = document.page_content
            
            # Create input pair for cross-encoder
            input_pair = [query, text]
            
            # Get score from cross-encoder
            score = self.cross_encoder.predict(input_pair)
            
            # Normalize score if needed
            if self.normalize_scores and score > 1.0:
                # Sigmoid normalization for scores outside 0-1
                import math
                score = 1.0 / (1.0 + math.exp(-score))
                
            return float(score)
        except Exception as e:
            logger.error(f"Error scoring document: {e}")
            return 0.5  # Default score on error
    
    def _batch_score(self, query: str, documents: List[Document]) -> List[float]:
        """
        Score multiple documents in batches.
        
        Args:
            query: Query string
            documents: List of documents to score
            
        Returns:
            List of scores
        """
        try:
            # Create input pairs for cross-encoder
            pairs = [[query, doc.page_content] for doc in documents]
            
            # Split into batches
            batches = [pairs[i:i + self.batch_size] for i in range(0, len(pairs), self.batch_size)]
            
            # Update batch count stat
            self.stats["batch_count"] += len(batches)
            
            # Score each batch and combine results
            all_scores = []
            for batch in batches:
                # Get scores from cross-encoder
                batch_scores = self.cross_encoder.predict(batch)
                
                # Handle both list and numpy array returns
                if hasattr(batch_scores, "tolist"):
                    batch_scores = batch_scores.tolist()
                
                # Add to all scores
                all_scores.extend(batch_scores)
            
            # Normalize scores if needed
            if self.normalize_scores:
                # Check if any scores are outside 0-1 range
                if any(score > 1.0 or score < 0.0 for score in all_scores):
                    # Find max for min-max normalization
                    max_score = max(all_scores)
                    min_score = min(all_scores)
                    score_range = max_score - min_score
                    
                    if score_range > 0:
                        all_scores = [(score - min_score) / score_range for score in all_scores]
                    else:
                        # Default to 0.5 if all scores are the same
                        all_scores = [0.5 for _ in all_scores]
            
            return all_scores
        except Exception as e:
            logger.error(f"Error batch scoring documents: {e}")
            return [0.5] * len(documents)  # Default scores on error
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the reranker.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "total_rerankings": self.stats["total_rerankings"],
            "total_documents_reranked": self.stats["total_documents_reranked"],
            "cache_hits": self.stats["cache_hits"],
            "cache_size": len(self._score_cache),
            "max_cache_size": self._cache_size,
            "batch_count": self.stats["batch_count"],
            "model_initialized": self.is_initialized,
            "batch_size": self.batch_size
        }

class RerankerRetriever(BaseRetriever):
    """
    Retriever that wraps another retriever and reranks results.
    """
    
    def __init__(
        self,
        base_retriever: BaseRetriever,
        reranker: CrossEncoderReranker,
        fetch_k: int = 20,
        top_k: int = 4
    ):
        """
        Initialize the reranker retriever.
        
        Args:
            base_retriever: Base retriever to wrap
            reranker: Reranker to use for reranking
            fetch_k: Number of documents to fetch from base retriever
            top_k: Number of reranked documents to return
        """
        super().__init__()
        self.base_retriever = base_retriever
        self.reranker = reranker
        self.fetch_k = fetch_k
        self.top_k = top_k
    
    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager=None
    ) -> List[Document]:
        """
        Get relevant documents for a query using reranking.
        
        Args:
            query: The query string
            run_manager: Run manager for callbacks
            
        Returns:
            List of relevant documents
        """
        # Get documents from base retriever
        docs = self.base_retriever.get_relevant_documents(query, run_manager=run_manager)
        
        # Limit to fetch_k
        docs = docs[:self.fetch_k]
        
        # Rerank documents
        reranked_docs = self.reranker.rerank(query, docs, top_n=self.top_k)
        
        return reranked_docs

# Factory function to create a cross-encoder reranker
def create_cross_encoder_reranker(
    model_name_or_path: Optional[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    use_batched_scoring: bool = True,
    batch_size: int = 8,
    normalize_scores: bool = True,
    cache_scores: bool = True,
    cache_size: int = 1024
) -> CrossEncoderReranker:
    """
    Create a cross-encoder reranker.
    
    Args:
        model_name_or_path: Cross-encoder model name or path (None to initialize later)
        use_batched_scoring: Whether to use batched scoring
        batch_size: Batch size for scoring
        normalize_scores: Whether to normalize scores
        cache_scores: Whether to cache scores
        cache_size: Size of the score cache
        
    Returns:
        CrossEncoderReranker instance
    """
    reranker = CrossEncoderReranker(
        cross_encoder_model=None,
        use_batched_scoring=use_batched_scoring,
        batch_size=batch_size,
        normalize_scores=normalize_scores,
        cache_scores=cache_scores,
        cache_size=cache_size
    )
    
    # Initialize with model if provided
    if model_name_or_path:
        reranker.initialize(model_name_or_path)
    
    return reranker 