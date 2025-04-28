"""
Enhanced vector store for improved document retrieval.

Integrates with hybrid retrieval and reranking components
to provide more accurate and relevant document retrieval.
"""

import os
import shutil
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path

from langchain.schema import Document, BaseRetriever
from langchain_community.vectorstores import Chroma

# Get module logger
logger = logging.getLogger(__name__)

class EnhancedVectorStore:
    """
    Enhanced vector store with improved retrieval quality.
    
    Features:
    - Hybrid retrieval (dense + sparse)
    - Minimum similarity thresholds
    - Cross-encoder reranking
    - Incremental update support
    """
    
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        docs_directory: Optional[Union[str, Path]] = None,
        embedding_model=None,
        min_similarity_threshold: float = 0.25,
        use_hybrid_retrieval: bool = True,
        use_reranking: bool = True
    ):
        """
        Initialize the enhanced vector store.
        
        Args:
            persist_directory: Directory to store vector data
            docs_directory: Directory containing documentation
            embedding_model: Model for creating embeddings
            min_similarity_threshold: Minimum similarity threshold
            use_hybrid_retrieval: Whether to use hybrid retrieval
            use_reranking: Whether to use cross-encoder reranking
        """
        from mindnest.core.config import config
        
        # Use provided values or defaults from config
        self.persist_directory = persist_directory if persist_directory is not None else config.persist_directory
        self.docs_directory = Path(docs_directory) if docs_directory is not None else config.docs_dir
        self.embedding_model = embedding_model
        
        # Retrieval settings
        self.min_similarity_threshold = min_similarity_threshold
        self.use_hybrid_retrieval = use_hybrid_retrieval
        self.use_reranking = use_reranking
        
        # Initialize tracking
        self.track_doc_changes = True
        self.document_tracker = None
        self.document_processor = None
        
        # Initialize retrievers
        self.vector_store = None
        self.bm25_retriever = None
        self.hybrid_retriever = None
        self.reranker = None
        self.retriever = None
        
        # Import only when needed to avoid circular imports
        from mindnest.utils.document_tracker import DocumentTracker
        from mindnest.utils.document_processor import DocumentProcessor
        
        # Initialize document tracker
        self.document_tracker = DocumentTracker(docs_dir=str(self.docs_directory))
        
        # Initialize document processor
        self.document_processor = DocumentProcessor(docs_dir=str(self.docs_directory))
    
    def _initialize_embedding_model(self):
        """Initialize the embedding model if not already provided."""
        if self.embedding_model is not None:
            return
            
        try:
            from mindnest.core.config import config
            from langchain_community.embeddings import HuggingFaceEmbeddings
            
            logger.info(f"Initializing embeddings with model: {config.embedding_model_name}")
            self.embedding_model = HuggingFaceEmbeddings(model_name=config.embedding_model_name)
            logger.info("Embeddings initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing embeddings: {e}")
            try:
                # Fallback to basic embeddings
                from langchain_community.embeddings import FakeEmbeddings
                logger.warning("Falling back to simple embeddings")
                self.embedding_model = FakeEmbeddings(size=384)
            except Exception as e2:
                logger.critical(f"Fatal error initializing embeddings: {e2}")
                raise
    
    def initialize_or_update(self, force_rebuild: bool = False) -> BaseRetriever:
        """
        Initialize or update the vector store and retriever.
        
        Args:
            force_rebuild: Whether to force rebuilding the vector store
            
        Returns:
            Initialized retriever
        """
        logger.info("Initializing enhanced vector store...")
        
        # Initialize embedding model if needed
        if self.embedding_model is None:
            self._initialize_embedding_model()
        
        # Check for document changes if tracking is enabled
        if self.track_doc_changes and self.document_tracker:
            document_changes = self.document_tracker.scan_documents()
            has_changes = any(len(changes) > 0 for changes in document_changes.values())
        else:
            has_changes = False
        
        # Determine if we need to rebuild
        should_rebuild = force_rebuild or not os.path.exists(self.persist_directory)
        
        # Initialize or update vector store
        if should_rebuild:
            logger.info("Rebuilding vector store from scratch")
            self.vector_store = self._full_rebuild()
        elif has_changes:
            logger.info("Updating vector store incrementally")
            self.vector_store = self._incremental_update(document_changes)
        else:
            logger.info("Loading existing vector store")
            self.vector_store = self._load_vector_store()
        
        # Initialize BM25 retriever if using hybrid
        if self.use_hybrid_retrieval:
            self._initialize_bm25_retriever()
        
        # Initialize the retriever pipeline
        self._initialize_retriever_pipeline()
        
        return self.retriever
    
    def _full_rebuild(self):
        """
        Perform a full rebuild of the vector store.
        
        Returns:
            The newly built vector store
        """
        try:
            logger.info("Starting full vector store rebuild")
            
            # Clear existing vector store if it exists
            if os.path.exists(self.persist_directory):
                logger.info(f"Removing existing vector store at {self.persist_directory}")
                shutil.rmtree(self.persist_directory)
            
            # Process all documents in the docs directory
            documents = self.document_processor.process_directory(directory=self.docs_directory)
            
            if not documents:
                logger.warning("No documents found to add to vector store")
                return None
            
            # Create new vector store
            logger.info(f"Creating new vector store with {len(documents)} document chunks")
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding_model,
                persist_directory=self.persist_directory
            )
            
            # Initialize BM25 if using hybrid retrieval
            if self.use_hybrid_retrieval:
                self._initialize_bm25_retriever(documents)
            
            # Persist to disk
            vector_store.persist()
            logger.info("Vector store created and persisted successfully")
            
            return vector_store
            
        except Exception as e:
            logger.error(f"Error rebuilding vector store: {e}")
            raise
    
    def _incremental_update(self, document_changes: Dict[str, List[str]]):
        """
        Perform an incremental update of the vector store.
        
        Args:
            document_changes: Dictionary of document changes
            
        Returns:
            The updated vector store
        """
        try:
            # Load the existing vector store
            logger.info("Loading existing vector store for incremental update")
            vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_model
            )
            
            # Process new and modified files
            new_files = document_changes["new"]
            modified_files = document_changes["modified"]
            deleted_files = document_changes["deleted"]
            
            if new_files or modified_files:
                files_to_process = new_files + modified_files
                logger.info(f"Processing {len(files_to_process)} new or modified files")
                
                # Process documents using document processor
                documents = self.document_processor.process_files(files_to_process)
                
                if documents:
                    logger.info(f"Adding {len(documents)} document chunks to vector store")
                    vector_store.add_documents(documents)
                    
                    # Update BM25 if using hybrid retrieval
                    if self.use_hybrid_retrieval and self.bm25_retriever:
                        logger.info(f"Updating BM25 index with {len(documents)} documents")
                        self.bm25_retriever.add_documents(documents)
                else:
                    logger.warning("No valid documents found in new or modified files")
            
            # Process deleted files
            if deleted_files:
                logger.info(f"Processing {len(deleted_files)} deleted files")
                
                # Get all document IDs to check which ones to delete
                all_ids = vector_store.get()["ids"]
                all_metadatas = vector_store.get()["metadatas"]
                
                # Find IDs of documents from deleted files
                ids_to_delete = []
                for i, metadata in enumerate(all_metadatas):
                    source = metadata.get("source")
                    if source and any(source == deleted_file for deleted_file in deleted_files):
                        ids_to_delete.append(all_ids[i])
                
                if ids_to_delete:
                    logger.info(f"Deleting {len(ids_to_delete)} document chunks from vector store")
                    vector_store.delete(ids_to_delete)
                else:
                    logger.info("No documents to delete from vector store")
            
            # Persist changes
            vector_store.persist()
            logger.info("Incremental update completed and persisted")
            
            return vector_store
            
        except Exception as e:
            logger.error(f"Error in incremental update: {e}")
            # Attempt a full rebuild as a fallback
            logger.warning("Attempting full rebuild as a fallback")
            return self._full_rebuild()
    
    def _load_vector_store(self):
        """
        Load the existing vector store.
        
        Returns:
            The loaded vector store
        """
        try:
            logger.info(f"Loading vector store from {self.persist_directory}")
            vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_model
            )
            
            # Load document count
            doc_count = len(vector_store.get()["ids"])
            logger.info(f"Loaded vector store with {doc_count} documents")
            
            # Load documents for BM25 if using hybrid
            if self.use_hybrid_retrieval:
                self._initialize_bm25_retriever()
            
            return vector_store
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            logger.warning("Attempting recovery by rebuilding")
            return self._full_rebuild()
    
    def _initialize_bm25_retriever(self, documents=None):
        """
        Initialize the BM25 retriever.
        
        Args:
            documents: Documents to index (optional)
        """
        if not self.use_hybrid_retrieval:
            return
            
        try:
            from mindnest.utils.bm25_retriever import create_bm25_retriever
            
            logger.info("Initializing BM25 retriever")
            
            # If documents not provided, load from vector store
            if documents is None and self.vector_store is not None:
                try:
                    # Get documents from vector store
                    results = self.vector_store.get()
                    ids = results["ids"]
                    documents = []
                    
                    # Recreate documents from results
                    for i, doc_id in enumerate(ids):
                        doc = Document(
                            page_content=results["documents"][i],
                            metadata=results["metadatas"][i]
                        )
                        documents.append(doc)
                        
                    logger.info(f"Loaded {len(documents)} documents from vector store for BM25")
                except Exception as e:
                    logger.error(f"Error loading documents from vector store: {e}")
                    documents = []
            
            # Create BM25 retriever
            if documents:
                self.bm25_retriever = create_bm25_retriever(documents=documents)
                logger.info("BM25 retriever initialized with documents")
            else:
                self.bm25_retriever = create_bm25_retriever()
                logger.info("BM25 retriever initialized without documents")
        except Exception as e:
            logger.error(f"Error initializing BM25 retriever: {e}")
            self.bm25_retriever = None
    
    def _initialize_retriever_pipeline(self):
        """Initialize the retriever pipeline with all components."""
        try:
            logger.info("Initializing retriever pipeline")
            
            # Create base vector retriever
            base_retriever = self.vector_store.as_retriever(
                search_kwargs={"k": 10}
            )
            current_retriever = base_retriever
            
            # Add hybrid retrieval if enabled
            if self.use_hybrid_retrieval and self.bm25_retriever:
                from mindnest.utils.hybrid_retriever import create_hybrid_retriever
                
                logger.info("Adding hybrid retrieval")
                hybrid_retriever = create_hybrid_retriever(
                    vector_retriever=base_retriever,
                    bm25_retriever=self.bm25_retriever,
                    min_similarity_threshold=self.min_similarity_threshold,
                    use_reciprocal_rank_fusion=True
                )
                self.hybrid_retriever = hybrid_retriever
                current_retriever = hybrid_retriever
            
            # Add reranking if enabled
            if self.use_reranking:
                from mindnest.utils.cross_encoder_reranker import (
                    create_cross_encoder_reranker, 
                    RerankerRetriever
                )
                
                logger.info("Adding cross-encoder reranking")
                # Create reranker with fallback (don't initialize model yet)
                reranker = create_cross_encoder_reranker(
                    model_name_or_path=None,  # Will initialize on first use
                    use_batched_scoring=True,
                    batch_size=8,
                    normalize_scores=True
                )
                self.reranker = reranker
                
                # Create reranker retriever
                reranker_retriever = RerankerRetriever(
                    base_retriever=current_retriever,
                    reranker=reranker,
                    fetch_k=20,
                    top_k=4
                )
                current_retriever = reranker_retriever
            
            # Set final retriever
            self.retriever = current_retriever
            logger.info("Retriever pipeline initialization complete")
        except Exception as e:
            logger.error(f"Error initializing retriever pipeline: {e}")
            # Fall back to basic retriever
            self.retriever = self.vector_store.as_retriever(
                search_kwargs={"k": 4}
            )
            logger.warning("Falling back to basic retriever")
    
    def get_retriever(self) -> BaseRetriever:
        """
        Get the current retriever.
        
        Returns:
            The configured retriever
        """
        if self.retriever is None:
            self.initialize_or_update()
            
        return self.retriever
    
    def search(
        self, 
        query: str, 
        k: int = 4,
        fetch_k: int = 20,
        threshold: Optional[float] = None
    ) -> List[Document]:
        """
        Search for documents relevant to a query.
        
        Args:
            query: Query string
            k: Number of documents to return
            fetch_k: Number of documents to fetch before filtering
            threshold: Similarity threshold (optional)
            
        Returns:
            List of relevant documents
        """
        # Initialize if needed
        if self.retriever is None:
            self.initialize_or_update()
        
        try:
            # Handle different retriever types
            if self.use_hybrid_retrieval and self.hybrid_retriever:
                # Use hybrid retriever with threshold
                actual_threshold = threshold if threshold is not None else self.min_similarity_threshold
                docs = self.hybrid_retriever._get_relevant_documents(
                    query, 
                    k=k, 
                    fetch_k=fetch_k, 
                    filter_threshold=actual_threshold
                )
            elif self.retriever:
                # Use configured retriever
                docs = self.retriever.get_relevant_documents(query)[:k]
            else:
                # Fallback to basic vector store
                docs = self.vector_store.similarity_search(query, k=k)
            
            logger.info(f"Retrieved {len(docs)} documents for query: {query}")
            return docs
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def get_document_count(self) -> int:
        """Get the number of documents in the vector store."""
        try:
            if self.vector_store:
                return len(self.vector_store.get()["ids"])
            return 0
        except Exception as e:
            logger.error(f"Error getting document count: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store and retrievers.
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            "document_count": self.get_document_count(),
            "min_similarity_threshold": self.min_similarity_threshold,
            "use_hybrid_retrieval": self.use_hybrid_retrieval,
            "use_reranking": self.use_reranking
        }
        
        # Add hybrid stats if available
        if self.hybrid_retriever:
            stats["hybrid_retriever"] = self.hybrid_retriever.get_stats()
            
        # Add BM25 stats if available
        if self.bm25_retriever:
            stats["bm25_retriever"] = self.bm25_retriever.get_stats()
            
        # Add reranker stats if available
        if self.reranker:
            stats["reranker"] = self.reranker.get_stats()
            
        return stats

# Factory function to create an enhanced vector store
def create_enhanced_vectorstore(
    persist_directory: Optional[str] = None,
    docs_directory: Optional[Union[str, Path]] = None,
    embedding_model=None,
    min_similarity_threshold: float = 0.25,
    use_hybrid_retrieval: bool = True,
    use_reranking: bool = True
) -> EnhancedVectorStore:
    """
    Create an enhanced vector store.
    
    Args:
        persist_directory: Directory to store vector data
        docs_directory: Directory containing documentation
        embedding_model: Model for creating embeddings
        min_similarity_threshold: Minimum similarity threshold
        use_hybrid_retrieval: Whether to use hybrid retrieval
        use_reranking: Whether to use cross-encoder reranking
        
    Returns:
        EnhancedVectorStore instance
    """
    return EnhancedVectorStore(
        persist_directory=persist_directory,
        docs_directory=docs_directory,
        embedding_model=embedding_model,
        min_similarity_threshold=min_similarity_threshold,
        use_hybrid_retrieval=use_hybrid_retrieval,
        use_reranking=use_reranking
    ) 