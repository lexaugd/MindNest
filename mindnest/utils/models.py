"""
Data models for the MindNest application.
Contains Pydantic models for API requests and responses.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class Question(BaseModel):
    """Model for question requests."""
    text: str = Field(..., description="The question text")

class SourceDocument(BaseModel):
    """Model for source document information."""
    source: str = Field(..., description="Source file path")
    content: str = Field(..., description="Document content")
    relevance_score: Optional[float] = Field(None, description="Relevance score")

class Answer(BaseModel):
    """Model for answer responses."""
    text: str = Field(..., description="The answer text")
    sources: List[str] = Field(default_factory=list, description="Source file paths")
    source_documents: Optional[List[SourceDocument]] = Field(
        default_factory=list,
        description="More detailed information about source documents"
    )
    query_type: Optional[str] = Field(None, description="Type of query that was processed")
    processing_time: Optional[float] = Field(None, description="Time taken to process the query in ms")

class DocumentInfo(BaseModel):
    """Model for document information."""
    filename: str = Field(..., description="Document filename")
    path: Optional[str] = Field(None, description="Document path")
    file_type: Optional[str] = Field(None, description="Document file type")
    chunk_count: int = Field(0, description="Number of chunks in the document")
    last_updated: Optional[str] = Field(None, description="Last update timestamp")

class DocumentsList(BaseModel):
    """Model for list of documents."""
    documents: List[DocumentInfo] = Field(default_factory=list, description="List of documents")
    total_count: int = Field(0, description="Total number of documents")
    total_chunks: int = Field(0, description="Total number of document chunks")

class ModelConfig(BaseModel):
    """Model for LLM configuration."""
    use_small_model: bool = Field(False, description="Whether to use the smaller LLM")
    max_context_tokens: int = Field(800, description="Maximum context tokens for LLM")
    conversation_mode: Optional[str] = Field(
        "professional", 
        description="Conversation mode (professional, friendly, concise)"
    )
    query_classifier_mode: Optional[str] = Field(
        "embeddings",
        description="Query classifier mode (embeddings, neural, hybrid, regex)"
    )

class CacheStats(BaseModel):
    """Model for cache statistics."""
    memory_hits: int = Field(0, description="Number of memory cache hits")
    disk_hits: int = Field(0, description="Number of disk cache hits")
    misses: int = Field(0, description="Number of cache misses")
    total_queries: int = Field(0, description="Total number of queries")
    hit_rate_percent: float = Field(0.0, description="Cache hit rate percentage")
    memory_cache_size: int = Field(0, description="Current memory cache size")
    memory_max_size: int = Field(0, description="Maximum memory cache size")
    disk_cache_size: int = Field(0, description="Current disk cache size")
    disk_cache_enabled: bool = Field(False, description="Whether disk cache is enabled")

class SystemStats(BaseModel):
    """Model for system statistics."""
    vector_store_documents: int = Field(0, description="Number of documents in vector store")
    document_count: int = Field(0, description="Number of original documents")
    file_extensions: Dict[str, int] = Field(default_factory=dict, description="Count of documents by file extension")
    model_name: str = Field("", description="Current LLM name")
    embedding_model: str = Field("", description="Current embedding model name")
    cache_stats: Optional[CacheStats] = Field(None, description="Cache statistics")
    query_stats: Optional[Dict[str, Any]] = Field(None, description="Query statistics")

class HealthCheck(BaseModel):
    """Model for health check response."""
    status: str = Field("healthy", description="System status")
    llm: str = Field("", description="LLM status")
    vectorstore: str = Field("", description="Vector store status")
    model: str = Field("", description="Current model name")
    uptime: Optional[float] = Field(None, description="Server uptime in seconds") 