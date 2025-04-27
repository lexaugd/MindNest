"""
Advanced cache for query results to improve response times.
Provides both in-memory LRU cache and disk persistence.
"""

import os
import json
import time
import hashlib
from typing import Dict, List, Any, Optional, Union
from collections import OrderedDict

from utils.config import config
from utils.logger import get_logger

# Get module logger
logger = get_logger(__name__)

class QueryCache:
    """
    Advanced cache for storing query results with both memory and disk caching.
    Implements LRU (Least Recently Used) for memory caching.
    """
    
    def __init__(
        self,
        max_memory_size: Optional[int] = None,
        disk_cache_enabled: Optional[bool] = None,
        disk_cache_path: Optional[str] = None
    ):
        """
        Initialize cache with optional configuration.
        
        Args:
            max_memory_size: Maximum number of items in memory cache (default: from config)
            disk_cache_enabled: Whether to enable disk caching (default: from config)
            disk_cache_path: Path to disk cache file (default: from config)
        """
        # Use provided values or defaults from config
        self.max_memory_size = max_memory_size if max_memory_size is not None else config.memory_cache_size
        self.disk_cache_enabled = disk_cache_enabled if disk_cache_enabled is not None else config.disk_cache_enabled
        self.disk_cache_path = disk_cache_path if disk_cache_path is not None else config.disk_cache_path
        
        # LRU cache using OrderedDict
        self.cache = OrderedDict()
        
        # Disk cache
        self.disk_cache = {}
        
        # Statistics
        self.memory_hits = 0
        self.disk_hits = 0
        self.misses = 0
        
        # Load disk cache if enabled
        if self.disk_cache_enabled:
            self._load_disk_cache()
    
    def _load_disk_cache(self) -> None:
        """Load the disk cache from disk."""
        if not os.path.exists(self.disk_cache_path):
            return
            
        try:
            with open(self.disk_cache_path, 'r') as f:
                self.disk_cache = json.load(f)
            logger.info(f"Loaded {len(self.disk_cache)} items from disk cache")
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading disk cache: {e}")
            self.disk_cache = {}
    
    def _save_disk_cache(self) -> None:
        """Save the disk cache to disk."""
        if not self.disk_cache_enabled:
            return
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.disk_cache_path), exist_ok=True)
            
            with open(self.disk_cache_path, 'w') as f:
                json.dump(self.disk_cache, f)
            logger.debug(f"Saved {len(self.disk_cache)} items to disk cache")
        except IOError as e:
            logger.error(f"Error saving disk cache: {e}")
    
    def _hash_key(self, key: str) -> str:
        """Create a hash of the key for disk storage."""
        return hashlib.md5(key.encode('utf-8')).hexdigest()
    
    def __contains__(self, key: str) -> bool:
        """Check if a key is in the cache (memory or disk)."""
        # Check memory cache first
        if key in self.cache:
            return True
            
        # Then check disk cache if enabled
        if self.disk_cache_enabled:
            hashed_key = self._hash_key(key)
            return hashed_key in self.disk_cache
            
        return False
    
    def get(self, key: str, default: Any = None, k: Optional[int] = None) -> Any:
        """
        Get a value from the cache.
        
        Args:
            key: The cache key
            default: Default value if key not found
            k: Number of results (not used in retrieval, just for stats)
            
        Returns:
            The cached value or default
        """
        # Check memory cache first
        if key in self.cache:
            # Update LRU order by moving to end
            value = self.cache.pop(key)
            self.cache[key] = value
            self.memory_hits += 1
            logger.debug(f"Memory cache hit: {key}")
            return value
            
        # Check disk cache if enabled
        if self.disk_cache_enabled:
            hashed_key = self._hash_key(key)
            if hashed_key in self.disk_cache:
                # Load from disk cache and promote to memory cache
                value = self.disk_cache[hashed_key]
                
                # Add to memory cache (which may trigger eviction)
                self.set(key, value, k)
                
                self.disk_hits += 1
                logger.debug(f"Disk cache hit: {key}")
                return value
                
        # Cache miss
        self.misses += 1
        logger.debug(f"Cache miss: {key}")
        return default
    
    def set(self, key: str, value: Any, k: Optional[int] = None) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: The cache key
            value: The value to cache
            k: Number of results (not used in storage, just for stats)
        """
        # Check if key already exists in memory cache
        if key in self.cache:
            # Remove existing entry
            self.cache.pop(key)
            
        # Add to memory cache
        self.cache[key] = value
        
        # Evict if memory cache is full
        if len(self.cache) > self.max_memory_size:
            # LRU eviction: remove oldest item
            oldest_key, oldest_value = self.cache.popitem(last=False)
            logger.debug(f"LRU eviction of key: {oldest_key}")
            
            # Store evicted item in disk cache if enabled
            if self.disk_cache_enabled:
                hashed_key = self._hash_key(oldest_key)
                self.disk_cache[hashed_key] = oldest_value
                # Periodically save disk cache
                if len(self.disk_cache) % 10 == 0:
                    self._save_disk_cache()
                
        # Also store in disk cache if enabled
        if self.disk_cache_enabled:
            hashed_key = self._hash_key(key)
            self.disk_cache[hashed_key] = value
            
            # Save disk cache every 10 writes to avoid excessive I/O
            if len(self.disk_cache) % 10 == 0:
                self._save_disk_cache()
    
    def clear(self) -> None:
        """Clear both memory and disk caches."""
        # Clear memory cache
        self.cache.clear()
        
        # Clear disk cache
        if self.disk_cache_enabled:
            self.disk_cache.clear()
            self._save_disk_cache()
            
        # Reset statistics
        self.memory_hits = 0
        self.disk_hits = 0
        self.misses = 0
        
        logger.info("Query cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_queries = self.memory_hits + self.disk_hits + self.misses
        hit_rate = ((self.memory_hits + self.disk_hits) / total_queries * 100) if total_queries > 0 else 0
        
        return {
            "memory_hits": self.memory_hits,
            "disk_hits": self.disk_hits,
            "misses": self.misses,
            "total_queries": total_queries,
            "hit_rate_percent": round(hit_rate, 2),
            "memory_cache_size": len(self.cache),
            "memory_max_size": self.max_memory_size,
            "disk_cache_size": len(self.disk_cache),
            "disk_cache_enabled": self.disk_cache_enabled
        }


# For testing purposes
if __name__ == "__main__":
    # Create a new cache with small memory size to test eviction
    cache = QueryCache(max_memory_size=3)
    
    # Test storing and retrieving items
    test_queries = [
        "What is a document store?",
        "How to implement vector search?",
        "What is a language model?",
        "How to optimize query performance?",
        "What is document chunking?"
    ]
    
    # Store test items
    for i, query in enumerate(test_queries):
        test_result = [
            {"content": f"Test content for {query}...", "metadata": {"source": f"docs/test{i}.txt"}},
            {"content": f"More info about {query}...", "metadata": {"source": f"docs/sample{i}.md"}}
        ]
        cache.set(query, test_result)
        print(f"Added to cache: {query}")
    
    # Retrieve items (first should be evicted)
    for query in test_queries:
        result = cache.get(query)
        if result:
            print(f"Cache hit for '{query}': {result[0]['content'][:20]}...")
        else:
            print(f"Cache miss for '{query}'")
    
    # Print statistics
    print(f"\nCache statistics: {cache.get_stats()}") 