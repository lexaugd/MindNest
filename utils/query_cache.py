import json
import os
import time
from typing import Dict, List, Any, Optional, Tuple
from functools import lru_cache
from datetime import datetime, timedelta

class QueryCache:
    """
    Implements a caching mechanism for document queries to improve response time.
    Both memory-based (LRU) and disk-based caching are supported.
    """
    
    def __init__(
        self, 
        cache_file: str = "utils/query_cache.json",
        max_memory_items: int = 100,
        max_disk_items: int = 1000,
        ttl_hours: float = 24.0
    ):
        """
        Initialize the query cache.
        
        Args:
            cache_file: File to store persistent cache
            max_memory_items: Maximum number of items to keep in memory cache
            max_disk_items: Maximum number of items to keep in disk cache 
            ttl_hours: Time-to-live for cache entries in hours
        """
        self.cache_file = cache_file
        self.max_memory_items = max_memory_items
        self.max_disk_items = max_disk_items
        self.ttl_seconds = ttl_hours * 3600
        
        # Statistics
        self.stats = {
            "memory_hits": 0,
            "disk_hits": 0,
            "misses": 0,
            "total_queries": 0
        }
        
        # Load disk cache if it exists
        self.disk_cache = self._load_disk_cache()
        
        # Memory cache (simple dict for now)
        self.memory_cache = {}
    
    def _load_disk_cache(self) -> Dict:
        """Load the disk cache from file or create a new one."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                    # Perform cache cleanup on load
                    self._clean_expired_entries(cache_data)
                    return cache_data
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading cache file: {e}. Creating new cache.")
        
        # Create cache structure if it doesn't exist or can't be loaded
        return {
            "metadata": {
                "created": datetime.now().isoformat(),
                "last_cleanup": datetime.now().isoformat()
            },
            "queries": {}
        }
    
    def _save_disk_cache(self) -> None:
        """Save the disk cache to file."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.disk_cache, f, indent=2)
        except IOError as e:
            print(f"Error saving cache file: {e}")
    
    def _clean_expired_entries(self, cache_data: Dict) -> None:
        """Remove expired entries from the cache."""
        now = time.time()
        expired_keys = []
        
        for key, entry in cache_data["queries"].items():
            # Check if the entry has expired
            if entry["timestamp"] + self.ttl_seconds < now:
                expired_keys.append(key)
        
        # Remove expired entries
        for key in expired_keys:
            del cache_data["queries"][key]
        
        # Update last cleanup timestamp
        cache_data["metadata"]["last_cleanup"] = datetime.now().isoformat()
        
        # Trim cache if it exceeds max size
        self._trim_cache_if_needed(cache_data)
    
    def _trim_cache_if_needed(self, cache_data: Dict) -> None:
        """Trim the cache if it exceeds the maximum size."""
        if len(cache_data["queries"]) > self.max_disk_items:
            # Sort entries by access time (oldest first)
            sorted_entries = sorted(
                cache_data["queries"].items(),
                key=lambda x: x[1]["last_accessed"]
            )
            
            # Remove oldest entries until we're under the limit
            entries_to_remove = len(cache_data["queries"]) - self.max_disk_items
            for i in range(entries_to_remove):
                if i < len(sorted_entries):
                    del cache_data["queries"][sorted_entries[i][0]]
    
    def _normalize_query(self, query: str, k: int = 3) -> str:
        """
        Normalize the query string to enable better cache hits.
        
        Args:
            query: The query string
            k: Number of results to retrieve
            
        Returns:
            Normalized query string for cache lookup
        """
        # Basic normalization: lowercase, strip whitespace
        normalized = query.lower().strip()
        # Create a cache key that includes the k parameter
        return f"{normalized}:{k}"
    
    def _trim_memory_cache_if_needed(self) -> None:
        """Trim the memory cache if it exceeds the maximum size."""
        if len(self.memory_cache) > self.max_memory_items:
            # Sort entries by access time (oldest first)
            sorted_entries = sorted(
                self.memory_cache.items(),
                key=lambda x: x[1]["last_accessed"]
            )
            
            # Remove oldest entries until we're under the limit
            entries_to_remove = len(self.memory_cache) - self.max_memory_items
            for i in range(entries_to_remove):
                if i < len(sorted_entries):
                    del self.memory_cache[sorted_entries[i][0]]
    
    def get(self, query: str, k: int = 3) -> Optional[List[Dict]]:
        """
        Get a query result from cache.
        
        Args:
            query: The query string
            k: Number of results to retrieve
            
        Returns:
            The cached query result or None if not found
        """
        self.stats["total_queries"] += 1
        
        # Normalize the query
        normalized_query = self._normalize_query(query, k)
        
        # Try memory cache first
        if normalized_query in self.memory_cache:
            entry = self.memory_cache[normalized_query]
            # Check if the entry has expired
            if entry["timestamp"] + self.ttl_seconds >= time.time():
                # Update access time
                entry["last_accessed"] = time.time()
                self.stats["memory_hits"] += 1
                return entry["result"]
            else:
                # Remove expired entry from memory cache
                del self.memory_cache[normalized_query]
        
        # Try disk cache
        if normalized_query in self.disk_cache["queries"]:
            entry = self.disk_cache["queries"][normalized_query]
            
            # Check if the entry has expired
            if entry["timestamp"] + self.ttl_seconds >= time.time():
                # Update access time
                entry["last_accessed"] = time.time()
                self._save_disk_cache()
                
                # Store in memory cache for future use
                self.memory_cache[normalized_query] = {
                    "result": entry["result"],
                    "timestamp": entry["timestamp"],
                    "last_accessed": time.time()
                }
                
                # Trim memory cache if needed
                self._trim_memory_cache_if_needed()
                
                self.stats["disk_hits"] += 1
                return entry["result"]
            else:
                # Remove expired entry from disk cache
                del self.disk_cache["queries"][normalized_query]
                self._save_disk_cache()
        
        # Cache miss
        self.stats["misses"] += 1
        return None
    
    def set(self, query: str, result: List[Dict], k: int = 3) -> None:
        """
        Store a query result in cache.
        
        Args:
            query: The query string
            result: The query result to cache
            k: Number of results that were retrieved
        """
        # Normalize the query
        normalized_query = self._normalize_query(query, k)
        now = time.time()
        
        # Store in memory cache
        self.memory_cache[normalized_query] = {
            "result": result,
            "timestamp": now,
            "last_accessed": now
        }
        
        # Trim memory cache if needed
        self._trim_memory_cache_if_needed()
        
        # Store in disk cache
        self.disk_cache["queries"][normalized_query] = {
            "result": result,
            "timestamp": now,
            "last_accessed": now
        }
        
        # Save disk cache
        self._save_disk_cache()
        
        # Perform cleanup periodically
        last_cleanup = datetime.fromisoformat(self.disk_cache["metadata"]["last_cleanup"])
        if datetime.now() - last_cleanup > timedelta(hours=1):
            self._clean_expired_entries(self.disk_cache)
            self._save_disk_cache()
    
    def clear(self) -> None:
        """Clear all caches."""
        # Clear memory cache
        self.memory_cache = {}
        
        # Clear disk cache
        self.disk_cache["queries"] = {}
        self.disk_cache["metadata"]["last_cleanup"] = datetime.now().isoformat()
        self._save_disk_cache()
        
        print("Query cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        # Update cache sizes for accurate reporting
        memory_size = len(self.memory_cache)
        disk_size = len(self.disk_cache["queries"])
        
        total = self.stats["total_queries"]
        hit_rate = 0
        if total > 0:
            hit_rate = (self.stats["memory_hits"] + self.stats["disk_hits"]) / total * 100
            
        return {
            "memory_hits": self.stats["memory_hits"],
            "disk_hits": self.stats["disk_hits"],
            "misses": self.stats["misses"],
            "total_queries": total,
            "hit_rate_percent": round(hit_rate, 2),
            "memory_cache_size": memory_size,
            "disk_cache_size": disk_size,
            "memory_max_size": self.max_memory_items,
            "disk_max_size": self.max_disk_items,
            "ttl_hours": self.ttl_seconds / 3600
        }


# For testing purposes
if __name__ == "__main__":
    # Create a new cache
    cache = QueryCache()
    
    # Test storing and retrieving items
    test_query = "What is a document store?"
    test_result = [
        {"content": "Document stores are NoSQL databases...", "metadata": {"source": "docs/test.txt"}},
        {"content": "A document-oriented database...", "metadata": {"source": "docs/sample.md"}}
    ]
    
    # Store the result
    cache.set(test_query, test_result)
    
    # Retrieve the result
    retrieved = cache.get(test_query)
    print(f"Retrieved from cache: {retrieved is not None}")
    if retrieved:
        print(f"First result: {retrieved[0]['content'][:30]}...")
    
    # Test with a query that's not in the cache
    missing = cache.get("This query doesn't exist")
    print(f"Missing query returns: {missing}")
    
    # Print statistics
    print(f"Cache statistics: {cache.get_stats()}") 