import os
import sys
import argparse

# Add parent directory to sys.path to allow imports from the main package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mindnest.utils.incremental_vectorstore import IncrementalVectorStore
from mindnest.utils.query_cache import QueryCache

# Initialize the cache
cache = QueryCache()

def get_vectorstore():
    """Initialize the vector store using our optimized implementation."""
    try:
        # Use our optimized incremental vector store implementation
        vector_store = IncrementalVectorStore()
        return vector_store.initialize_or_update()
    except Exception as e:
        print(f"Error initializing vector store: {e}")
        return None

def query_documents(query, k=3):
    """Run a query against the vector store and print results."""
    # Check cache first
    print(f"Searching for: '{query}'")
    cached_results = cache.get(query, k)
    
    if cached_results:
        print("Found results in cache!")
        # Display cached results
        print(f"Found {len(cached_results)} relevant documents:")
        for i, result in enumerate(cached_results):
            source = result.get("metadata", {}).get("source", "Unknown")
            content = result.get("content", "").strip()
            print(f"\nResult {i+1} from {source}:")
            print("-" * 50)
            if len(content) > 500:
                content = content[:500] + "... (truncated)"
            print(content)
            print("-" * 50)
            
        # Print cache statistics
        stats = cache.get_stats()
        print(f"\nCache hit rate: {stats['hit_rate_percent']}%")
        return
    
    # If not in cache, query the vector store
    vectorstore = get_vectorstore()
    
    if vectorstore is None:
        print("Error: Vector store is not available")
        return
    
    # Run the query
    docs = vectorstore.similarity_search(query, k=k)
    
    # Convert to a cacheable format
    cacheable_results = []
    
    # Display results
    print(f"Found {len(docs)} relevant documents:")
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown")
        content = doc.page_content.strip()
        
        # Add to cacheable results
        cacheable_results.append({
            "content": content,
            "metadata": doc.metadata
        })
        
        print(f"\nResult {i+1} from {source}:")
        print("-" * 50)
        # Display a snippet of the content
        if len(content) > 500:
            content = content[:500] + "... (truncated)"
        print(content)
        print("-" * 50)
    
    # Cache the results for future queries
    if cacheable_results:
        cache.set(query, cacheable_results, k)
        print("Results cached for future queries")

def clear_cache():
    """Clear the query cache."""
    cache.clear()
    print("Query cache cleared")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query documents using vector search")
    parser.add_argument("query", nargs="*", help="The search query")
    parser.add_argument("-k", type=int, default=3, help="Number of results to return (default: 3)")
    parser.add_argument("--clear-cache", action="store_true", help="Clear the query cache")
    parser.add_argument("--cache-stats", action="store_true", help="Show cache statistics")
    
    args = parser.parse_args()
    
    if args.clear_cache:
        clear_cache()
        sys.exit(0)
    
    if args.cache_stats:
        stats = cache.get_stats()
        print("Query Cache Statistics:")
        print(f"  Memory hits: {stats['memory_hits']}")
        print(f"  Disk hits: {stats['disk_hits']}")
        print(f"  Misses: {stats['misses']}")
        print(f"  Total queries: {stats['total_queries']}")
        print(f"  Hit rate: {stats['hit_rate_percent']}%")
        print(f"  Memory cache size: {stats['memory_cache_size']}/{stats['memory_max_size']}")
        print(f"  Disk cache size: {stats['disk_cache_size']}/{stats['disk_max_size']}")
        sys.exit(0)
    
    # Check if query is provided
    if not args.query:
        parser.print_help()
        sys.exit(1)
    
    # Join the query words into a single string
    query_text = " ".join(args.query)
    
    # Run the query
    query_documents(query_text, k=args.k) 