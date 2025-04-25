import os
import sys
import argparse
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def get_vectorstore():
    """Initialize embeddings and get the vector store."""
    try:
        # Initialize embeddings
        print("Initializing embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        print("Embeddings initialized successfully")
        
        # Check for vector store
        persist_directory = "chroma_db"
        if os.path.exists(persist_directory):
            # Load existing vector store
            print("Loading existing vector store")
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings
            )
            print(f"Loaded vector store with {len(vectorstore.get()['ids'])} documents")
            return vectorstore
        else:
            print("Error: Vector store directory not found")
            return None
    except Exception as e:
        print(f"Error initializing: {e}")
        return None

def query_documents(query, k=3):
    """Run a query against the vector store and print results."""
    # Get the vector store
    vectorstore = get_vectorstore()
    
    if vectorstore is None:
        print("Error: Vector store is not available")
        return
    
    # Run the query
    print(f"\nSearching for: '{query}'")
    docs = vectorstore.similarity_search(query, k=k)
    
    # Display results
    print(f"Found {len(docs)} relevant documents:")
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown")
        print(f"\nResult {i+1} from {source}:")
        print("-" * 50)
        # Display a snippet of the content
        content = doc.page_content.strip()
        if len(content) > 500:
            content = content[:500] + "... (truncated)"
        print(content)
        print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query documents using vector search")
    parser.add_argument("query", nargs="+", help="The search query")
    parser.add_argument("-k", type=int, default=3, help="Number of results to return (default: 3)")
    
    args = parser.parse_args()
    
    # Join the query words into a single string
    query_text = " ".join(args.query)
    
    # Run the query
    query_documents(query_text, k=args.k) 