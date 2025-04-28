"""
Configuration management for MindNest application
"""

import os
from pathlib import Path
from typing import Optional, Any, Dict

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class Config:
    """Central configuration manager for the MindNest application."""
    
    def __init__(self):
        """Initialize configuration with default values and environment overrides."""
        # Base paths
        self.base_dir = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.models_dir = self.base_dir / "models"
        self.docs_dir = self.base_dir / "docs"
        self.static_dir = self.base_dir / "static"
        
        # Vector store settings
        self.persist_directory = os.environ.get("VECTORSTORE_DIR", "chroma_db")
        self.embedding_model_name = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        
        # Document processing settings
        self.chunk_size = int(os.environ.get("CHUNK_SIZE", "1000"))
        self.chunk_overlap = int(os.environ.get("CHUNK_OVERLAP", "200"))
        
        # LLM settings
        self.model_name = os.environ.get("MODEL_NAME", "Wizard-Vicuna-13B-Uncensored.Q4_K_M.gguf")
        self.small_model_name = os.environ.get("SMALL_MODEL_NAME", "llama-2-7b.Q4_K_M.gguf")
        self.use_small_model = os.environ.get("USE_SMALL_MODEL", "false").lower() == "true"
        
        # Performance settings
        self.temperature = float(os.environ.get("TEMPERATURE", "0.3"))
        self.max_tokens = int(os.environ.get("MAX_TOKENS", "2000"))
        self.top_p = float(os.environ.get("TOP_P", "0.95"))
        self.context_window = int(os.environ.get("CONTEXT_WINDOW", "4096"))
        self.batch_size = int(os.environ.get("BATCH_SIZE", "1024"))
        
        # Cache settings
        self.cache_dir = self.base_dir / "cache"
        self.enable_disk_cache = os.environ.get("ENABLE_DISK_CACHE", "true").lower() == "true"
        self.max_cache_size = int(os.environ.get("MAX_CACHE_SIZE", "1000"))
        
        # Application settings
        self.host = os.environ.get("HOST", "0.0.0.0")
        self.port = int(os.environ.get("PORT", "8000"))
    
    def get_model_path(self, use_small=None):
        """Get the path to the model file based on configuration."""
        if use_small is None:
            use_small = self.use_small_model
        
        model_file = self.small_model_name if use_small else self.model_name
        return os.path.join(self.models_dir, model_file)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

# Create global config instance
config = Config()

# System configuration variables (kept for backward compatibility)
max_context_tokens = 800  # Cap context for faster responses
conversation_mode = "professional"  # Default to professional mode
query_classifier_mode = "zero-shot"  # Use zero-shot classification by default
query_classifier = None

# Update functions
def update_max_context_tokens(value: int) -> None:
    """Update max context tokens value."""
    global max_context_tokens
    max_context_tokens = value

def update_conversation_mode(value: str) -> None:
    """Update conversation mode."""
    global conversation_mode
    conversation_mode = value

def update_query_classifier_mode(value: str) -> None:
    """Update query classifier mode."""
    global query_classifier_mode, query_classifier
    query_classifier_mode = value
    
    # Reset the classifier to force reinitialization
    if query_classifier is not None:
        query_classifier.classifier_type = value
        query_classifier.classifier = None  # Force reinitialization

def initialize_query_classifier():
    """Initialize query classifier."""
    global query_classifier, query_classifier_mode
    
    if query_classifier_mode == "regex":
        return
        
    try:
        from mindnest.utils.query_classification.classifier import QueryClassifier
        from mindnest.core.llm_manager import embeddings
        
        print(f"Initializing query classifier in {query_classifier_mode} mode...")
        
        # Explicitly clear the old classifier
        query_classifier = None
        
        # Create a new instance to ensure clean initialization
        query_classifier = QueryClassifier(embeddings, classifier_type=query_classifier_mode)
        
        # Force initialization of the underlying classifier
        query_classifier.initialize()
        
        # Ensure the classifier cache is clear
        query_classifier.clear_cache()
        
        print("Query classifier initialized successfully")
    except Exception as e:
        print(f"Warning: Failed to initialize query classifier: {e}")
        print("Falling back to regex-based classification")
        query_classifier_mode = "regex"
        query_classifier = None

def get_env_var(name: str, default: Any = None) -> Any:
    """Get environment variable with fallback."""
    return os.environ.get(name, default)

def get_bool_env_var(name: str, default: bool = False) -> bool:
    """Get boolean environment variable with fallback."""
    val = os.environ.get(name, str(default).lower())
    return val.lower() in ('true', '1', 'yes', 'y', 't') 