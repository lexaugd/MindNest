"""
Configuration management system for MindNest.
Centralizes all configuration settings and provides environment variable overrides.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

class Config:
    """Central configuration manager for the MindNest application."""
    
    def __init__(self):
        """Initialize configuration with default values and environment overrides."""
        # Base paths
        self.base_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
        self.temperature = float(os.environ.get("TEMPERATURE", "0.3"))
        self.max_tokens = int(os.environ.get("MAX_TOKENS", "2000"))
        self.top_p = float(os.environ.get("TOP_P", "0.95"))
        self.context_window = int(os.environ.get("CONTEXT_WINDOW", "2048"))
        self.batch_size = int(os.environ.get("BATCH_SIZE", "1024"))
        
        # Cache settings
        self.cache_enabled = os.environ.get("CACHE_ENABLED", "true").lower() == "true"
        self.memory_cache_size = int(os.environ.get("MEMORY_CACHE_SIZE", "1024"))
        self.disk_cache_enabled = os.environ.get("DISK_CACHE_ENABLED", "true").lower() == "true"
        self.disk_cache_path = os.environ.get("DISK_CACHE_PATH", "utils/query_cache.json")
        
        # Query settings
        self.query_classifier_mode = os.environ.get("QUERY_CLASSIFIER_MODE", "embeddings")
        self.conversation_mode = os.environ.get("CONVERSATION_MODE", "professional")
        
        # Server settings
        self.host = os.environ.get("HOST", "0.0.0.0")
        self.port = int(os.environ.get("PORT", "8000"))
    
    def get_model_path(self, use_small_model: Optional[bool] = None) -> str:
        """Get the absolute path to the model file."""
        if use_small_model is None:
            use_small_model = self.use_small_model
            
        model = self.small_model_name if use_small_model else self.model_name
        return str(self.models_dir / model)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary."""
        # Filter out private attributes and methods
        return {k: v for k, v in self.__dict__.items() 
                if not k.startswith('_') and not callable(v)}
    
    def __str__(self) -> str:
        """String representation of the configuration."""
        return str(self.to_dict())


# Create a singleton instance
config = Config()

# For testing purposes
if __name__ == "__main__":
    print("MindNest Configuration:")
    for key, value in config.to_dict().items():
        print(f"{key}: {value}") 