# Model Support

## Overview

MindNest provides intelligent support for different language models, optimizing prompts, context windows, and other parameters based on model capabilities. This adaptive approach ensures optimal performance regardless of the underlying model used.

## Supported Models

The system supports different model sizes and capabilities:

- **Small Models**: Efficient models like Llama-2-7B with smaller context windows
- **Large Models**: More powerful models like Wizard-Vicuna-13B with larger context windows

## Model-Specific Optimizations

### Context Window Management

Context window size is automatically managed based on model capabilities:

- **Small Models**: More aggressive truncation, typically using 2048-token windows
- **Large Models**: More generous context allocation, typically using 4096+ token windows
- **Token Counting**: Accurate token counting to prevent overflow
- **Content Prioritization**: Important content is prioritized to fit within limits

### Prompt Templates

Different prompt templates are used based on model capabilities:

- **Small Models**: More structured, explicit prompts with clear instructions
- **Large Models**: More flexible prompts that leverage the model's capabilities

Example prompt adaptation:

```python
if model_size == "small":
    # Structured template for small models
    template = """
    Answer the question based ONLY on the context provided below.
    
    CONTEXT:
    {context}
    
    QUESTION:
    {query}
    
    INSTRUCTIONS:
    - Use ONLY information from the context
    - Keep your answer clear and direct
    ...
    """
else:
    # More flexible template for larger models
    template = """
    Answer the following question based on the provided context. 
    
    If the question asks for a brief or concise answer, keep your response short...
    ...
    """
```

### Response Post-Processing

Different post-processing strategies based on model capabilities:

- **Small Models**: More aggressive formatting, length constraints, and quality checks
- **Large Models**: Minimal modifications to preserve model output quality

### Model Configuration

Model parameters can be configured through the API:

```python
@router.post("/config/model", status_code=200)
async def configure_model(config: ModelConfig):
    # Update model size if specified
    if config.use_small_model != (os.environ.get("USE_SMALL_MODEL", "").lower() == "true"):
        os.environ["USE_SMALL_MODEL"] = str(config.use_small_model).lower()
        
        # Return a message indicating restart needed for model change
        return {
            "status": "success", 
            "message": "Configuration updated. Server restart required for model change to take effect.",
            "config": {
                "use_small_model": config.use_small_model,
                ...
            }
        }
```

## Model Capabilities Dictionary

Model capabilities are tracked in a standardized dictionary format:

```python
def get_model_capabilities() -> Dict[str, Any]:
    """
    Get a dictionary of model capabilities for context-aware processing.
    """
    # Determine model size based on configuration
    model_size = "small" if use_small_model else "large"
    
    # Small model capabilities
    if model_size == "small":
        return {
            "model_size": "small",
            "context_window": 2048,  # Typical context window for 7B models
            "document_limit": 3,     # Retrieve fewer documents for small models
            "concise_limit": 2,      # Be more selective for concise queries
            "max_response_length": 250,  # Keep responses shorter for small models
            "prefers_structured_prompts": True,  # Small models do better with structure
            "context_compression_pct": 0.70,  # More aggressive compression
            "optimal_token_limit": 1536  # Lower token limit for small models
        }
    # Large model capabilities
    else:
        return {
            "model_size": "large", 
            "context_window": 8192,  # Higher context window for 13B+ models
            "document_limit": 5,     # Can handle more documents
            ...
        }
```

## LLM Initialization

LLMs are initialized with size-appropriate parameters:

```python
# Choose model based on configuration
selected_model = small_model_name if use_small_model else model_name
model_path = os.path.abspath(f"models/{selected_model}")

# Set n_ctx and batch size based on model size
n_ctx = 2048 if use_small_model else 4096  # Smaller context for smaller model
n_batch = 1024 if use_small_model else 1024  # Can be faster for smaller models

llm = LlamaCpp(
    model_path=model_path,
    temperature=0.3,
    max_tokens=2000,
    top_p=0.95,
    verbose=True,
    n_ctx=n_ctx,
    n_batch=n_batch,  # Increased from 512 to 1024 for faster processing
    n_gpu_layers=40,  # Use GPU acceleration if available
    repeat_penalty=1.1,
    f16_kv=True,
    use_mlock=True,  # Keep model in memory
    seed=42,  # Consistent results
    logits_all=False,  # Don't compute logits for all tokens (speeds up)
    stop=["</s>"],  # Stop token for faster completion
)
```

## Benefits

- **Flexibility**: Support for a range of model sizes and capabilities
- **Optimal Performance**: Each model operates within its strengths
- **Resource Efficiency**: Smaller models can be used when appropriate
- **Consistency**: Reliable performance across different models 