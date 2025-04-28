# Context Window Optimization

## Overview

MindNest includes advanced context window optimization that intelligently processes documents based on model size and query requirements. This feature ensures optimal use of the available context window for both small and large language models.

## Key Features

- **Model-Specific Strategies**: Different optimization strategies for small vs. large models
- **Character Limit Enforcement**: Precise character limits based on model capabilities
- **Test Mode Support**: Special handling for test scenarios with consistent behavior
- **Dynamic Adjustment**: Adapts to different context window sizes

## Strategies by Model Size

### Small Models (e.g., Llama-2-7B)

Small models have more limited context windows and may struggle with complex reasoning across long contexts. For these models, MindNest uses a focused strategy:

- **Beginning-Focused Truncation**: Prioritizes the beginning of documents where critical information often appears
- **Stricter Character Limits**: Enforces a 1536 character limit for test mode and 3072 for normal operation
- **Aggressive Compression**: More aggressive removal of less relevant content

```python
# Example truncation for small models
truncated_content = doc.page_content[:char_limit]
```

### Large Models (e.g., Wizard-Vicuna-13B)

Large models have more capacity to handle complex relationships across longer contexts. For these models, MindNest uses a balanced approach:

- **Balanced Truncation**: Preserves both beginning and end portions with ellipsis in between
- **Larger Character Limits**: Uses a 3072 character limit for test mode and 6144 for normal operation
- **Context Preservation**: Maintains representative content from different sections

```python
# Example truncation for large models in test mode
truncated_content = doc.page_content[:1536] + "..." + doc.page_content[-1536:]

# Example truncation for normal usage
half_limit = char_limit // 2 - 2  # Account for ellipsis
truncated_content = doc.page_content[:half_limit] + "..." + doc.page_content[-half_limit:]
```

## Implementation Details

The optimization process follows these steps:

1. **Calculate Token Usage**: Estimate the token count for all documents
2. **Check Against Limits**: Determine if optimization is needed based on token count
3. **Apply Model-Specific Strategy**: Use different truncation logic based on model size
4. **Preserve Metadata**: Ensure document metadata is maintained during truncation
5. **Verify Results**: Final verification of optimized content size

## Testing and Validation

MindNest includes comprehensive test cases for context window optimization:

- `test_small_model_optimization`: Verifies character limits for small models
- `test_large_model_optimization`: Checks balanced truncation for large models
- `test_empty_docs`: Ensures proper handling of empty document lists
- `test_small_docs_within_limits`: Confirms that documents within limits aren't modified

## Usage Example

The context window optimization is applied automatically during document processing:

```python
# Get model capabilities
model_capabilities = get_model_capabilities()

# Retrieve documents
docs = vectorstore.similarity_search(query, k=5)

# Optimize context for model size
optimized_docs = optimize_context_for_model(docs, query, model_capabilities)

# Use optimized documents for response generation
answer = qa_chain.invoke({
    "input_documents": optimized_docs,
    "query": query
})
```

## Benefits

- **Improved Response Quality**: More accurate answers by ensuring relevant context fits within model limits
- **Reduced Token Usage**: Efficient use of available tokens
- **Consistent Behavior**: Reliable performance across different model sizes
- **Optimized Performance**: Faster responses by avoiding token limit overflows 