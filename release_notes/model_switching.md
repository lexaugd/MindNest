# Model Switching in MindNest

## Overview
MindNest provides a model switching capability that allows users to choose between two LLM models:

1. **Default Model**: `Wizard-Vicuna-13B-Uncensored.Q4_K_M.gguf` (7.3GB)
   - Higher quality responses
   - Larger context window (4096 tokens)
   - Requires more memory

2. **Faster Model**: `llama-2-7b.Q4_K_M.gguf` (3.8GB)
   - Faster response times
   - Smaller context window (2048 tokens)
   - Requires less memory
   - Good for resource-constrained environments

## UI Controls
The model switching feature is accessible through the UI settings panel:

1. Click on the ⚙️ (Settings) icon in the top-right corner of the interface
2. Find the "Use Faster Model" checkbox
3. Check the box to use the smaller, faster model
4. Click "Apply Settings" to save your changes

## Technical Details

### What Happens When Switching Models

When you switch models through the UI:

1. The application updates the `USE_SMALL_MODEL` environment variable
2. The UI displays a notification that a restart is required
3. Upon server restart, the application loads the appropriate model:
   - If `USE_SMALL_MODEL=true`, it loads the 7B model with a 2048 token context window
   - If `USE_SMALL_MODEL=false`, it loads the 13B model with a 4096 token context window

### Server Restart Required

**Important**: The model change takes effect only after restarting the server. This is because:
- LLMs are loaded into memory at startup
- Switching models requires freeing the current model and loading the new one
- This process is too resource-intensive to perform during runtime

## Use Cases

### When to Use the Default (13B) Model:
- Complex queries requiring deeper understanding
- Technical documentation with nuanced explanations
- Code analysis that requires sophisticated reasoning
- When response quality is more important than speed
- When memory resources are not constrained

### When to Use the Faster (7B) Model:
- Simple factual queries about the documentation
- Quick lookups of function usage or definitions
- High-volume query scenarios where speed is crucial
- On systems with limited memory (8GB RAM or less)
- When battery life is a concern on laptops

## Instructions

### Switching to the Faster Model:

1. Open the MindNest web interface
2. Click the Settings (⚙️) icon in the top right
3. Check the "Use Faster Model" checkbox
4. Click "Apply Settings"
5. You'll see a notification: "Configuration updated. Server restart required for model change to take effect."
6. Restart the server using one of these methods:
   - If running from terminal: Press Ctrl+C and then run `python main.py` again
   - If running as a service: Restart the service
   - If using Docker: Restart the container

### Switching Back to the Default Model:

1. Open the MindNest web interface
2. Click the Settings (⚙️) icon in the top right
3. Uncheck the "Use Faster Model" checkbox
4. Click "Apply Settings"
5. You'll see a notification about a restart being required
6. Restart the server as described above

## Performance Comparison

| Feature | Default (13B) Model | Faster (7B) Model |
|---------|---------------------|-------------------|
| File Size | 7.3GB | 3.8GB |
| Memory Usage | ~8-12GB RAM | ~4-6GB RAM |
| Context Window | 4096 tokens | 2048 tokens |
| Response Time | Slower | ~1.5-2x faster |
| Response Quality | High | Good, but less nuanced |
| Reasoning Capability | Complex reasoning | Simpler reasoning |

## Troubleshooting

### Common Issues:

1. **"Model not found" error**:
   - Ensure both model files exist in the `models/` directory
   - Verify file names match exactly with configuration

2. **High memory usage with small model**:
   - Reduce `BATCH_SIZE` in .env file
   - Lower `CONTEXT_WINDOW` for additional memory savings

3. **Server crashes when switching models**:
   - Ensure sufficient system memory (8GB+ recommended)
   - Close other memory-intensive applications
   - Try reducing `CONTEXT_WINDOW` in .env file 