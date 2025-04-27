#!/usr/bin/env python3
"""
MindNest Direct Starter
----------------------
Simplified direct starter for MindNest without the launcher layer.
Provides command-line arguments for configuration.
"""

import os
import sys
import argparse
import subprocess
import webbrowser
import time

def main():
    """Parse command-line arguments and launch the MindNest application directly."""
    parser = argparse.ArgumentParser(description="Launch MindNest Application")
    parser.add_argument("--lightweight", action="store_true", 
                       help="Run in lightweight mode with minimal dependencies")
    parser.add_argument("--lightweight-model", action="store_true", 
                       help="Use the lightweight language model (Llama-2-7B)")
    parser.add_argument("--port", type=int, default=8000,
                       help="Port to run the server on (default: 8000)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="Host to run the server on (default: 0.0.0.0)")
    parser.add_argument("--no-browser", action="store_true",
                       help="Don't open browser automatically")
    
    args = parser.parse_args()
    
    # Set environment variables based on arguments
    if args.lightweight_model:
        os.environ["USE_SMALL_MODEL"] = "true"
        print("Using lightweight model: llama-2-7b.Q4_K_M.gguf")
    else:
        os.environ["USE_SMALL_MODEL"] = "false"
        print("Using standard model: Wizard-Vicuna-13B-Uncensored.Q4_K_M.gguf")
    
    if args.lightweight:
        os.environ["MINDNEST_MODE"] = "lightweight"
        print("Running in lightweight mode")
    else:
        os.environ["MINDNEST_MODE"] = "standard"
        print("Running in standard mode")
    
    # Set server host and port
    os.environ["HOST"] = args.host
    os.environ["PORT"] = str(args.port)
    
    print(f"Starting server on {args.host}:{args.port}")
    
    # Open browser if requested
    if not args.no_browser:
        # Delay browser opening to give server time to start
        def open_browser():
            time.sleep(2)
            url = f"http://localhost:{args.port}"
            print(f"Opening browser at {url}")
            webbrowser.open(url)
        
        import threading
        threading.Thread(target=open_browser, daemon=True).start()
    
    # Run the main application using uvicorn instead of direct import
    import uvicorn
    uvicorn.run("main:app", host=args.host, port=args.port, log_level="info")
    
if __name__ == "__main__":
    main() 