#!/usr/bin/env python3
"""
MindNest Launcher
A convenient launcher for the MindNest AI Documentation System
"""

import os
import sys
import argparse
import subprocess
import webbrowser
import time
import signal
import platform
import importlib.util
import pkg_resources
from pathlib import Path

# Configuration
APP_NAME = "MindNest"
DEFAULT_PORT = 8000
DEFAULT_HOST = "0.0.0.0"
DEFAULT_URL = f"http://localhost:{DEFAULT_PORT}"
VENV_PATH = Path("venv")
PYTHON_PATH = VENV_PATH / ("Scripts" if platform.system() == "Windows" else "bin") / "python"

def activate_virtual_env():
    """Set up the virtual environment for the subprocess"""
    venv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv")
    
    if platform.system() == "Windows":
        python_path = os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        python_path = os.path.join(venv_dir, "bin", "python")
    
    if not os.path.exists(python_path):
        print(f"⚠️ Virtual environment not found at {venv_dir}")
        print("ℹ️ Using system Python instead")
        return sys.executable
    
    return python_path

def update_env_file(use_small_model=None, context_window=None, query_mode=None):
    """Update .env file with specified configuration"""
    env_path = Path(".env")
    
    # Create .env file from template if it doesn't exist
    if not env_path.exists():
        template_path = Path("env.example")
        if template_path.exists():
            print("ℹ️ Creating .env file from template")
            with open(template_path, 'r') as src:
                with open(env_path, 'w') as dst:
                    dst.write(src.read())
        else:
            print("⚠️ env.example template not found")
            return False
    
    # Read current content
    with open(env_path, 'r') as f:
        lines = f.readlines()
    
    # Update values
    updated = {}
    new_lines = []
    
    for line in lines:
        if use_small_model is not None and line.startswith("USE_SMALL_MODEL="):
            new_lines.append(f"USE_SMALL_MODEL={str(use_small_model).lower()}\n")
            updated["USE_SMALL_MODEL"] = str(use_small_model).lower()
        elif context_window is not None and line.startswith("CONTEXT_WINDOW="):
            new_lines.append(f"CONTEXT_WINDOW={context_window}\n")
            updated["CONTEXT_WINDOW"] = str(context_window)
        elif query_mode is not None and line.startswith("QUERY_CLASSIFIER_MODE="):
            new_lines.append(f"QUERY_CLASSIFIER_MODE={query_mode}\n")
            updated["QUERY_CLASSIFIER_MODE"] = query_mode
        else:
            new_lines.append(line)
    
    # Write updated content
    with open(env_path, 'w') as f:
        f.writelines(new_lines)
    
    # Log changes
    if updated:
        print("✓ Updated configuration:")
        for key, value in updated.items():
            print(f"  - {key}: {value}")
    
    return True

def check_dependencies():
    """Check if all required dependencies are installed"""
    req_file = Path("requirements.txt")
    if not req_file.exists():
        print("⚠️ requirements.txt file not found")
        return False
    
    print("🔍 Checking dependencies...")
    missing_packages = []
    outdated_packages = []
    
    # Read requirements
    with open(req_file, 'r') as f:
        lines = [line.strip() for line in f.readlines() 
                if line.strip() and not line.startswith('#')]
    
    # Check each requirement
    for line in lines:
        # Parse requirement line
        if "==" in line:
            package, version = line.split("==")
        elif ">=" in line:
            package, version = line.split(">=")
        else:
            package, version = line, None
        
        # Check if package is installed
        spec = importlib.util.find_spec(package.replace("-", "_"))
        if spec is None:
            missing_packages.append(line)
            continue
        
        # Check version if specified
        if version:
            try:
                installed_version = pkg_resources.get_distribution(package).version
                if installed_version != version and "==" in line:
                    outdated_packages.append((package, installed_version, version))
            except pkg_resources.DistributionNotFound:
                missing_packages.append(line)
    
    if missing_packages:
        print(f"⚠️ Missing {len(missing_packages)} required packages:")
        for pkg in missing_packages[:5]:
            print(f"  - {pkg}")
        if len(missing_packages) > 5:
            print(f"    ...and {len(missing_packages) - 5} more")
        return False
    
    if outdated_packages:
        print(f"⚠️ Found {len(outdated_packages)} outdated packages:")
        for pkg, current, required in outdated_packages[:5]:
            print(f"  - {pkg}: {current} (required: {required})")
        if len(outdated_packages) > 5:
            print(f"    ...and {len(outdated_packages) - 5} more")
        return False
    
    print("✓ All dependencies are installed and up to date")
    return True

def install_dependencies(auto_accept=False):
    """Install missing dependencies"""
    req_file = Path("requirements.txt")
    if not req_file.exists():
        print("⚠️ requirements.txt file not found")
        return False
    
    if not auto_accept:
        response = input("Would you like to install/update the required dependencies? (y/n): ")
        if response.lower() not in ["y", "yes"]:
            return False
    
    print("📦 Installing dependencies...")
    
    # Determine the Python executable to use
    python_exe = activate_virtual_env()
    
    # Install dependencies
    try:
        subprocess.run(
            [python_exe, "-m", "pip", "install", "-r", "requirements.txt"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        print(f"Error details: {e.stderr}")
        return False

def start_server(mode="full", model="default", open_browser=True):
    """Start the MindNest server"""
    python_exe = activate_virtual_env()
    
    # Configure environment based on parameters
    if model.lower() == "small":
        update_env_file(use_small_model=True)
    elif model.lower() == "default":
        update_env_file(use_small_model=False)
    
    # Determine which script to run
    if mode.lower() == "lightweight":
        script = "run_server.py"
        print(f"🚀 Starting {APP_NAME} in Lightweight mode (vector search only)")
    else:
        script = "main.py"
        print(f"🚀 Starting {APP_NAME} in Full mode (with LLM)")
    
    # Start the server process
    try:
        server_process = subprocess.Popen(
            [python_exe, script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Wait for server to start
        print("⏳ Server starting...")
        server_ready = False
        start_time = time.time()
        timeout = 30  # seconds
        
        while not server_ready and time.time() - start_time < timeout:
            output = server_process.stdout.readline()
            if output:
                print(f"   {output.strip()}")
                if "Starting server on" in output or "Uvicorn running on" in output:
                    server_ready = True
                    print(f"✓ Server started at {DEFAULT_URL}")
                    
                    # Open browser if requested
                    if open_browser:
                        print("🌐 Opening web browser...")
                        webbrowser.open(DEFAULT_URL)
                    
                    # Continue printing output
                    print("\n📋 Server logs (press Ctrl+C to stop):")
            
            # Check if process is still running
            if server_process.poll() is not None:
                print("❌ Server process terminated unexpectedly")
                return False
            
            time.sleep(0.1)
        
        if not server_ready:
            print("⚠️ Server startup timed out")
            return False
        
        # Keep printing server output
        try:
            while True:
                output = server_process.stdout.readline()
                if output:
                    print(output.strip())
                
                # Check if process is still running
                if server_process.poll() is not None:
                    print("⚠️ Server process terminated")
                    break
                
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n⏹️ Stopping server...")
            server_process.send_signal(signal.SIGINT)
            server_process.wait(timeout=5)
            print("✓ Server stopped")
    
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return False
    
    return True

def print_logo():
    """Print a simple ASCII logo"""
    logo = r"""
 __  __ _           _ _   _           _   
|  \/  (_)_ __   __| | \ | | ___  ___| |_ 
| |\/| | | '_ \ / _` |  \| |/ _ \/ __| __|
| |  | | | | | | (_| | |\  |  __/\__ \ |_ 
|_|  |_|_|_| |_|\__,_|_| \_|\___||___/\__|
                                          
AI-Powered Documentation System - v1.0
"""
    print(logo)

def check_environment():
    """Check if all requirements are met"""
    all_ok = True
    
    # Check if virtual environment exists
    venv_path = Path("venv")
    if not venv_path.exists():
        print("⚠️ Virtual environment not found")
        print("ℹ️ Please create a virtual environment and install dependencies:")
        print("   python -m venv venv")
        if platform.system() == "Windows":
            print("   venv\\Scripts\\activate")
        else:
            print("   source venv/bin/activate")
        print("   pip install -r requirements.txt")
        all_ok = False
    
    # Check if model files exist
    models_path = Path("models")
    model_files = list(models_path.glob("*.gguf"))
    if not model_files:
        print("⚠️ No model files found in models/ directory")
        print("ℹ️ Please download the model files as described in README.md")
        all_ok = False
    
    # Check for required models
    default_model = models_path / "Wizard-Vicuna-13B-Uncensored.Q4_K_M.gguf"
    small_model = models_path / "llama-2-7b.Q4_K_M.gguf"
    
    missing_models = []
    if not default_model.exists():
        missing_models.append("Wizard-Vicuna-13B-Uncensored.Q4_K_M.gguf")
    if not small_model.exists():
        missing_models.append("llama-2-7b.Q4_K_M.gguf")
    
    if missing_models:
        print(f"⚠️ Missing model files: {', '.join(missing_models)}")
        print("ℹ️ The following model files are available:")
        for model in model_files:
            print(f"   - {model.name}")
        all_ok = False
    
    # Check for .env file
    env_path = Path(".env")
    if not env_path.exists():
        template_path = Path("env.example")
        if template_path.exists():
            print("⚠️ .env file not found, but env.example template exists")
            print("ℹ️ A new .env file will be created from the template when launching")
        else:
            print("❌ Neither .env nor env.example files found")
            all_ok = False
    
    # Check dependencies
    if not check_dependencies():
        all_ok = False
    
    return all_ok

def main():
    """Main launcher function"""
    print_logo()
    
    parser = argparse.ArgumentParser(description=f"Launcher for {APP_NAME}")
    parser.add_argument("--mode", choices=["full", "lightweight"], default="full",
                        help="Server mode: full (with LLM) or lightweight (vector search only)")
    parser.add_argument("--model", choices=["default", "small"], default="default",
                        help="Model to use: default (13B) or small (7B)")
    parser.add_argument("--no-browser", action="store_true",
                        help="Don't open browser automatically")
    parser.add_argument("--check", action="store_true",
                        help="Check environment and exit")
    parser.add_argument("--install-deps", action="store_true",
                        help="Install or update dependencies")
    parser.add_argument("--auto-install", action="store_true",
                        help="Automatically install dependencies without prompting")
    
    args = parser.parse_args()
    
    # If just installing dependencies
    if args.install_deps:
        install_dependencies(auto_accept=args.auto_install)
        return 0
    
    # If check only, just run the check and exit
    if args.check:
        if check_environment():
            print("✓ Environment check passed")
            return 0
        else:
            return 1
    
    # Check environment before starting
    if not check_environment():
        print("\n⚠️ Warning: Environment check failed")
        
        # Check if we should auto-install dependencies
        if args.auto_install:
            print("🔄 Attempting to fix issues automatically...")
            install_dependencies(auto_accept=True)
            
            # Check again after installation
            if not check_environment():
                print("\n❌ Still issues with environment after auto-fix")
                response = input("Continue anyway? (y/n): ")
                if response.lower() not in ["y", "yes"]:
                    return 1
        else:
            # Offer to install dependencies if they're missing
            missing_deps = not check_dependencies()
            if missing_deps:
                if install_dependencies():
                    print("🔄 Checking environment again after dependency installation...")
                    check_environment()
            
            # Ask if user wants to continue despite environment issues
            response = input("Continue anyway? (y/n): ")
            if response.lower() not in ["y", "yes"]:
                return 1
    
    # Start the server
    print("\n" + "=" * 60)
    result = start_server(
        mode=args.mode,
        model=args.model,
        open_browser=not args.no_browser
    )
    
    return 0 if result else 1

if __name__ == "__main__":
    sys.exit(main()) 