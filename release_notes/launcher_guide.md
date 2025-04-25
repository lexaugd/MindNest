# MindNest Launcher Guide

## Overview
The MindNest Launcher provides a convenient way to start the MindNest application with different configuration options. It offers a command-line interface with several options for customizing how the application runs.

## Features
- Start the application in full mode (with LLM) or lightweight mode (vector search only)
- Choose between the default (13B) or smaller (7B) language models
- Automatically check environment and verify required files
- Apply configuration changes without manually editing .env files
- Open web browser automatically when the server starts
- Display real-time server logs in the console

## Usage
You can run the launcher using Python:

```bash
python mindnest_launcher.py [options]
```

Or directly (if you've made it executable):

```bash
./mindnest_launcher.py [options]
```

## Options

### Mode Selection
Choose which mode to run MindNest in:

```bash
# Start with full capabilities (default)
python mindnest_launcher.py --mode full

# Start in lightweight mode (vector search only, no LLM)
python mindnest_launcher.py --mode lightweight
```

### Model Selection
Choose which language model to use:

```bash
# Use the default 13B model (default)
python mindnest_launcher.py --model default

# Use the smaller 7B model
python mindnest_launcher.py --model small
```

### Browser Control
Control whether to open the browser automatically:

```bash
# Don't open browser automatically
python mindnest_launcher.py --no-browser
```

### Environment Check
Perform an environment check without starting the server:

```bash
python mindnest_launcher.py --check
```

## Examples

### Start with Default Settings
Start MindNest with the default 13B model in full mode:

```bash
python mindnest_launcher.py
```

### Start with Smaller Model
Start MindNest with the smaller 7B model for faster performance:

```bash
python mindnest_launcher.py --model small
```

### Start in Lightweight Mode
Start MindNest in lightweight mode (vector search only) for minimal resource usage:

```bash
python mindnest_launcher.py --mode lightweight
```

### Start with Custom Configuration
Start MindNest with a specific configuration:

```bash
python mindnest_launcher.py --model small --mode full --no-browser
```

## Troubleshooting

### Missing Model Files
If you see warnings about missing model files:
1. Download the required model files as described in the README
2. Place them in the `models/` directory
3. Make sure the filenames match exactly what the application expects

### Virtual Environment Issues
If the launcher can't find the virtual environment:
1. Create a new virtual environment: `python -m venv venv`
2. Activate it: `source venv/bin/activate` (Unix/Mac) or `venv\Scripts\activate` (Windows)
3. Install dependencies: `pip install -r requirements.txt`

### Server Startup Failures
If the server fails to start:
1. Check the error messages in the console
2. Verify that all required dependencies are installed
3. Ensure port 8000 is not already in use by another application

## Creating a Desktop Shortcut
You can create a desktop shortcut to launch MindNest with specific configurations:

### On macOS:
1. Create an AppleScript file (e.g., `MindNest.scpt`) with the following content:
   ```applescript
   tell application "Terminal"
     do script "cd /path/to/MindNest && ./mindnest_launcher.py"
   end tell
   ```
2. Save it as an application using Script Editor

### On Windows:
1. Create a batch file (e.g., `MindNest.bat`) with:
   ```batch
   @echo off
   cd /d "C:\path\to\MindNest"
   python mindnest_launcher.py
   ```
2. Create a shortcut to this batch file

### On Linux:
1. Create a .desktop file (e.g., `mindnest.desktop`):
   ```
   [Desktop Entry]
   Name=MindNest
   Exec=/path/to/MindNest/mindnest_launcher.py
   Terminal=true
   Type=Application
   ```
2. Place it in `~/.local/share/applications/` 