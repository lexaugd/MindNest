# Local AI Code Assistant

This is a local AI assistant that can answer questions about your company's codebase and documentation. It uses open-source models and runs completely locally on your machine.

## Features

- Local processing - no data leaves your machine
- Uses Llama 2 model for natural language understanding
- Vector database for efficient document retrieval
- Simple chat interface
- Free and open-source

## Setup

1. Install Python 3.9+ if you haven't already
2. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the Llama 2 model (instructions below)
4. Create a `.env` file with your configuration
5. Run the application:
   ```bash
   python main.py
   ```

## Downloading the Model

1. Visit [TheBloke's Llama-2-7B-GGUF](https://huggingface.co/TheBloke/Llama-2-7B-GGUF) on Hugging Face
2. Download the `llama-2-7b.Q4_K_M.gguf` model file
3. Place it in the `models` directory

## Usage

1. Start the server using `python main.py`
2. Open your browser to `http://localhost:8000`
3. Start chatting with your codebase!

## Adding Your Documentation

Place your documentation files in the `docs` directory. Supported formats:
- Markdown (.md)
- Text (.txt)
- Python files (.py)
- JavaScript files (.js)
- TypeScript files (.ts)

The system will automatically index these files when started. 