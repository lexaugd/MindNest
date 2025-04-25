# MindNest - AI Documentation System

An intelligent documentation system that uses AI to understand and answer questions about your codebase.

## Features

- **Multi-Format Support**: Understands Java, Groovy, and text documentation
- **Smart Document Processing**: Automatically processes and indexes your documentation
- **Intelligent Q&A**: Answers questions about your code accurately
- **Source Attribution**: Always shows which files the information comes from

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Unix/Mac
# or
venv\Scripts\activate     # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the Llama model:
- Place `llama-2-7b.Q4_K_M.gguf` in the `models/` directory

4. Run the application:
```bash
python main.py
```

## Project Structure

```
MindNest/
├── docs/                 # Documentation and code examples
│   ├── java/            # Java code examples
│   ├── groovy/          # Groovy code examples
│   └── txt/             # Text documentation
├── models/              # AI model files
├── static/              # Static web files
├── main.py             # Main application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Usage

1. Place your documentation in the `docs/` directory
2. Start the server: `python main.py`
3. Ask questions about your code
4. Get accurate, source-attributed answers

## Technical Details

- Uses Llama 2 7B (Q4_K_M quantized version)
- Optimized for CPU usage
- Supports context window of 2048 tokens
- Uses HuggingFace embeddings for document processing

## Limitations

- Best suited for codebases under 100 files
- Response time may vary based on system resources
- Requires restart to load new documentation

![image](https://github.com/user-attachments/assets/bb04756e-5cdf-4486-a5b3-2ff9ae7305ff)
![Uploading image.png…]()

## License

