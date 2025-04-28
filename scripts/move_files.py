#!/usr/bin/env python3
"""
Script to move files from the root directory to their appropriate locations
in the new package structure.
"""

import os
import shutil
import sys

def move_file(src, dst, copy_only=False):
    """Move or copy a file from source to destination."""
    # Create destination directory if it doesn't exist
    dst_dir = os.path.dirname(dst)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    
    # Move or copy the file
    if os.path.exists(src):
        if not copy_only:
            print(f"Moving {src} to {dst}")
            shutil.move(src, dst)
        else:
            print(f"Copying {src} to {dst}")
            shutil.copy2(src, dst)
    else:
        print(f"Warning: Source file {src} does not exist")

def update_imports(file_path, old_prefix, new_prefix):
    """Update import statements in a file."""
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} does not exist")
        return
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace import statements
    updated_content = content.replace(f"from {old_prefix}", f"from {new_prefix}")
    updated_content = updated_content.replace(f"import {old_prefix}", f"import {new_prefix}")
    
    with open(file_path, 'w') as f:
        f.write(updated_content)
    
    print(f"Updated imports in {file_path}")

def main():
    """Main function to move utility files to the new package structure."""
    # Move utility files
    utils_files = [
        ('utils/query_cache.py', 'mindnest/utils/query_cache.py'),
        ('utils/document_tracker.py', 'mindnest/utils/document_tracker.py'),
        ('utils/incremental_vectorstore.py', 'mindnest/utils/incremental_vectorstore.py'),
        ('utils/query_optimization.py', 'mindnest/utils/query_optimization.py'),
        ('utils/responses.py', 'mindnest/utils/responses.py'),
        ('utils/models.py', 'mindnest/utils/models.py'),
        ('utils/logger.py', 'mindnest/utils/logger.py'),
        ('utils/config.py', 'mindnest/utils/config.py', True),  # Copy only to avoid conflicts
        ('utils/document_processor.py', 'mindnest/utils/document_processor.py', True),  # Copy only to avoid conflicts
        ('utils/token_counter.py', 'mindnest/utils/token_counter.py', True)  # Copy only to avoid conflicts
    ]
    
    for src, dst, *args in utils_files:
        copy_only = args[0] if args else False
        move_file(src, dst, copy_only)
    
    # Move query classification directory
    src_query_classification = 'utils/query_classification'
    dst_query_classification = 'mindnest/utils/query_classification'
    
    if os.path.exists(src_query_classification):
        if not os.path.exists(dst_query_classification):
            os.makedirs(dst_query_classification)
        
        # Copy files in query_classification directory
        for file in os.listdir(src_query_classification):
            if file.endswith('.py'):
                src_file = os.path.join(src_query_classification, file)
                dst_file = os.path.join(dst_query_classification, file)
                shutil.copy2(src_file, dst_file)
                print(f"Copied {src_file} to {dst_file}")
    
    # Move test files
    test_files = [
        ('tests/test_context_optimizer.py', 'tests/test_context_optimizer.py'),
        ('tests/test_document_processor.py', 'tests/test_document_processor.py'),
        ('tests/test_config.py', 'tests/test_config.py'),
        ('test_api_integration.py', 'tests/test_api_integration.py'),
        ('test_optimizer.py', 'tests/test_optimizer.py'),
        ('verify_context_optimization.py', 'tests/verify_context_optimization.py')
    ]
    
    for src, dst in test_files:
        if os.path.exists(src):
            move_file(src, dst, copy_only=True)  # Copy test files instead of moving
    
    # Move script files
    script_files = [
        ('cleanup_docs.py', 'scripts/cleanup_docs.py'),
        ('doc_chunker.py', 'scripts/doc_chunker.py')
    ]
    
    for src, dst in script_files:
        move_file(src, dst, copy_only=True)  # Copy script files instead of moving
    
    # Create wrapper scripts
    create_wrapper_script('run_direct.py', 'from mindnest.app import run_server\n\nif __name__ == "__main__":\n    run_server()\n')
    create_wrapper_script('run_server.py', 'from mindnest.app import run_server\n\nif __name__ == "__main__":\n    run_server()\n')
    
    print("Files moved successfully")

def create_wrapper_script(filename, content):
    """Create a wrapper script."""
    with open(filename, 'w') as f:
        f.write(content)
    print(f"Created wrapper script {filename}")

if __name__ == "__main__":
    main() 