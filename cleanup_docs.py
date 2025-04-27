#!/usr/bin/env python3
"""
Document cleanup script for MindNest.

This script organizes the docs folder by:
1. Consolidating duplicate content between txt and markdown files
2. Cleaning up empty or placeholder files
3. Breaking up large files into smaller, more focused documents

Usage:
    python cleanup_docs.py
"""

import os
import re
import shutil
from pathlib import Path

def is_placeholder(content):
    """Check if a file is just a placeholder with minimal content."""
    # If file is very small and has no substantial content
    if len(content.strip()) < 20:
        return True
    return False

def is_duplicate(file1_path, file2_path):
    """Check if two files have identical or very similar content."""
    with open(file1_path, 'r', encoding='utf-8', errors='ignore') as f1:
        content1 = f1.read()
    
    with open(file2_path, 'r', encoding='utf-8', errors='ignore') as f2:
        content2 = f2.read()
    
    # Simple exact match
    if content1 == content2:
        return True
    
    # Check if they're similar after removing markdown formatting
    # Strip markdown formatting
    content1_clean = re.sub(r'#+ |```.*?```|\*\*|\*|__|\[.*?\]\(.*?\)', '', content1, flags=re.DOTALL)
    content2_clean = re.sub(r'#+ |```.*?```|\*\*|\*|__|\[.*?\]\(.*?\)', '', content2, flags=re.DOTALL)
    
    # If after removing formatting they're very similar
    similarity = len(set(content1_clean.split()) & set(content2_clean.split())) / max(1, len(set(content1_clean.split() + content2_clean.split())))
    return similarity > 0.85

def split_large_document(file_path, max_size=30000, output_dir=None):
    """Split a large document into smaller files by sections."""
    if output_dir is None:
        output_dir = os.path.dirname(file_path)
    
    filename = os.path.basename(file_path)
    name, ext = os.path.splitext(filename)
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # If not too large, just return the original
    if len(content) <= max_size:
        return [file_path]
    
    # Try to split by markdown headings
    if ext.lower() in ['.md', '.markdown']:
        # Find all headings
        heading_pattern = r'^(#+)\s+(.+)$'
        headings = [(m.group(1), m.group(2), m.start()) for m in re.finditer(heading_pattern, content, re.MULTILINE)]
        
        # If we have headings, split on level 1 or 2 headings
        if headings:
            # Filter for only top-level and second-level headings (# and ##)
            top_headings = [h for h in headings if len(h[0]) <= 2]
            
            if top_headings:
                parts = []
                for i, (heading_level, heading_text, start_pos) in enumerate(top_headings):
                    # Get end position (start of next heading or end of file)
                    end_pos = top_headings[i+1][2] if i < len(top_headings)-1 else len(content)
                    
                    # Get section content
                    section = content[start_pos:end_pos]
                    
                    # Clean heading text for filename
                    clean_heading = re.sub(r'[^\w\s-]', '', heading_text).strip().lower()
                    clean_heading = re.sub(r'[-\s]+', '-', clean_heading)
                    
                    # Create filename for this section
                    section_filename = f"{name}-{clean_heading}{ext}"
                    section_path = os.path.join(output_dir, section_filename)
                    
                    # Write section to file
                    with open(section_path, 'w', encoding='utf-8') as out_file:
                        out_file.write(section)
                    
                    parts.append(section_path)
                
                return parts
    
    # Fallback: split by size if no clear section boundaries
    chunks = []
    chunk_size = max_size
    
    # Simple chunking by size
    for i in range(0, len(content), chunk_size):
        chunk = content[i:i+chunk_size]
        
        # Try to break at paragraph boundary if possible
        if i+chunk_size < len(content):
            # Find last paragraph break
            last_para = chunk.rfind('\n\n')
            if last_para > chunk_size * 0.5:  # If we found a paragraph break in the second half
                chunk = chunk[:last_para]
                i = i + last_para
        
        # Create chunk filename
        chunk_filename = f"{name}-part{len(chunks)+1}{ext}"
        chunk_path = os.path.join(output_dir, chunk_filename)
        
        # Write chunk to file
        with open(chunk_path, 'w', encoding='utf-8') as out_file:
            out_file.write(chunk)
        
        chunks.append(chunk_path)
    
    return chunks

def cleanup_docs_folder():
    """Clean up and organize the docs folder."""
    docs_dir = Path('docs')
    
    # Check if docs directory exists
    if not docs_dir.exists():
        print("Documentation folder not found.")
        return
    
    print("Starting document cleanup...")
    
    # Create backup
    backup_dir = docs_dir.parent / "docs_backup"
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    
    # Create backup
    print("Creating backup of docs folder...")
    shutil.copytree(docs_dir, backup_dir)
    
    # Scan for placeholder files
    placeholder_files = []
    for root, dirs, files in os.walk(docs_dir):
        for file in files:
            if file.startswith('.'):  # Skip hidden files
                continue
                
            file_path = os.path.join(root, file)
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                if is_placeholder(content):
                    placeholder_files.append(file_path)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    # Consolidate duplicate content
    print("Checking for duplicate content...")
    txt_dir = docs_dir / "txt"
    md_dir = docs_dir / "md"
    edi_dir = docs_dir / "edi"
    
    duplicates = []
    
    # Check for duplicates between txt and markdown versions
    if txt_dir.exists() and (md_dir.exists() or edi_dir.exists()):
        for txt_file in txt_dir.glob("*.txt"):
            base_name = txt_file.stem
            
            # Check for duplicate in markdown folder
            md_file = md_dir / f"{base_name}.md" if md_dir.exists() else None
            edi_md_file = edi_dir / f"{base_name}.md" if edi_dir.exists() else None
            
            # If either md version exists and is a duplicate
            if (md_file and md_file.exists() and is_duplicate(txt_file, md_file)):
                duplicates.append((str(txt_file), str(md_file)))
            elif (edi_md_file and edi_md_file.exists() and is_duplicate(txt_file, edi_md_file)):
                duplicates.append((str(txt_file), str(edi_md_file)))
    
    # Handle large files
    print("Processing large files...")
    large_files = []
    for root, dirs, files in os.walk(docs_dir):
        for file in files:
            if file.startswith('.'):  # Skip hidden files
                continue
                
            file_path = os.path.join(root, file)
            
            try:
                # Check file size
                if os.path.getsize(file_path) > 30000:  # Files larger than ~30KB
                    large_files.append(file_path)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    # Print summary before taking any action
    print("\nSummary of issues found:")
    print(f"- {len(placeholder_files)} placeholder files")
    print(f"- {len(duplicates)} duplicate file pairs")
    print(f"- {len(large_files)} large files that could be split")
    
    # Ask for confirmation
    print("\nReady to apply fixes:")
    print("1. Remove placeholder files")
    print("2. Consolidate duplicate content (keeps markdown, removes txt duplicates)")
    print("3. Split large files into smaller sections")
    
    confirm = input("\nDo you want to proceed? (y/n): ")
    if confirm.lower() != 'y':
        print("Cleanup aborted. No changes made.")
        return
    
    # Apply fixes
    print("\nApplying fixes...")
    
    # 1. Remove placeholders
    for file_path in placeholder_files:
        if os.path.exists(file_path):
            print(f"Removing placeholder file: {file_path}")
            os.remove(file_path)
        else:
            print(f"Placeholder file already removed: {file_path}")
    
    # 2. Remove duplicated txt files (prefer markdown)
    for txt_path, md_path in duplicates:
        if os.path.exists(txt_path):
            print(f"Removing duplicate file: {txt_path} (keeping {md_path})")
            os.remove(txt_path)
        else:
            print(f"Duplicate file already removed: {txt_path}")
    
    # 3. Split large files
    for file_path in large_files:
        if not os.path.exists(file_path):
            print(f"Large file already processed: {file_path}")
            continue
            
        print(f"Splitting large file: {file_path}")
        split_files = split_large_document(file_path)
        
        # If we successfully split into multiple files, remove the original
        if len(split_files) > 1:
            print(f"  Split into {len(split_files)} smaller files")
            os.remove(file_path)
    
    print("\nCleanup complete!")
    print(f"A backup of the original docs folder has been saved to {backup_dir}")
    
if __name__ == "__main__":
    cleanup_docs_folder() 