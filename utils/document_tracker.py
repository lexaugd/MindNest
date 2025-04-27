"""
Document tracking for MindNest.
Tracks document changes to enable incremental updates to the vector store.
"""

import os
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Set, Optional, Union
from pathlib import Path

from utils.config import config
from utils.logger import get_logger

# Get module logger
logger = get_logger(__name__)

class DocumentTracker:
    """
    Tracks document changes in the docs directory to enable incremental updates
    to the vector store rather than full rebuilds.
    """
    
    def __init__(
        self, 
        docs_dir: Optional[Union[str, Path]] = None, 
        tracking_file: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the document tracker.
        
        Args:
            docs_dir: Directory containing the documents to track (default: from config)
            tracking_file: File to store tracking information (default: utils/doc_tracking.json)
        """
        self.docs_dir = str(docs_dir if docs_dir is not None else config.docs_dir)
        self.tracking_file = str(tracking_file if tracking_file is not None else "utils/doc_tracking.json")
        self.tracking_data = self._load_tracking_data()
    
    def _load_tracking_data(self) -> Dict:
        """Load existing tracking data or create new tracking data structure."""
        if os.path.exists(self.tracking_file):
            try:
                with open(self.tracking_file, 'r') as f:
                    data = json.load(f)
                logger.info(f"Loaded tracking data for {len(data.get('files', {}))} documents")
                return data
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Error loading tracking data: {e}. Creating new tracking data.")
        
        # Create tracking data structure if it doesn't exist or can't be loaded
        logger.info("Creating new tracking data")
        return {
            "last_update": "",
            "files": {}
        }
    
    def _save_tracking_data(self) -> None:
        """Save tracking data to the tracking file."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.tracking_file), exist_ok=True)
        
        try:
            with open(self.tracking_file, 'w') as f:
                json.dump(self.tracking_data, f, indent=2)
            logger.debug(f"Saved tracking data for {len(self.tracking_data.get('files', {}))} documents")
        except IOError as e:
            logger.error(f"Error saving tracking data: {e}")
    
    def _calculate_file_hash(self, filepath: str) -> str:
        """Calculate MD5 hash of file contents for change detection."""
        hash_md5 = hashlib.md5()
        try:
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except IOError as e:
            logger.error(f"Error calculating hash for {filepath}: {e}")
            return ""
    
    def scan_documents(self) -> Dict[str, List[str]]:
        """
        Scan the documents directory and identify new, modified, and deleted files.
        
        Returns:
            Dict with lists of new, modified, and deleted file paths
        """
        changes = {
            "new": [],
            "modified": [],
            "deleted": []
        }
        
        logger.info(f"Scanning documents in {self.docs_dir}")
        
        # Get current files in the docs directory
        current_files = set()
        try:
            for root, _, files in os.walk(self.docs_dir):
                for file in files:
                    # Skip hidden files and directories
                    if file.startswith('.'):
                        continue
                        
                    filepath = os.path.join(root, file)
                    current_files.add(filepath)
                    file_hash = self._calculate_file_hash(filepath)
                    
                    # Skip files with empty hash (likely unreadable)
                    if not file_hash:
                        continue
                    
                    file_mtime = os.path.getmtime(filepath)
                    
                    if filepath not in self.tracking_data["files"]:
                        # New file
                        changes["new"].append(filepath)
                        self.tracking_data["files"][filepath] = {
                            "hash": file_hash,
                            "last_modified": file_mtime,
                            "last_indexed": datetime.now().isoformat()
                        }
                        logger.debug(f"New file detected: {filepath}")
                    elif file_hash != self.tracking_data["files"][filepath]["hash"]:
                        # Modified file
                        changes["modified"].append(filepath)
                        self.tracking_data["files"][filepath]["hash"] = file_hash
                        self.tracking_data["files"][filepath]["last_modified"] = file_mtime
                        self.tracking_data["files"][filepath]["last_indexed"] = datetime.now().isoformat()
                        logger.debug(f"Modified file detected: {filepath}")
        except Exception as e:
            logger.error(f"Error scanning documents directory: {e}")
        
        # Check for deleted files
        tracked_files = set(self.tracking_data["files"].keys())
        deleted_files = tracked_files - current_files
        
        for filepath in deleted_files:
            changes["deleted"].append(filepath)
            del self.tracking_data["files"][filepath]
            logger.debug(f"Deleted file detected: {filepath}")
        
        # Update the last update timestamp
        self.tracking_data["last_update"] = datetime.now().isoformat()
        
        # Save changes to tracking file
        self._save_tracking_data()
        
        # Log summary of changes
        logger.info(f"Document scan results: {len(changes['new'])} new, "
                   f"{len(changes['modified'])} modified, {len(changes['deleted'])} deleted")
        
        return changes
    
    def get_file_extensions(self) -> Dict[str, int]:
        """Get a count of files by extension for reporting purposes."""
        extensions = {}
        for filepath in self.tracking_data["files"]:
            ext = os.path.splitext(filepath)[1].lower()
            if ext in extensions:
                extensions[ext] += 1
            else:
                extensions[ext] = 1
        return extensions
    
    def get_document_count(self) -> int:
        """Get the total number of tracked documents."""
        return len(self.tracking_data["files"])
    
    def get_last_update(self) -> str:
        """Get the timestamp of the last update to the tracking data."""
        return self.tracking_data["last_update"]
    
    def get_file_metadata(self, filepath: str) -> Optional[Dict]:
        """Get metadata for a specific file."""
        return self.tracking_data["files"].get(filepath)
    
    def get_all_metadata(self) -> Dict[str, Dict]:
        """Get metadata for all tracked files."""
        return self.tracking_data["files"]


# For testing purposes
if __name__ == "__main__":
    tracker = DocumentTracker()
    changes = tracker.scan_documents()
    
    print("\nDocument tracking scan results:")
    print(f"New files: {len(changes['new'])}")
    print(f"Modified files: {len(changes['modified'])}")
    print(f"Deleted files: {len(changes['deleted'])}")
    print(f"Total tracked documents: {tracker.get_document_count()}")
    
    ext_counts = tracker.get_file_extensions()
    print("\nFile extensions:")
    for ext, count in sorted(ext_counts.items()):
        print(f"  {ext}: {count}")
        
    print(f"\nLast update: {tracker.get_last_update()}") 