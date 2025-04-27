import os
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Set, Optional

class DocumentTracker:
    """
    Tracks document changes in the docs directory to enable incremental updates
    to the vector store rather than full rebuilds.
    """
    
    def __init__(self, docs_dir: str = "docs", tracking_file: str = "utils/doc_tracking.json"):
        """
        Initialize the document tracker.
        
        Args:
            docs_dir: Directory containing the documents to track
            tracking_file: File to store tracking information
        """
        self.docs_dir = docs_dir
        self.tracking_file = tracking_file
        self.tracking_data = self._load_tracking_data()
    
    def _load_tracking_data(self) -> Dict:
        """Load existing tracking data or create new tracking data structure."""
        if os.path.exists(self.tracking_file):
            try:
                with open(self.tracking_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading tracking data: {e}. Creating new tracking data.")
        
        # Create tracking data structure if it doesn't exist or can't be loaded
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
        except IOError as e:
            print(f"Error saving tracking data: {e}")
    
    def _calculate_file_hash(self, filepath: str) -> str:
        """Calculate MD5 hash of file contents for change detection."""
        hash_md5 = hashlib.md5()
        try:
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except IOError as e:
            print(f"Error calculating hash for {filepath}: {e}")
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
        
        # Get current files in the docs directory
        current_files = set()
        for root, _, files in os.walk(self.docs_dir):
            for file in files:
                # Skip hidden files and directories
                if file.startswith('.'):
                    continue
                    
                filepath = os.path.join(root, file)
                current_files.add(filepath)
                file_hash = self._calculate_file_hash(filepath)
                file_mtime = os.path.getmtime(filepath)
                
                if filepath not in self.tracking_data["files"]:
                    # New file
                    changes["new"].append(filepath)
                    self.tracking_data["files"][filepath] = {
                        "hash": file_hash,
                        "last_modified": file_mtime,
                        "last_indexed": datetime.now().isoformat()
                    }
                elif file_hash != self.tracking_data["files"][filepath]["hash"]:
                    # Modified file
                    changes["modified"].append(filepath)
                    self.tracking_data["files"][filepath]["hash"] = file_hash
                    self.tracking_data["files"][filepath]["last_modified"] = file_mtime
                    self.tracking_data["files"][filepath]["last_indexed"] = datetime.now().isoformat()
        
        # Check for deleted files
        tracked_files = set(self.tracking_data["files"].keys())
        deleted_files = tracked_files - current_files
        
        for filepath in deleted_files:
            changes["deleted"].append(filepath)
            del self.tracking_data["files"][filepath]
        
        # Update the last update timestamp
        self.tracking_data["last_update"] = datetime.now().isoformat()
        
        # Save changes to tracking file
        self._save_tracking_data()
        
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


# For testing purposes
if __name__ == "__main__":
    tracker = DocumentTracker()
    changes = tracker.scan_documents()
    
    print(f"Document tracking scan results:")
    print(f"New files: {len(changes['new'])}")
    print(f"Modified files: {len(changes['modified'])}")
    print(f"Deleted files: {len(changes['deleted'])}")
    print(f"Total tracked documents: {tracker.get_document_count()}")
    print(f"File extensions: {tracker.get_file_extensions()}")
    print(f"Last update: {tracker.get_last_update()}") 