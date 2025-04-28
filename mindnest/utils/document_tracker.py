"""
Document tracking utility for MindNest
"""

import os
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

class DocumentTracker:
    """
    Utility class to track document statistics and metadata
    """
    
    def __init__(self, docs_dir: str = "docs"):
        """Initialize the document tracker"""
        self.docs_dir = docs_dir
        self.last_update_time = None
        self._stats = {}
    
    def _scan_documents(self):
        """Scan the documents directory to collect statistics"""
        if not os.path.exists(self.docs_dir):
            self._stats = {
                "count": 0,
                "extensions": {},
                "last_update": None
            }
            return

        extensions = {}
        count = 0
        latest_mtime = 0
        
        for root, _, files in os.walk(self.docs_dir):
            for file in files:
                if file.startswith('.'):  # Skip hidden files
                    continue
                    
                file_path = os.path.join(root, file)
                count += 1
                
                # Track file extensions
                ext = os.path.splitext(file)[1].lower()
                if ext in extensions:
                    extensions[ext] += 1
                else:
                    extensions[ext] = 1
                
                # Track last modification time
                try:
                    mtime = os.path.getmtime(file_path)
                    latest_mtime = max(latest_mtime, mtime)
                except:
                    pass
        
        # Store the stats
        self._stats = {
            "count": count,
            "extensions": extensions,
            "last_update": datetime.fromtimestamp(latest_mtime).isoformat() if latest_mtime > 0 else None
        }
        
        self.last_update_time = time.time()
    
    def get_document_count(self) -> int:
        """Get the total number of documents"""
        if self.last_update_time is None or time.time() - self.last_update_time > 300:  # Refresh every 5 minutes
            self._scan_documents()
        return self._stats.get("count", 0)
    
    def get_file_extensions(self) -> Dict[str, int]:
        """Get a dictionary of file extensions and counts"""
        if self.last_update_time is None or time.time() - self.last_update_time > 300:
            self._scan_documents()
        return self._stats.get("extensions", {})
    
    def get_last_update(self) -> Optional[str]:
        """Get the timestamp of the most recently updated document"""
        if self.last_update_time is None or time.time() - self.last_update_time > 300:
            self._scan_documents()
        return self._stats.get("last_update", None)
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get all document statistics"""
        if self.last_update_time is None or time.time() - self.last_update_time > 300:
            self._scan_documents()
        return self._stats 