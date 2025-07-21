import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

class ProcessingStats:
    """Track and analyze processing statistics for batch operations"""
    
    def __init__(self):
        self.processing_history = []
        self.current_batch = None
        
    def start_batch(self, batch_name: str, total_files: int) -> None:
        """Initialize new batch processing session"""
        self.current_batch = {
            'batch_name': batch_name,
            'start_time': time.time(),
            'total_files': total_files,
            'processed_files': 0,
            'successful_files': 0,
            'failed_files': 0,
            'processing_times': [],
            'errors': [],
            'file_details': []
        }
    
    def record_file_processing(self, file_path: str, success: bool, 
                             processing_time: float, error: Optional[str] = None) -> None:
        """Record processing result for individual file"""
        if self.current_batch is None:
            return
        
        self.current_batch['processed_files'] += 1
        self.current_batch['processing_times'].append(processing_time)
        
        file_record = {
            'file_path': file_path,
            'success': success,
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat()
        }
        
        if success:
            self.current_batch['successful_files'] += 1
        else:
            self.current_batch['failed_files'] += 1
            if error:
                file_record['error'] = error
                self.current_batch['errors'].append(f"{Path(file_path).name}: {error}")
        
        self.current_batch['file_details'].append(file_record)
    
    def finish_batch(self) -> Dict[str, Any]:
        """Complete batch processing and calculate final statistics"""
        if self.current_batch is None:
            return {}
        
        end_time = time.time()
        total_time = end_time - self.current_batch['start_time']
        
        # Calculate statistics
        processing_times = self.current_batch['processing_times']
        batch_results = {
            'batch_name': self
