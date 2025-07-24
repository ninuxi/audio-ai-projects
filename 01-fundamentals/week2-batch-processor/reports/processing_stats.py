"""
Audio AI Projects - Advanced Audio Processing System
Copyright (c) 2025 Antonino Mainenti (ninuxi)
Licensed under MIT License - see LICENSE file
GitHub: https://github.com/ninuxi/audio-ai-projects
"""
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
            'batch_name': self.current_batch['batch_name'],
            'total_processing_time': total_time,
            'total_files': self.current_batch['total_files'],
            'processed_files': self.current_batch['processed_files'],
            'successful_files': self.current_batch['successful_files'],
            'failed_files': self.current_batch['failed_files'],
            'success_rate': (self.current_batch['successful_files'] / max(1, self.current_batch['processed_files'])) * 100,
            'average_processing_time': sum(processing_times) / max(1, len(processing_times)),
            'min_processing_time': min(processing_times) if processing_times else 0,
            'max_processing_time': max(processing_times) if processing_times else 0,
            'throughput': self.current_batch['processed_files'] / max(1, total_time),  # files per second
            'errors': self.current_batch['errors'],
            'file_details': self.current_batch['file_details'],
            'completion_time': datetime.now().isoformat()
        }
        
        # Add to processing history
        self.processing_history.append(batch_results)
        
        # Reset current batch
        self.current_batch = None
        
        return batch_results
    
    def get_processing_efficiency(self) -> Dict[str, Any]:
        """Analyze processing efficiency across batches"""
        if not self.processing_history:
            return {'message': 'No processing history available'}
        
        efficiency_stats = {
            'total_batches': len(self.processing_history),
            'total_files_processed': sum(b['processed_files'] for b in self.processing_history),
            'overall_success_rate': 0,
            'average_throughput': 0,
            'performance_trends': [],
            'bottlenecks': []
        }
        
        # Calculate overall metrics
        total_successful = sum(b['successful_files'] for b in self.processing_history)
        total_processed = sum(b['processed_files'] for b in self.processing_history)
        
        if total_processed > 0:
            efficiency_stats['overall_success_rate'] = (total_successful / total_processed) * 100
        
        # Average throughput
        throughputs = [b['throughput'] for b in self.processing_history]
        efficiency_stats['average_throughput'] = sum(throughputs) / len(throughputs)
        
        # Performance trends
        for i, batch in enumerate(self.processing_history):
            efficiency_stats['performance_trends'].append({
                'batch_number': i + 1,
                'batch_name': batch['batch_name'],
                'success_rate': batch['success_rate'],
                'throughput': batch['throughput'],
                'average_time': batch['average_processing_time']
            })
        
        # Identify bottlenecks
        slow_batches = [b for b in self.processing_history if b['average_processing_time'] > 5.0]
        if slow_batches:
            efficiency_stats['bottlenecks'].append(f"Slow processing detected in {len(slow_batches)} batches")
        
        low_success_batches = [b for b in self.processing_history if b['success_rate'] < 90]
        if low_success_batches:
            efficiency_stats['bottlenecks'].append(f"Low success rate in {len(low_success_batches)} batches")
        
        return efficiency_stats
    
    def save_processing_report(self, output_file: str) -> str:
        """Save comprehensive processing report"""
        report = {
            'processing_history': self.processing_history,
            'efficiency_analysis': self.get_processing_efficiency(),
            'report_generated': datetime.now().isoformat()
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return output_file
    
    def get_current_progress(self) -> Dict[str, Any]:
        """Get current batch processing progress"""
        if self.current_batch is None:
            return {'message': 'No active batch processing'}
        
        elapsed_time = time.time() - self.current_batch['start_time']
        processed = self.current_batch['processed_files']
        total = self.current_batch['total_files']
        
        progress = {
            'batch_name': self.current_batch['batch_name'],
            'progress_percentage': (processed / max(1, total)) * 100,
            'processed_files': processed,
            'total_files': total,
            'elapsed_time': elapsed_time,
            'estimated_remaining_time': 0,
            'current_success_rate': (self.current_batch['successful_files'] / max(1, processed)) * 100
        }
        
        # Estimate remaining time
        if processed > 0:
            avg_time_per_file = elapsed_time / processed
            remaining_files = total - processed
            progress['estimated_remaining_time'] = avg_time_per_file * remaining_files
        
        return progress
