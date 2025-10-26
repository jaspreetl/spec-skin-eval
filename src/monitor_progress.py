#!/usr/bin/env python3
"""
Monitor the progress of the Claude classification.
"""

import json
import os
import time
from datetime import datetime

def monitor_progress():
    """Monitor the classification progress."""
    results_file = "claude_classification_results.json"
    
    if not os.path.exists(results_file):
        print("No results file found yet. Classification may not have started.")
        return
    
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        total_processed = len(results)
        successful = len([r for r in results if r.get('success', False)])
        failed = total_processed - successful
        
        print(f"📊 CLASSIFICATION PROGRESS - {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 60)
        print(f"Images processed: {total_processed}/300")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Progress: {(total_processed/300)*100:.1f}%")
        
        if total_processed > 0:
            # Check if we have full raw responses
            has_raw_response = 'full_raw_response' in results[0]
            print(f"Raw responses captured: {'✅ Yes' if has_raw_response else '❌ No'}")
            
            if has_raw_response:
                sample_raw = results[0]['full_raw_response']
                print(f"Sample raw response length: {len(sample_raw)} characters")
                print(f"Sample raw response preview: {sample_raw[:100]}...")
        
        # Estimate time remaining
        if total_processed > 0:
            avg_time_per_image = 4.5  # seconds
            remaining_images = 300 - total_processed
            estimated_remaining_minutes = (remaining_images * avg_time_per_image) / 60
            print(f"Estimated time remaining: {estimated_remaining_minutes:.1f} minutes")
        
        print()
        
    except Exception as e:
        print(f"Error reading results: {e}")

if __name__ == "__main__":
    monitor_progress()
