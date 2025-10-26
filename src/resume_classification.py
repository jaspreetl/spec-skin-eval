#!/usr/bin/env python3
"""
Script to resume Claude classification from where it left off.
This will process the remaining images after the error.
"""

import os
import json
import pandas as pd
from claude_skin_classifier import ClaudeSkinClassifier
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def main():
    """Resume classification from where it left off."""
    
    # Check if API key is provided
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("Error: Please set the ANTHROPIC_API_KEY environment variable")
        return False
    
    # Load existing results
    results_file = "claude_classification_results.json"
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            existing_results = json.load(f)
        processed_ids = {r['image_id'] for r in existing_results}
        print(f"Found {len(existing_results)} existing results")
    else:
        existing_results = []
        processed_ids = set()
        print("No existing results found, starting from beginning")
    
    # Load ground truth data
    ground_truth_csv = "balanced_dataset/ground_truth_labels.csv"
    df = pd.read_csv(ground_truth_csv)
    
    # Find remaining images to process
    remaining_df = df[~df['dataset_id'].isin(processed_ids)]
    print(f"Remaining images to process: {len(remaining_df)}")
    
    if len(remaining_df) == 0:
        print("All images have been processed!")
        return True
    
    # Initialize classifier
    print("Initializing Claude skin classifier...")
    classifier = ClaudeSkinClassifier(api_key=api_key)
    
    # Process remaining images
    print(f"Processing remaining {len(remaining_df)} images...")
    print("=" * 60)
    
    for idx, row in remaining_df.iterrows():
        image_id = row['dataset_id']
        print(f"Processing image {image_id + 1}/300: {row['image_id']}")
        
        # Check if image file exists
        image_path = row['path']
        if not os.path.exists(image_path):
            print(f"  ❌ Image not found: {image_path}")
            result = classifier._create_error_result(row, "Image file not found")
            existing_results.append(result)
        else:
            # Classify the image
            result = classifier.classify_image(image_path, row)
            
            if result['success']:
                print(f"  ✅ Classification: {result['normalized_prediction']} ({result['predicted_subtype']})")
                print(f"     Ground truth: {result['ground_truth_label']} ({result['ground_truth_subtype']})")
                print(f"     Correct: {result['is_correct']}")
                print(f"     Time: {result['processing_time_ms']}ms")
            else:
                print(f"  ❌ Error: {result['error']}")
            
            existing_results.append(result)
        
        # Save intermediate results every 10 images
        if (len(existing_results) - len(processed_ids)) % 10 == 0:
            with open(results_file, 'w') as f:
                json.dump(existing_results, f, indent=2)
            print(f"  💾 Saved intermediate results ({len(existing_results)} total)")
        
        print("-" * 40)
    
    # Save final results
    with open(results_file, 'w') as f:
        json.dump(existing_results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("CLASSIFICATION SUMMARY")
    print("=" * 60)
    print(f"Total images processed: {len(existing_results)}")
    
    successful = sum(1 for r in existing_results if r['success'])
    failed = sum(1 for r in existing_results if not r['success'])
    print(f"Successful classifications: {successful}")
    print(f"Failed classifications: {failed}")
    print(f"Success rate: {(successful/len(existing_results))*100:.1f}%")
    
    if successful > 0:
        correct_predictions = sum(1 for r in existing_results if r['is_correct'])
        accuracy = (correct_predictions / successful) * 100
        print(f"Accuracy: {accuracy:.1f}% ({correct_predictions}/{successful})")
    
    print(f"\nResults saved to: {results_file}")
    
    # Run analysis
    print("\nRunning analysis...")
    os.system(f"python src/analyze_claude_results.py --results {results_file}")
    
    return True

if __name__ == "__main__":
    main()
