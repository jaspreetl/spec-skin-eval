#!/usr/bin/env python3
"""
Script to analyze GPT classification results and generate evaluation metrics.
"""

import json
import pandas as pd
from collections import Counter
import argparse
import os

def load_results(json_file):
    """Load classification results from JSON file."""
    with open(json_file, 'r') as f:
        return json.load(f)

def load_csv_results(csv_file):
    """Load simple CSV results and convert to analysis format."""
    df = pd.read_csv(csv_file)
    results = []
    
    for _, row in df.iterrows():
        # Handle both old and new CSV formats
        ground_truth_col = 'ground_truth_label' if 'ground_truth_label' in row else 'true_label'
        
        # Convert CSV format to analysis format
        success = row['predicted_label'] not in ['ERROR', 'FILE_NOT_FOUND', 'QUOTA_EXCEEDED']
        is_correct = success and (row['predicted_label'] == row[ground_truth_col])
        
        result = {
            'image_id': row.get('image_id', ''),
            'image_filename': os.path.basename(row.get('path', '')),
            'image_path': row.get('path', ''),
            'ground_truth_label': row[ground_truth_col],
            'predicted_label': row['predicted_label'],
            'is_correct': is_correct,
            'success': success,
            'error': None if success else row['predicted_label'],
            'processing_time_ms': row.get('processing_time_ms', 0),
            'api_cost_usd': row.get('api_cost_usd', 0.0),
            'timestamp': row.get('timestamp', '')
        }
        results.append(result)
    
    return results

def analyze_results(results):
    """Analyze classification results and generate metrics."""
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    print("GPT SKIN CLASSIFICATION ANALYSIS")
    print("=" * 50)
    
    # Basic statistics
    total_images = len(df)
    successful = len(df[df['success'] == True])
    failed = len(df[df['success'] == False])
    
    print(f"Total images processed: {total_images}")
    print(f"Successful classifications: {successful}")
    print(f"Failed classifications: {failed}")
    print(f"Success rate: {(successful/total_images)*100:.1f}%")
    
    if successful == 0:
        print("No successful classifications to analyze.")
        return
    
    # Accuracy analysis
    successful_df = df[df['success'] == True]
    correct_predictions = len(successful_df[successful_df['is_correct'] == True])
    accuracy = (correct_predictions / successful) * 100
    
    print(f"\nACCURACY ANALYSIS")
    print("-" * 30)
    print(f"Overall accuracy: {accuracy:.1f}% ({correct_predictions}/{successful})")
    
    # Accuracy by category
    print(f"\nAccuracy by ground truth category:")
    for category in successful_df['ground_truth_label'].unique():
        cat_df = successful_df[successful_df['ground_truth_label'] == category]
        cat_correct = len(cat_df[cat_df['is_correct'] == True])
        cat_accuracy = (cat_correct / len(cat_df)) * 100
        print(f"  {category}: {cat_accuracy:.1f}% ({cat_correct}/{len(cat_df)})")
    
    # Confusion matrix
    print(f"\nCONFUSION MATRIX")
    print("-" * 30)
    confusion = pd.crosstab(
        successful_df['ground_truth_label'], 
        successful_df['predicted_label'], 
        margins=True
    )
    print(confusion)
    
    # Most common misclassifications
    print(f"\nMOST COMMON MISCLASSIFICATIONS")
    print("-" * 30)
    incorrect_df = successful_df[successful_df['is_correct'] == False]
    if len(incorrect_df) > 0:
        misclass_counts = Counter()
        for _, row in incorrect_df.iterrows():
            misclass_counts[(row['ground_truth_label'], row['predicted_label'])] += 1
        
        print("Top 5 misclassifications (Ground Truth → Predicted):")
        for (gt, pred), count in misclass_counts.most_common(5):
            print(f"  {gt} → {pred}: {count} times")
    else:
        print("No misclassifications found!")
    
    # Processing time analysis
    print(f"\nPROCESSING TIME ANALYSIS")
    print("-" * 30)
    processing_times = successful_df['processing_time_ms'].dropna()
    if len(processing_times) > 0:
        print(f"  Average time: {processing_times.mean():.1f}ms")
        print(f"  Median time: {processing_times.median():.1f}ms")
        print(f"  Min time: {processing_times.min():.1f}ms")
        print(f"  Max time: {processing_times.max():.1f}ms")
    
    # Cost analysis
    print(f"\nCOST ANALYSIS")
    print("-" * 30)
    costs = successful_df['api_cost_usd'].dropna()
    if len(costs) > 0:
        total_cost = costs.sum()
        avg_cost = costs.mean()
        print(f"  Total cost: ${total_cost:.6f}")
        print(f"  Average cost per image: ${avg_cost:.6f}")
        print(f"  Estimated cost for 1000 images: ${avg_cost * 1000:.2f}")
    
    # Error analysis
    if failed > 0:
        print(f"\nERROR ANALYSIS")
        print("-" * 30)
        error_df = df[df['success'] == False]
        error_counts = Counter()
        for _, row in error_df.iterrows():
            error = row.get('error', row.get('predicted_label', 'Unknown'))
            error_counts[error] += 1
        
        for error, count in error_counts.most_common():
            print(f"  {error}: {count}")
    
    # Sample incorrect predictions
    incorrect_df = successful_df[successful_df['is_correct'] == False]
    if len(incorrect_df) > 0:
        print(f"\nSAMPLE INCORRECT PREDICTIONS")
        print("-" * 30)
        sample_size = min(5, len(incorrect_df))
        for _, row in incorrect_df.head(sample_size).iterrows():
            print(f"  Image: {row['image_filename']}")
            print(f"    Ground truth: {row['ground_truth_label']}")
            print(f"    Predicted: {row['predicted_label']}")
            if 'processing_time_ms' in row and pd.notna(row['processing_time_ms']):
                print(f"    Processing time: {row['processing_time_ms']:.1f}ms")
            print()
    
    # Performance summary
    print(f"\nPERFORMANCE SUMMARY")
    print("-" * 30)
    print(f"✅ Success Rate: {(successful/total_images)*100:.1f}%")
    print(f"🎯 Accuracy: {accuracy:.1f}%")
    if len(costs) > 0:
        print(f"💰 Total Cost: ${costs.sum():.6f}")
    if len(processing_times) > 0:
        print(f"⏱️  Avg Time: {processing_times.mean():.1f}ms")

def main():
    """Main function to analyze results."""
    parser = argparse.ArgumentParser(description='Analyze GPT classification results')
    parser.add_argument('--results', default='results/gpt_classification_results.json',
                       help='Path to results JSON file')
    parser.add_argument('--csv', default='results/gpt_predictions.csv',
                       help='Path to CSV results file (fallback)')
    
    args = parser.parse_args()
    
    try:
        # Try to load JSON results first
        if os.path.exists(args.results):
            print(f"Loading detailed results from: {args.results}")
            results = load_results(args.results)
        elif os.path.exists(args.csv):
            print(f"Loading CSV results from: {args.csv}")
            results = load_csv_results(args.csv)
        else:
            print(f"Error: No results files found.")
            print(f"Looked for:")
            print(f"  - {args.results}")
            print(f"  - {args.csv}")
            print("Please run the classification script first.")
            return
        
        analyze_results(results)
        
    except Exception as e:
        print(f"Error analyzing results: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
