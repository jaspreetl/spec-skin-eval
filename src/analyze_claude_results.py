#!/usr/bin/env python3
"""
Script to analyze Claude classification results and generate evaluation metrics.
"""

import json
import pandas as pd
from collections import Counter
import argparse

def load_results(json_file):
    """Load classification results from JSON file."""
    with open(json_file, 'r') as f:
        return json.load(f)

def analyze_results(results):
    """Analyze classification results and generate metrics."""
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    print("CLAUDE SKIN CLASSIFICATION ANALYSIS")
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
        successful_df['normalized_prediction'], 
        margins=True
    )
    print(confusion)
    
    # Subtype analysis for acne
    acne_df = successful_df[successful_df['ground_truth_label'] == 'acne']
    if len(acne_df) > 0:
        print(f"\nACNE SUBTYPE ANALYSIS")
        print("-" * 30)
        print("Ground truth vs Predicted acne subtypes:")
        subtype_confusion = pd.crosstab(
            acne_df['ground_truth_subtype'],
            acne_df['predicted_subtype'],
            margins=True
        )
        print(subtype_confusion)
    
    # Confidence analysis
    print(f"\nCONFIDENCE ANALYSIS")
    print("-" * 30)
    confidence_counts = Counter()
    for _, row in successful_df.iterrows():
        if row['api_response'] and 'confidence' in row['api_response']:
            confidence_counts[row['api_response']['confidence']] += 1
    
    for conf_level, count in confidence_counts.most_common():
        percentage = (count / successful) * 100
        print(f"  {conf_level}: {count} ({percentage:.1f}%)")
    
    # Processing time analysis
    print(f"\nPROCESSING TIME ANALYSIS")
    print("-" * 30)
    processing_times = successful_df['processing_time_ms'].dropna()
    if len(processing_times) > 0:
        print(f"  Average time: {processing_times.mean():.1f}ms")
        print(f"  Median time: {processing_times.median():.1f}ms")
        print(f"  Min time: {processing_times.min():.1f}ms")
        print(f"  Max time: {processing_times.max():.1f}ms")
    
    # Error analysis
    if failed > 0:
        print(f"\nERROR ANALYSIS")
        print("-" * 30)
        error_df = df[df['success'] == False]
        error_counts = Counter(error_df['error'].dropna())
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
            print(f"    Ground truth: {row['ground_truth_label']} ({row['ground_truth_subtype']})")
            print(f"    Predicted: {row['normalized_prediction']} ({row['predicted_subtype']})")
            if row['api_response'] and 'reasoning' in row['api_response']:
                print(f"    Reasoning: {row['api_response']['reasoning']}")
            print()

def main():
    """Main function to analyze results."""
    parser = argparse.ArgumentParser(description='Analyze Claude classification results')
    parser.add_argument('--results', default='claude_classification_results.json',
                       help='Path to results JSON file')
    
    args = parser.parse_args()
    
    try:
        results = load_results(args.results)
        analyze_results(results)
    except FileNotFoundError:
        print(f"Error: Results file not found at {args.results}")
        print("Please run the classification script first.")
    except Exception as e:
        print(f"Error analyzing results: {e}")

if __name__ == "__main__":
    main()
