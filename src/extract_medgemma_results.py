#!/usr/bin/env python3
"""
Script to extract MedGemma classification results to CSV format matching Claude's detailed_results.csv
"""

import json
import pandas as pd
import os
import sys

def extract_medgemma_results_to_csv(json_file, output_csv):
    """Extract MedGemma results to CSV format matching Claude's detailed_results.csv"""
    
    if not os.path.exists(json_file):
        print(f"Error: Results file not found at {json_file}")
        return False
    
    print(f"Loading results from {json_file}...")
    with open(json_file, 'r') as f:
        results = json.load(f)
    
    if not results:
        print("Error: No results found in JSON file")
        return False
    
    print(f"Processing {len(results)} results...")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Extract the columns matching Claude's format
    detailed_results = []
    
    for _, row in df.iterrows():
        # Extract confidence and reasoning from api_response
        confidence = 'N/A'
        reasoning = 'N/A'
        
        if pd.notna(row.get('api_response')) and row.get('api_response'):
            api_response = row['api_response']
            if isinstance(api_response, dict):
                confidence = api_response.get('confidence', 'N/A')
                reasoning = api_response.get('reasoning', 'N/A')
        
        detailed_results.append({
            'image_id': row.get('image_id', ''),
            'image_filename': row.get('image_filename', ''),
            'ground_truth_label': row.get('ground_truth_label', ''),
            'ground_truth_subtype': row.get('ground_truth_subtype', ''),
            'normalized_prediction': row.get('normalized_prediction', ''),
            'predicted_subtype': row.get('predicted_subtype', ''),
            'is_correct': row.get('is_correct', False),
            'success': row.get('success', False),
            'processing_time_ms': row.get('processing_time_ms', 0),
            'confidence': confidence,
            'reasoning': reasoning
        })
    
    # Create DataFrame and save
    detailed_df = pd.DataFrame(detailed_results)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', exist_ok=True)
    
    detailed_df.to_csv(output_csv, index=False)
    print(f"✅ Detailed results saved to: {output_csv}")
    print(f"   Total rows: {len(detailed_df)}")
    print(f"   Successful: {detailed_df['success'].sum()}")
    print(f"   Correct: {detailed_df['is_correct'].sum()}")
    
    return True

def main():
    """Main function"""
    # Default paths
    json_file = "medgemma_classification_results.json"
    output_csv = "analysis_results/medgemma_detailed_results.csv"
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_csv = sys.argv[2]
    
    print("=" * 60)
    print("MedGemma Results Extraction")
    print("=" * 60)
    print(f"Input JSON: {json_file}")
    print(f"Output CSV: {output_csv}")
    print()
    
    success = extract_medgemma_results_to_csv(json_file, output_csv)
    
    if success:
        print("\n" + "=" * 60)
        print("✅ Extraction complete!")
        print("=" * 60)
    else:
        print("\n❌ Extraction failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()

