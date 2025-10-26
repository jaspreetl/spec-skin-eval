#!/usr/bin/env python3
"""
Script to extract acne types from image names in the skin diseases dataset.
This script analyzes the image filenames to identify specific types of acne and related conditions.
"""

import pandas as pd
import re
from collections import Counter
import os

def extract_acne_types_from_metadata(metadata_path):
    """
    Extract acne types from image names in the metadata CSV file.
    
    Args:
        metadata_path (str): Path to the metadata CSV file
        
    Returns:
        dict: Dictionary with acne types as keys and counts as values
    """
    # Read the metadata CSV
    df = pd.read_csv(metadata_path)
    
    # Filter for acne-related entries
    acne_df = df[df['condition'] == 'Acne'].copy()
    
    print(f"Found {len(acne_df)} acne-related images")
    
    # Extract acne types from image names
    acne_types = []
    
    for image_id in acne_df['image_id']:
        # Remove file extension
        name = os.path.splitext(image_id)[0]
        
        # Extract acne type using various patterns
        acne_type = extract_acne_type_from_name(name)
        acne_types.append(acne_type)
    
    # Count occurrences
    type_counts = Counter(acne_types)
    
    return type_counts, acne_df

def extract_acne_type_from_name(name):
    """
    Extract acne type from image filename.
    
    Args:
        name (str): Image filename without extension
        
    Returns:
        str: Extracted acne type or 'unknown'
    """
    # Convert to lowercase for easier matching
    name_lower = name.lower()
    
    # Define patterns for different acne types
    patterns = {
        'acne-cystic': 'Cystic Acne',
        'acne-open-comedo': 'Open Comedo (Blackhead)',
        'acne-excoriated': 'Excoriated Acne',
        'acne-closed-comedo': 'Closed Comedo (Whitehead)',
        'acne-papular': 'Papular Acne',
        'acne-pustular': 'Pustular Acne',
        'acne-nodular': 'Nodular Acne',
        'acne-conglobata': 'Acne Conglobata',
        'acne-fulminans': 'Acne Fulminans',
        'hidradenitis-suppurativa': 'Hidradenitis Suppurativa',
        'perioral-dermatitis': 'Perioral Dermatitis',
        'rosacea': 'Rosacea',
        'milia': 'Milia',
        'sebaceous-glands': 'Sebaceous Glands',
        'hyperhidrosis': 'Hyperhidrosis',
        'rhnophymas': 'Rhinophyma'
    }
    
    # Check for specific patterns
    for pattern, acne_type in patterns.items():
        if pattern in name_lower:
            return acne_type
    
    # Check for general acne patterns
    if 'acne' in name_lower:
        return 'General Acne'
    
    # Check for rosacea patterns
    if 'rosacea' in name_lower:
        return 'Rosacea'
    
    # Check for other skin conditions that might be acne-related
    if any(term in name_lower for term in ['comedo', 'comedone', 'blackhead', 'whitehead']):
        return 'Comedo'
    
    if any(term in name_lower for term in ['cyst', 'cystic']):
        return 'Cystic Lesion'
    
    if any(term in name_lower for term in ['pustule', 'pustular']):
        return 'Pustular Lesion'
    
    if any(term in name_lower for term in ['papule', 'papular']):
        return 'Papular Lesion'
    
    if any(term in name_lower for term in ['nodule', 'nodular']):
        return 'Nodular Lesion'
    
    # If no specific pattern matches, return the original name or 'unknown'
    if name_lower.startswith('acne-'):
        return f"Acne ({name_lower.replace('acne-', '').replace('-', ' ').title()})"
    
    return 'Unknown'

def main():
    """Main function to extract and display acne types."""
    # Path to metadata file
    metadata_path = "/Users/bhavyagopal/Desktop/spec-skin-eval/data/metadata.csv"
    
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found at {metadata_path}")
        return
    
    print("Extracting acne types from image names...")
    print("=" * 50)
    
    # Extract acne types
    type_counts, acne_df = extract_acne_types_from_metadata(metadata_path)
    
    # Display results
    print(f"\nFound {len(type_counts)} different acne types:")
    print("-" * 50)
    
    # Sort by count (descending)
    sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
    
    for acne_type, count in sorted_types:
        print(f"{acne_type:<30} : {count:>3} images")
    
    print("\n" + "=" * 50)
    print(f"Total acne-related images: {sum(type_counts.values())}")
    
    # Save results to CSV
    output_path = "/Users/bhavyagopal/Desktop/spec-skin-eval/data/acne_types.csv"
    results_df = pd.DataFrame(sorted_types, columns=['acne_type', 'count'])
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    # Show some example image names for each type
    print("\nExample image names for each type:")
    print("-" * 50)
    
    for acne_type, count in sorted_types[:10]:  # Show top 10 types
        examples = acne_df[acne_df['image_id'].apply(
            lambda x: extract_acne_type_from_name(os.path.splitext(x)[0]) == acne_type
        )]['image_id'].head(3).tolist()
        
        print(f"\n{acne_type} ({count} images):")
        for example in examples:
            print(f"  - {example}")

if __name__ == "__main__":
    main()
