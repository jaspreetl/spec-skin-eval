#!/usr/bin/env python3
"""
Script to prepare a balanced dataset for skin condition evaluation.
Creates a curated dataset with specific distribution of acne and non-acne conditions.
"""

import pandas as pd
import numpy as np
import os
import shutil
from pathlib import Path
import random

def setup_random_seed(seed=42):
    """Set random seed for reproducible sampling."""
    random.seed(seed)
    np.random.seed(seed)

def load_metadata(metadata_path):
    """Load the metadata CSV file."""
    df = pd.read_csv(metadata_path)
    print(f"Loaded {len(df)} total images")
    print(f"Available conditions: {df['condition'].nunique()}")
    return df

def sample_acne_images(df, acne_types_mapping):
    """
    Sample True Acne images according to the specified distribution.
    
    Args:
        df: DataFrame with all images
        acne_types_mapping: Dictionary mapping acne types to counts
        
    Returns:
        List of sampled image records
    """
    acne_df = df[df['condition'] == 'Acne'].copy()
    print(f"\nFound {len(acne_df)} total acne images")
    
    sampled_images = []
    
    # Sample each acne type
    for acne_type, target_count in acne_types_mapping.items():
        # Filter images for this specific acne type
        type_images = acne_df[acne_df['image_id'].apply(
            lambda x: extract_acne_type_from_name(os.path.splitext(x)[0]) == acne_type
        )]
        
        available_count = len(type_images)
        actual_count = min(target_count, available_count)
        
        if actual_count < target_count:
            print(f"Warning: Only {actual_count} {acne_type} images available (requested {target_count})")
        
        # Sample the images
        if actual_count > 0:
            sampled = type_images.sample(n=actual_count, random_state=42)
            sampled_images.extend(sampled.to_dict('records'))
            print(f"Sampled {actual_count} {acne_type} images")
        else:
            print(f"No {acne_type} images found")
    
    return sampled_images

def sample_condition_images(df, condition, target_count, label_prefix=""):
    """
    Sample images from a specific condition.
    
    Args:
        df: DataFrame with all images
        condition: Condition name to sample from
        target_count: Number of images to sample
        label_prefix: Optional prefix for the label
        
    Returns:
        List of sampled image records
    """
    condition_df = df[df['condition'] == condition]
    available_count = len(condition_df)
    actual_count = min(target_count, available_count)
    
    if actual_count < target_count:
        print(f"Warning: Only {actual_count} {condition} images available (requested {target_count})")
    
    if actual_count > 0:
        sampled = condition_df.sample(n=actual_count, random_state=42)
        records = sampled.to_dict('records')
        
        # Add label prefix if specified
        if label_prefix:
            for record in records:
                record['evaluation_label'] = f"{label_prefix}_{condition}"
        else:
            for record in records:
                record['evaluation_label'] = condition
                
        return records
    else:
        print(f"No {condition} images found")
        return []

def extract_acne_type_from_name(name):
    """Extract acne type from image filename (reused from extract_acne_types.py)."""
    name_lower = name.lower()
    
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
    
    for pattern, acne_type in patterns.items():
        if pattern in name_lower:
            return acne_type
    
    if 'acne' in name_lower:
        return 'General Acne'
    
    if 'rosacea' in name_lower:
        return 'Rosacea'
    
    if any(term in name_lower for term in ['comedo', 'comedone', 'blackhead', 'whitehead']):
        return 'Comedo'
    
    if any(term in name_lower for term in ['cyst', 'cystic']):
        return 'Cystic Lesion'
    
    if any(term in name_lower for term in ['pustule', 'pustular']):
        return 'Pustular Acne'
    
    if any(term in name_lower for term in ['papule', 'papular']):
        return 'Papular Lesion'
    
    if any(term in name_lower for term in ['nodule', 'nodular']):
        return 'Nodular Lesion'
    
    if name_lower.startswith('acne-'):
        return f"Acne ({name_lower.replace('acne-', '').replace('-', ' ').title()})"
    
    return 'General Acne'

def create_balanced_dataset(metadata_path, output_dir="balanced_dataset"):
    """Create the balanced dataset according to specifications."""
    
    # Set random seed for reproducibility
    setup_random_seed(42)
    
    # Load metadata
    df = load_metadata(metadata_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    
    all_sampled_images = []
    
    print("\n" + "="*60)
    print("SAMPLING TRUE ACNE IMAGES (100 total)")
    print("="*60)
    
    # True Acne distribution
    acne_types = {
        'Cystic Acne': 30,
        'Pustular Acne': 25,
        'Open Comedo (Blackhead)': 20,
        'Closed Comedo (Whitehead)': 15,
        'General Acne': 10
    }
    
    true_acne_images = sample_acne_images(df, acne_types)
    
    # Add evaluation labels for true acne
    for img in true_acne_images:
        img['evaluation_label'] = f"True_Acne_{img['image_id']}"
        img['category'] = 'True Acne'
    
    all_sampled_images.extend(true_acne_images)
    print(f"Total True Acne images sampled: {len(true_acne_images)}")
    
    print("\n" + "="*60)
    print("SAMPLING ACNE-LIKE/CONFUSABLE IMAGES (50 total)")
    print("="*60)
    
    # Acne-like/Confusable distribution
    confusable_conditions = {
        'Rosacea': 30,
        'Perioral Dermatitis': 20
    }
    
    confusable_images = []
    for condition, count in confusable_conditions.items():
        if condition == 'Perioral Dermatitis':
            # This is actually part of Acne condition, need to filter from acne images
            acne_df = df[df['condition'] == 'Acne']
            perioral_images = acne_df[acne_df['image_id'].str.contains('perioral-dermatitis', na=False)]
            if len(perioral_images) > 0:
                sampled = perioral_images.sample(n=min(count, len(perioral_images)), random_state=42)
                records = sampled.to_dict('records')
                for record in records:
                    record['evaluation_label'] = f"Acne_like_{condition.replace(' ', '_')}"
                    record['category'] = 'Acne-like/Confusable'
                confusable_images.extend(records)
                print(f"Sampled {len(records)} {condition} images")
        else:
            # For Rosacea, it's also part of Acne condition
            acne_df = df[df['condition'] == 'Acne']
            rosacea_images = acne_df[acne_df['image_id'].str.contains('rosacea', na=False)]
            if len(rosacea_images) > 0:
                sampled = rosacea_images.sample(n=min(count, len(rosacea_images)), random_state=42)
                records = sampled.to_dict('records')
                for record in records:
                    record['evaluation_label'] = f"Acne_like_{condition.replace(' ', '_')}"
                    record['category'] = 'Acne-like/Confusable'
                confusable_images.extend(records)
                print(f"Sampled {len(records)} {condition} images")
    
    all_sampled_images.extend(confusable_images)
    print(f"Total Acne-like/Confusable images sampled: {len(confusable_images)}")
    
    print("\n" + "="*60)
    print("SAMPLING NON-ACNE IMAGES (150 total)")
    print("="*60)
    
    # Inflammatory/Similar Conditions (70 images)
    print("\nInflammatory/Similar Conditions (70 images):")
    inflammatory_conditions = {
        'Atopic Dermatitis': 25,
        'Eczema': 20,
        'Psoriasis': 15,
        'Poison Ivy': 10
    }
    
    inflammatory_images = []
    for condition, count in inflammatory_conditions.items():
        records = sample_condition_images(df, condition, count, "Inflammatory")
        for record in records:
            record['category'] = 'Inflammatory/Similar'
        inflammatory_images.extend(records)
        print(f"Sampled {len(records)} {condition} images")
    
    all_sampled_images.extend(inflammatory_images)
    
    # Infections (30 images)
    print("\nInfections (30 images):")
    infection_conditions = {
        'Cellulitis': 15,
        'Ringworm': 10,
        'Herpes HPV': 5
    }
    
    infection_images = []
    for condition, count in infection_conditions.items():
        records = sample_condition_images(df, condition, count, "Infection")
        for record in records:
            record['category'] = 'Infection'
        infection_images.extend(records)
        print(f"Sampled {len(records)} {condition} images")
    
    all_sampled_images.extend(infection_images)
    
    # Serious Conditions (20 images)
    print("\nSerious Conditions (20 images):")
    serious_conditions = {
        'Melanoma': 15,
        'Actinic Carcinoma': 5
    }
    
    serious_images = []
    for condition, count in serious_conditions.items():
        records = sample_condition_images(df, condition, count, "Serious")
        for record in records:
            record['category'] = 'Serious'
        serious_images.extend(records)
        print(f"Sampled {len(records)} {condition} images")
    
    all_sampled_images.extend(serious_images)
    
    # Other Conditions (30 images)
    print("\nOther Conditions (30 images):")
    other_conditions = {
        'Vascular Tumors': 10,
        'Lupus': 10,
        'Light Diseases': 5,
        'Systemic Disease': 5
    }
    
    other_images = []
    for condition, count in other_conditions.items():
        records = sample_condition_images(df, condition, count, "Other")
        for record in records:
            record['category'] = 'Other'
        other_images.extend(records)
        print(f"Sampled {len(records)} {condition} images")
    
    all_sampled_images.extend(other_images)
    
    # Create final dataset
    final_df = pd.DataFrame(all_sampled_images)
    
    # Add some additional metadata
    final_df['dataset_id'] = range(len(final_df))
    final_df['original_condition'] = final_df['condition']
    
    # Add acne type column for acne-related conditions
    final_df['acne_type'] = None
    for idx, row in final_df.iterrows():
        if row['category'] in ['True Acne', 'Acne-like/Confusable']:
            final_df.at[idx, 'acne_type'] = extract_acne_type_from_name(os.path.splitext(row['image_id'])[0])
    
    # Save the ground truth CSV
    ground_truth_path = f"{output_dir}/ground_truth_labels.csv"
    final_df.to_csv(ground_truth_path, index=False)
    
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"Total images: {len(final_df)}")
    print(f"True Acne: {len(final_df[final_df['category'] == 'True Acne'])}")
    print(f"Acne-like/Confusable: {len(final_df[final_df['category'] == 'Acne-like/Confusable'])}")
    print(f"Inflammatory/Similar: {len(final_df[final_df['category'] == 'Inflammatory/Similar'])}")
    print(f"Infection: {len(final_df[final_df['category'] == 'Infection'])}")
    print(f"Serious: {len(final_df[final_df['category'] == 'Serious'])}")
    print(f"Other: {len(final_df[final_df['category'] == 'Other'])}")
    
    print(f"\nGround truth labels saved to: {ground_truth_path}")
    
    # Show distribution by evaluation label
    print("\nDistribution by evaluation label:")
    print(final_df['evaluation_label'].value_counts().sort_index())
    
    return final_df

def main():
    """Main function to create the balanced dataset."""
    # Use relative path from script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    metadata_path = os.path.join(project_root, "data", "metadata.csv")
    
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found at {metadata_path}")
        return
    
    print("Creating balanced dataset for skin condition evaluation...")
    print("This will sample 300 images total (150 acne + 150 non-acne)")
    
    # Create the balanced dataset
    balanced_df = create_balanced_dataset(metadata_path)
    
    print("\nDataset preparation complete!")
    print("The balanced dataset is ready for evaluation.")

if __name__ == "__main__":
    main()
