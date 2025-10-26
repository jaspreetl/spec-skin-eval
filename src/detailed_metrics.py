#!/usr/bin/env python3
"""
Detailed metrics analysis for Claude skin classification results.
Calculates precision, recall, F1-score, and detailed confusion matrices.
"""

import json
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def load_results(json_file):
    """Load classification results from JSON file."""
    with open(json_file, 'r') as f:
        return json.load(f)

def calculate_detailed_metrics(results):
    """Calculate detailed metrics for the classification results."""
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    print("DETAILED CLAUDE SKIN CLASSIFICATION METRICS")
    print("=" * 60)
    
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
    
    # Filter successful classifications
    successful_df = df[df['success'] == True]
    
    # Prepare data for sklearn metrics
    y_true = successful_df['ground_truth_label'].values
    y_pred = successful_df['normalized_prediction'].values
    
    # Calculate basic metrics
    accuracy = (y_true == y_pred).mean()
    
    print(f"\nOVERALL METRICS")
    print("-" * 30)
    print(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    # Calculate precision, recall, F1 for each class
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Get unique classes
    classes = np.unique(np.concatenate([y_true, y_pred]))
    
    print(f"\nPER-CLASS METRICS")
    print("-" * 30)
    print(f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 50)
    
    for i, class_name in enumerate(classes):
        if i < len(precision):
            print(f"{class_name:<12} {precision[i]:<10.3f} {recall[i]:<10.3f} {f1[i]:<10.3f}")
    
    # Macro and micro averages
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
    recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    print(f"\nAVERAGE METRICS")
    print("-" * 30)
    print(f"Macro Average - Precision: {precision_macro:.3f}, Recall: {recall_macro:.3f}, F1: {f1_macro:.3f}")
    print(f"Micro Average - Precision: {precision_micro:.3f}, Recall: {recall_micro:.3f}, F1: {f1_micro:.3f}")
    
    # Detailed confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    print(f"\nDETAILED CONFUSION MATRIX")
    print("-" * 30)
    print("Rows: Ground Truth, Columns: Predicted")
    print(f"{'':<12}", end="")
    for class_name in classes:
        print(f"{class_name:<12}", end="")
    print()
    
    for i, true_class in enumerate(classes):
        print(f"{true_class:<12}", end="")
        for j, pred_class in enumerate(classes):
            print(f"{cm[i,j]:<12}", end="")
        print()
    
    # Calculate metrics for acne vs non-acne
    print(f"\nACNE vs NON-ACNE ANALYSIS")
    print("-" * 30)
    
    # True Positives, False Positives, True Negatives, False Negatives
    tp = ((y_true == 'acne') & (y_pred == 'acne')).sum()
    fp = ((y_true == 'non_acne') & (y_pred == 'acne')).sum()
    tn = ((y_true == 'non_acne') & (y_pred == 'non_acne')).sum()
    fn = ((y_true == 'acne') & (y_pred == 'non_acne')).sum()
    
    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Negatives (FN): {fn}")
    
    # Calculate metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall for acne
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Recall for non-acne
    precision_acne = tp / (tp + fp) if (tp + fp) > 0 else 0
    precision_non_acne = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    print(f"\nSensitivity (Recall for Acne): {sensitivity:.3f} ({sensitivity*100:.1f}%)")
    print(f"Specificity (Recall for Non-Acne): {specificity:.3f} ({specificity*100:.1f}%)")
    print(f"Precision for Acne: {precision_acne:.3f} ({precision_acne*100:.1f}%)")
    print(f"Precision for Non-Acne: {precision_non_acne:.3f} ({precision_non_acne*100:.1f}%)")
    
    # F1 scores
    f1_acne = 2 * (precision_acne * sensitivity) / (precision_acne + sensitivity) if (precision_acne + sensitivity) > 0 else 0
    f1_non_acne = 2 * (precision_non_acne * specificity) / (precision_non_acne + specificity) if (precision_non_acne + specificity) > 0 else 0
    
    print(f"F1-Score for Acne: {f1_acne:.3f}")
    print(f"F1-Score for Non-Acne: {f1_non_acne:.3f}")
    
    # Analysis by category
    print(f"\nANALYSIS BY CATEGORY")
    print("-" * 30)
    
    # Load ground truth data to get categories
    try:
        gt_df = pd.read_csv('balanced_dataset/ground_truth_labels.csv')
        # Merge with results
        merged_df = successful_df.merge(gt_df[['dataset_id', 'category']], left_on='image_id', right_on='dataset_id', how='left')
        
        for category in merged_df['category'].unique():
            if pd.isna(category):
                continue
            cat_df = merged_df[merged_df['category'] == category]
            cat_correct = (cat_df['ground_truth_label'] == cat_df['normalized_prediction']).sum()
            cat_total = len(cat_df)
            cat_accuracy = cat_correct / cat_total if cat_total > 0 else 0
            
            print(f"{category:<25}: {cat_accuracy:.3f} ({cat_correct}/{cat_total})")
            
    except Exception as e:
        print(f"Could not load category analysis: {e}")
    
    # Acne subtype analysis
    print(f"\nACNE SUBTYPE ANALYSIS")
    print("-" * 30)
    
    acne_df = successful_df[successful_df['ground_truth_label'] == 'acne']
    if len(acne_df) > 0:
        subtype_confusion = pd.crosstab(
            acne_df['ground_truth_subtype'],
            acne_df['predicted_subtype'],
            margins=True
        )
        print(subtype_confusion)
        
        # Calculate accuracy for each acne subtype
        print(f"\nAccuracy by Acne Subtype:")
        for subtype in acne_df['ground_truth_subtype'].unique():
            subtype_df = acne_df[acne_df['ground_truth_subtype'] == subtype]
            subtype_correct = (subtype_df['ground_truth_subtype'] == subtype_df['predicted_subtype']).sum()
            subtype_total = len(subtype_df)
            subtype_accuracy = subtype_correct / subtype_total if subtype_total > 0 else 0
            print(f"  {subtype:<20}: {subtype_accuracy:.3f} ({subtype_correct}/{subtype_total})")
    
    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision_acne': precision_acne,
        'precision_non_acne': precision_non_acne,
        'f1_acne': f1_acne,
        'f1_non_acne': f1_non_acne,
        'confusion_matrix': cm,
        'classes': classes
    }

def main():
    """Main function to run detailed metrics analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate detailed metrics for Claude classification results')
    parser.add_argument('--results', default='claude_classification_results.json',
                       help='Path to results JSON file')
    
    args = parser.parse_args()
    
    try:
        results = load_results(args.results)
        metrics = calculate_detailed_metrics(results)
        
        print(f"\nSUMMARY")
        print("=" * 60)
        print(f"Overall Accuracy: {metrics['accuracy']:.1%}")
        print(f"Acne Detection (Sensitivity): {metrics['sensitivity']:.1%}")
        print(f"Non-Acne Detection (Specificity): {metrics['specificity']:.1%}")
        print(f"Acne Precision: {metrics['precision_acne']:.1%}")
        print(f"Non-Acne Precision: {metrics['precision_non_acne']:.1%}")
        print(f"Macro F1-Score: {metrics['f1_macro']:.3f}")
        
    except FileNotFoundError:
        print(f"Error: Results file not found at {args.results}")
    except Exception as e:
        print(f"Error analyzing results: {e}")

if __name__ == "__main__":
    main()
