#!/usr/bin/env python3
"""
Script to run both analysis scripts and save results to CSV and Markdown files.
"""

import json
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import subprocess
import sys
from datetime import datetime
import os

def load_results(json_file):
    """Load classification results from JSON file."""
    with open(json_file, 'r') as f:
        return json.load(f)

def calculate_metrics(results):
    """Calculate all metrics and return as dictionary."""
    df = pd.DataFrame(results)
    successful_df = df[df['success'] == True]
    
    if len(successful_df) == 0:
        return None
    
    # Prepare data for sklearn metrics
    y_true = successful_df['ground_truth_label'].values
    y_pred = successful_df['normalized_prediction'].values
    
    # Calculate metrics
    accuracy = (y_true == y_pred).mean()
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
    recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    # Confusion matrix
    classes = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    # Acne vs non-acne specific metrics
    tp = ((y_true == 'acne') & (y_pred == 'acne')).sum()
    fp = ((y_true == 'non_acne') & (y_pred == 'acne')).sum()
    tn = ((y_true == 'non_acne') & (y_pred == 'non_acne')).sum()
    fn = ((y_true == 'acne') & (y_pred == 'non_acne')).sum()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision_acne = tp / (tp + fp) if (tp + fp) > 0 else 0
    precision_non_acne = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1_acne = 2 * (precision_acne * sensitivity) / (precision_acne + sensitivity) if (precision_acne + sensitivity) > 0 else 0
    f1_non_acne = 2 * (precision_non_acne * specificity) / (precision_non_acne + specificity) if (precision_non_acne + specificity) > 0 else 0
    
    return {
        'total_images': len(df),
        'successful': len(successful_df),
        'failed': len(df[df['success'] == False]),
        'success_rate': len(successful_df) / len(df),
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision_acne': precision_acne,
        'precision_non_acne': precision_non_acne,
        'f1_acne': f1_acne,
        'f1_non_acne': f1_non_acne,
        'confusion_matrix': cm,
        'classes': classes,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    }

def save_metrics_to_csv(metrics, output_file):
    """Save metrics to CSV file."""
    # Create summary metrics DataFrame
    summary_data = {
        'Metric': [
            'Total Images',
            'Successful Classifications',
            'Failed Classifications',
            'Success Rate (%)',
            'Overall Accuracy (%)',
            'Macro Precision',
            'Macro Recall',
            'Macro F1-Score',
            'Micro Precision',
            'Micro Recall',
            'Micro F1-Score',
            'Sensitivity (Acne Recall) (%)',
            'Specificity (Non-Acne Recall) (%)',
            'Acne Precision (%)',
            'Non-Acne Precision (%)',
            'Acne F1-Score',
            'Non-Acne F1-Score',
            'True Positives',
            'False Positives',
            'True Negatives',
            'False Negatives'
        ],
        'Value': [
            metrics['total_images'],
            metrics['successful'],
            metrics['failed'],
            round(metrics['success_rate'] * 100, 1),
            round(metrics['accuracy'] * 100, 1),
            round(metrics['precision_macro'], 3),
            round(metrics['recall_macro'], 3),
            round(metrics['f1_macro'], 3),
            round(metrics['precision_micro'], 3),
            round(metrics['recall_micro'], 3),
            round(metrics['f1_micro'], 3),
            round(metrics['sensitivity'] * 100, 1),
            round(metrics['specificity'] * 100, 1),
            round(metrics['precision_acne'] * 100, 1),
            round(metrics['precision_non_acne'] * 100, 1),
            round(metrics['f1_acne'], 3),
            round(metrics['f1_non_acne'], 3),
            int(metrics['tp']),
            int(metrics['fp']),
            int(metrics['tn']),
            int(metrics['fn'])
        ]
    }
    
    df = pd.DataFrame(summary_data)
    df.to_csv(output_file, index=False)
    print(f"Metrics saved to: {output_file}")

def save_confusion_matrix_to_csv(metrics, output_file):
    """Save confusion matrix to CSV file."""
    cm = metrics['confusion_matrix']
    classes = metrics['classes']
    
    # Create confusion matrix DataFrame
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_df.index.name = 'Ground Truth'
    cm_df.columns.name = 'Predicted'
    
    cm_df.to_csv(output_file)
    print(f"Confusion matrix saved to: {output_file}")

def save_detailed_results_to_csv(results, output_file):
    """Save detailed results to CSV file."""
    df = pd.DataFrame(results)
    
    # Select relevant columns for analysis
    analysis_columns = [
        'image_id', 'image_filename', 'ground_truth_label', 'ground_truth_subtype',
        'normalized_prediction', 'predicted_subtype', 'is_correct', 'success',
        'processing_time_ms'
    ]
    
    # Add confidence and reasoning if available
    if 'api_response' in df.columns:
        df['confidence'] = df['api_response'].apply(lambda x: x.get('confidence', 'N/A') if x else 'N/A')
        df['reasoning'] = df['api_response'].apply(lambda x: x.get('reasoning', 'N/A') if x else 'N/A')
        analysis_columns.extend(['confidence', 'reasoning'])
    
    # Add full raw response if available
    if 'full_raw_response' in df.columns:
        analysis_columns.append('full_raw_response')
    
    analysis_df = df[analysis_columns]
    analysis_df.to_csv(output_file, index=False)
    print(f"Detailed results saved to: {output_file}")

def save_analysis_to_markdown(metrics, results, output_file, model_name="Claude"):
    """Save comprehensive analysis to Markdown file."""
    
    # Format model name for display
    display_name = model_name.replace('_', ' ').title()
    if 'claude' in model_name.lower():
        display_name = "Claude"
    elif 'gpt' in model_name.lower():
        display_name = "GPT"
    elif 'medgemma' in model_name.lower():
        display_name = "MedGemma (Ollama)"
    
    with open(output_file, 'w') as f:
        f.write(f"# {display_name} Skin Classification Analysis Report\n\n")
        f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write(f"- **Total Images Processed:** {metrics['total_images']}\n")
        f.write(f"- **Success Rate:** {metrics['success_rate']*100:.1f}%\n")
        f.write(f"- **Overall Accuracy:** {metrics['accuracy']*100:.1f}%\n")
        f.write(f"- **Acne Detection (Sensitivity):** {metrics['sensitivity']*100:.1f}%\n")
        f.write(f"- **Non-Acne Detection (Specificity):** {metrics['specificity']*100:.1f}%\n\n")
        
        # Key Metrics
        f.write("## Key Metrics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Overall Accuracy | {metrics['accuracy']*100:.1f}% |\n")
        f.write(f"| Macro Precision | {metrics['precision_macro']:.3f} |\n")
        f.write(f"| Macro Recall | {metrics['recall_macro']:.3f} |\n")
        f.write(f"| Macro F1-Score | {metrics['f1_macro']:.3f} |\n")
        f.write(f"| Acne Precision | {metrics['precision_acne']*100:.1f}% |\n")
        f.write(f"| Acne Recall (Sensitivity) | {metrics['sensitivity']*100:.1f}% |\n")
        f.write(f"| Non-Acne Precision | {metrics['precision_non_acne']*100:.1f}% |\n")
        f.write(f"| Non-Acne Recall (Specificity) | {metrics['specificity']*100:.1f}% |\n\n")
        
        # Confusion Matrix
        f.write("## Confusion Matrix\n\n")
        f.write("| | Predicted Acne | Predicted Non-Acne | Total |\n")
        f.write("|---|---|---|---|\n")
        f.write(f"| **Actual Acne** | {metrics['tp']} | {metrics['fn']} | {metrics['tp'] + metrics['fn']} |\n")
        f.write(f"| **Actual Non-Acne** | {metrics['fp']} | {metrics['tn']} | {metrics['fp'] + metrics['tn']} |\n")
        f.write(f"| **Total** | {metrics['tp'] + metrics['fp']} | {metrics['fn'] + metrics['tn']} | {metrics['total_images']} |\n\n")
        
        # Analysis by Category
        f.write("## Analysis by Category\n\n")
        df = pd.DataFrame(results)
        successful_df = df[df['success'] == True]
        
        # Load ground truth data for category analysis
        try:
            gt_df = pd.read_csv('balanced_dataset/ground_truth_labels.csv')
            merged_df = successful_df.merge(gt_df[['dataset_id', 'category']], left_on='image_id', right_on='dataset_id', how='left')
            
            f.write("| Category | Accuracy | Correct/Total |\n")
            f.write("|----------|----------|---------------|\n")
            
            for category in merged_df['category'].unique():
                if pd.isna(category):
                    continue
                cat_df = merged_df[merged_df['category'] == category]
                cat_correct = (cat_df['ground_truth_label'] == cat_df['normalized_prediction']).sum()
                cat_total = len(cat_df)
                cat_accuracy = cat_correct / cat_total if cat_total > 0 else 0
                f.write(f"| {category} | {cat_accuracy:.1%} | {cat_correct}/{cat_total} |\n")
        except:
            f.write("Category analysis not available.\n")
        
        f.write("\n")
        
        # Key Findings
        f.write("## Key Findings\n\n")
        f.write("### Strengths\n")
        f.write(f"- **High Non-Acne Detection:** {metrics['specificity']*100:.1f}% specificity\n")
        f.write(f"- **High Acne Precision:** {metrics['precision_acne']*100:.1f}% precision when predicting acne\n")
        f.write("- **Conservative Approach:** Low false positive rate for acne\n\n")
        
        f.write("### Areas for Improvement\n")
        f.write(f"- **Low Acne Detection:** Only {metrics['sensitivity']*100:.1f}% of actual acne cases detected\n")
        f.write(f"- **High False Negatives:** {metrics['fn']} acne cases missed\n")
        f.write("- **Subtype Classification:** Poor performance on specific acne subtypes\n\n")
        
        # Recommendations
        f.write("## Recommendations\n\n")
        f.write("1. **Improve Acne Detection:** Focus on reducing false negatives\n")
        f.write("2. **Better Subtype Recognition:** Train on more diverse acne subtypes\n")
        f.write("3. **Acne-like Conditions:** Improve distinction between acne and rosacea/perioral dermatitis\n")
        f.write("4. **Clinical Validation:** Review misclassified cases with dermatologists\n\n")
        
        # Technical Details
        f.write("## Technical Details\n\n")
        
        # Calculate average processing time
        df = pd.DataFrame(results)
        successful_df = df[df['success'] == True]
        if len(successful_df) > 0 and 'processing_time_ms' in successful_df.columns:
            avg_time = successful_df['processing_time_ms'].mean() / 1000
            f.write(f"- **Average Processing Time:** {avg_time:.2f} seconds per image\n")
        
        # Get model name from results if available
        if 'model' in df.columns and len(df[df['model'].notna()]) > 0:
            model_info = df[df['model'].notna()]['model'].iloc[0]
            f.write(f"- **Model:** {model_info}\n")
        
        # Calculate success rate
        success_rate = metrics['success_rate'] * 100
        f.write(f"- **API Success Rate:** {success_rate:.1f}%\n")
        
        # Confidence if available
        if 'api_response' in df.columns:
            successful_with_response = successful_df[successful_df['api_response'].notna()]
            if len(successful_with_response) > 0:
                confidences = successful_with_response['api_response'].apply(
                    lambda x: x.get('confidence', '') if isinstance(x, dict) else ''
                )
                high_conf = (confidences == 'high').sum()
                if len(confidences) > 0:
                    high_conf_pct = (high_conf / len(confidences)) * 100
                    f.write(f"- **High Confidence Predictions:** {high_conf_pct:.1f}%\n")
        
        f.write("\n")
    
    print(f"Analysis report saved to: {output_file}")

def get_model_name_from_file(results_file):
    """Extract model name from results file path."""
    filename = os.path.basename(results_file)
    # Remove .json extension and common suffixes
    model_name = filename.replace('.json', '').replace('_classification_results', '').replace('_results', '')
    
    # Map common names
    if 'claude' in model_name.lower():
        return 'claude'
    elif 'gpt' in model_name.lower():
        return 'gpt'
    elif 'medgemma' in model_name.lower() or 'ollama' in model_name.lower():
        return 'medgemma_ollama'
    else:
        return model_name

def main():
    """Main function to run analysis and save results."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze classification results and save metrics.')
    parser.add_argument('results_file', nargs='?', default='claude_classification_results.json',
                       help='Path to classification results JSON file (default: claude_classification_results.json)')
    parser.add_argument('--output-prefix', default=None,
                       help='Prefix for output files (default: auto-detect from input filename)')
    
    args = parser.parse_args()
    
    results_file = args.results_file
    
    if not os.path.exists(results_file):
        print(f"Error: Results file not found at {results_file}")
        return
    
    # Determine model name and output prefix
    if args.output_prefix:
        model_prefix = args.output_prefix
    else:
        model_name = get_model_name_from_file(results_file)
        model_prefix = f"{model_name}_final"
    
    print(f"Analyzing results from: {results_file}")
    print(f"Using output prefix: {model_prefix}")
    print("Loading results and calculating metrics...")
    
    results = load_results(results_file)
    metrics = calculate_metrics(results)
    
    if metrics is None:
        print("No successful classifications to analyze.")
        return
    
    # Create output directory
    os.makedirs("analysis_results", exist_ok=True)
    
    # Save different types of results
    print("\nSaving analysis results...")
    
    # 1. Summary metrics CSV
    save_metrics_to_csv(metrics, f"analysis_results/{model_prefix}_summary_metrics.csv")
    
    # 2. Confusion matrix CSV
    save_confusion_matrix_to_csv(metrics, f"analysis_results/{model_prefix}_confusion_matrix.csv")
    
    # 3. Detailed results CSV
    save_detailed_results_to_csv(results, f"analysis_results/{model_prefix}_detailed_results.csv")
    
    # 4. Comprehensive markdown report
    save_analysis_to_markdown(metrics, results, f"analysis_results/{model_prefix}_analysis_report.md", model_prefix)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"Files saved in 'analysis_results/' directory with prefix '{model_prefix}':")
    print(f"- {model_prefix}_summary_metrics.csv: Key performance metrics")
    print(f"- {model_prefix}_confusion_matrix.csv: Confusion matrix data")
    print(f"- {model_prefix}_detailed_results.csv: Individual image results")
    print(f"- {model_prefix}_analysis_report.md: Comprehensive analysis report")
    print("\nYou can now view these files in any spreadsheet or markdown viewer!")

if __name__ == "__main__":
    main()
