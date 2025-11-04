#!/usr/bin/env python3
"""
Generate a comprehensive analysis report for GPT skin classification results.
Creates a detailed markdown report similar to the Claude analysis.
"""

import json
import pandas as pd
from collections import Counter
from datetime import datetime
import argparse
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support

def load_results(json_file):
    """Load GPT classification results from JSON file."""
    with open(json_file, 'r') as f:
        return json.load(f)

def calculate_metrics(results):
    """Calculate comprehensive classification metrics."""
    # Filter successful results
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        return None
    
    # Extract predictions and ground truth
    y_true = [r['ground_truth_label'] for r in successful_results]
    y_pred = [r['normalized_prediction'] for r in successful_results]
    
    # Calculate basic metrics
    correct_predictions = sum(1 for r in successful_results if r['is_correct'])
    total_successful = len(successful_results)
    accuracy = (correct_predictions / total_successful) * 100
    
    # Get unique labels
    labels = sorted(list(set(y_true + y_pred)))
    
    # Calculate precision, recall, f1 for each class
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    
    # Calculate macro averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    return {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'labels': labels,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'confusion_matrix': cm,
        'y_true': y_true,
        'y_pred': y_pred,
        'total_successful': total_successful,
        'correct_predictions': correct_predictions
    }

def analyze_by_category(results):
    """Analyze results by ground truth category."""
    successful_results = [r for r in results if r['success']]
    
    category_stats = {}
    for result in successful_results:
        category = result['ground_truth_label']
        if category not in category_stats:
            category_stats[category] = {'correct': 0, 'total': 0}
        
        category_stats[category]['total'] += 1
        if result['is_correct']:
            category_stats[category]['correct'] += 1
    
    # Calculate accuracy for each category
    for category in category_stats:
        stats = category_stats[category]
        stats['accuracy'] = (stats['correct'] / stats['total']) * 100
    
    return category_stats

def analyze_confidence(results):
    """Analyze confidence distribution."""
    successful_results = [r for r in results if r['success'] and r.get('api_response')]
    
    confidence_counts = Counter()
    confidence_accuracy = {}
    
    for result in successful_results:
        api_response = result.get('api_response', {})
        confidence = api_response.get('confidence', 'unknown')
        confidence_counts[confidence] += 1
        
        if confidence not in confidence_accuracy:
            confidence_accuracy[confidence] = {'correct': 0, 'total': 0}
        
        confidence_accuracy[confidence]['total'] += 1
        if result['is_correct']:
            confidence_accuracy[confidence]['correct'] += 1
    
    # Calculate accuracy by confidence
    for conf in confidence_accuracy:
        stats = confidence_accuracy[conf]
        stats['accuracy'] = (stats['correct'] / stats['total']) * 100
    
    return confidence_counts, confidence_accuracy

def analyze_processing_time(results):
    """Analyze processing time statistics."""
    successful_results = [r for r in results if r['success']]
    times = [r['processing_time_ms'] for r in successful_results if r.get('processing_time_ms')]
    
    if not times:
        return None
    
    return {
        'mean': np.mean(times),
        'median': np.median(times),
        'min': np.min(times),
        'max': np.max(times),
        'std': np.std(times)
    }

def analyze_costs(results):
    """Analyze API cost statistics."""
    successful_results = [r for r in results if r['success']]
    costs = [r['api_cost_usd'] for r in successful_results if r.get('api_cost_usd')]
    
    if not costs:
        return None
    
    total_cost = sum(costs)
    avg_cost = np.mean(costs)
    
    return {
        'total_cost': total_cost,
        'avg_cost': avg_cost,
        'cost_per_1000': avg_cost * 1000,
        'min_cost': np.min(costs),
        'max_cost': np.max(costs)
    }

def get_misclassifications(results, top_n=10):
    """Get most common misclassifications."""
    successful_results = [r for r in results if r['success']]
    incorrect_results = [r for r in successful_results if not r['is_correct']]
    
    misclass_counts = Counter()
    for result in incorrect_results:
        pair = (result['ground_truth_label'], result['normalized_prediction'])
        misclass_counts[pair] += 1
    
    return misclass_counts.most_common(top_n)

def generate_report(results, output_file):
    """Generate comprehensive analysis report."""
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    if not metrics:
        print("No successful results to analyze.")
        return
    
    category_stats = analyze_by_category(results)
    confidence_counts, confidence_accuracy = analyze_confidence(results)
    time_stats = analyze_processing_time(results)
    cost_stats = analyze_costs(results)
    misclassifications = get_misclassifications(results)
    
    # Generate report
    report = []
    
    # Header
    report.append("# GPT Skin Classification Analysis Report")
    report.append("")
    report.append(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Executive Summary
    report.append("## Executive Summary")
    report.append("")
    total_images = len(results)
    successful_images = len([r for r in results if r['success']])
    success_rate = (successful_images / total_images) * 100
    
    report.append(f"- **Total Images Processed:** {total_images}")
    report.append(f"- **Success Rate:** {success_rate:.1f}%")
    report.append(f"- **Overall Accuracy:** {metrics['accuracy']:.1f}%")
    report.append(f"- **Macro Precision:** {metrics['macro_precision']:.1f}%")
    report.append(f"- **Macro Recall:** {metrics['macro_recall']:.1f}%")
    report.append("")
    
    # Key Metrics
    report.append("## Key Metrics")
    report.append("")
    report.append("| Metric | Value |")
    report.append("|--------|-------|")
    report.append(f"| Overall Accuracy | {metrics['accuracy']:.1f}% |")
    report.append(f"| Macro Precision | {metrics['macro_precision']:.3f} |")
    report.append(f"| Macro Recall | {metrics['macro_recall']:.3f} |")
    report.append(f"| Macro F1-Score | {metrics['macro_f1']:.3f} |")
    report.append("")
    
    # Confusion Matrix (simplified for top conditions)
    report.append("## Confusion Matrix (Top 5 Conditions)")
    report.append("")
    
    # Get top 5 most common conditions
    condition_counts = Counter(metrics['y_true'])
    top_conditions = [cond for cond, _ in condition_counts.most_common(5)]
    
    # Create simplified confusion matrix
    report.append("| Ground Truth \\ Predicted | " + " | ".join(top_conditions) + " |")
    report.append("|" + "---|" * (len(top_conditions) + 1))
    
    for true_label in top_conditions:
        row = [f"**{true_label}**"]
        for pred_label in top_conditions:
            count = sum(1 for t, p in zip(metrics['y_true'], metrics['y_pred']) 
                       if t == true_label and p == pred_label)
            row.append(str(count))
        report.append("| " + " | ".join(row) + " |")
    report.append("")
    
    # Analysis by Category
    report.append("## Analysis by Category")
    report.append("")
    report.append("| Category | Accuracy | Correct/Total |")
    report.append("|----------|----------|---------------|")
    
    # Sort categories by accuracy (descending)
    sorted_categories = sorted(category_stats.items(), 
                             key=lambda x: x[1]['accuracy'], reverse=True)
    
    for category, stats in sorted_categories:
        report.append(f"| {category} | {stats['accuracy']:.1f}% | {stats['correct']}/{stats['total']} |")
    report.append("")
    
    # Most Common Misclassifications
    report.append("## Most Common Misclassifications")
    report.append("")
    report.append("| Ground Truth | Predicted | Count |")
    report.append("|--------------|-----------|-------|")
    
    for (gt, pred), count in misclassifications[:10]:
        report.append(f"| {gt} | {pred} | {count} |")
    report.append("")
    
    # Confidence Analysis
    if confidence_counts:
        report.append("## Confidence Analysis")
        report.append("")
        report.append("| Confidence Level | Count | Accuracy |")
        report.append("|------------------|-------|----------|")
        
        for conf in ['high', 'medium', 'low']:
            if conf in confidence_counts:
                count = confidence_counts[conf]
                acc = confidence_accuracy[conf]['accuracy']
                report.append(f"| {conf.title()} | {count} | {acc:.1f}% |")
        report.append("")
    
    # Performance Analysis
    report.append("## Key Findings")
    report.append("")
    
    # Identify strengths and weaknesses
    best_categories = sorted(category_stats.items(), 
                           key=lambda x: x[1]['accuracy'], reverse=True)[:3]
    worst_categories = sorted(category_stats.items(), 
                            key=lambda x: x[1]['accuracy'])[:3]
    
    report.append("### Strengths")
    for category, stats in best_categories:
        if stats['accuracy'] > 80:
            report.append(f"- **{category}:** {stats['accuracy']:.1f}% accuracy ({stats['correct']}/{stats['total']})")
    report.append("")
    
    report.append("### Areas for Improvement")
    for category, stats in worst_categories:
        if stats['accuracy'] < 50:
            report.append(f"- **{category}:** Only {stats['accuracy']:.1f}% accuracy ({stats['correct']}/{stats['total']})")
    report.append("")
    
    # Recommendations
    report.append("## Recommendations")
    report.append("")
    
    overall_acc = metrics['accuracy']
    if overall_acc < 70:
        report.append("1. **Improve Overall Accuracy:** Focus on better feature extraction and classification")
    if metrics['macro_precision'] < 0.7:
        report.append("2. **Reduce False Positives:** Improve precision across conditions")
    if metrics['macro_recall'] < 0.7:
        report.append("3. **Reduce False Negatives:** Improve recall for underperforming conditions")
    
    # Add specific recommendations for worst performing categories
    for category, stats in worst_categories[:2]:
        if stats['accuracy'] < 40:
            report.append(f"4. **{category} Detection:** Needs significant improvement ({stats['accuracy']:.1f}% accuracy)")
    
    report.append("")
    
    # Technical Details
    report.append("## Technical Details")
    report.append("")
    
    # Get model info from first successful result
    model_info = next((r['model'] for r in results if r['success']), 'gpt-4o-mini')
    report.append(f"- **Model:** {model_info}")
    
    if time_stats:
        report.append(f"- **Processing Time:** Average ~{time_stats['mean']/1000:.1f} seconds per image")
        report.append(f"- **Time Range:** {time_stats['min']/1000:.1f}s - {time_stats['max']/1000:.1f}s")
    
    if cost_stats:
        report.append(f"- **Total API Cost:** ${cost_stats['total_cost']:.6f}")
        report.append(f"- **Average Cost per Image:** ${cost_stats['avg_cost']:.6f}")
        report.append(f"- **Estimated Cost for 1000 images:** ${cost_stats['cost_per_1000']:.2f}")
    
    report.append(f"- **API Success Rate:** {success_rate:.1f}%")
    report.append("")
    
    # Save report
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"✅ Analysis report generated: {output_file}")
    return output_file

def main():
    """Main function to generate analysis report."""
    parser = argparse.ArgumentParser(description='Generate GPT classification analysis report')
    parser.add_argument('--results', default='results/gpt_classification_results.json',
                       help='Path to GPT results JSON file')
    parser.add_argument('--output', default='analysis_results/gpt_analysis_report.md',
                       help='Output markdown file path')
    
    args = parser.parse_args()
    
    try:
        if not os.path.exists(args.results):
            print(f"Error: Results file not found at {args.results}")
            print("Please run the GPT classification script first.")
            return
        
        print(f"Loading results from: {args.results}")
        results = load_results(args.results)
        
        print(f"Generating analysis report...")
        output_file = generate_report(results, args.output)
        
        print(f"\n📊 Report Summary:")
        successful = len([r for r in results if r['success']])
        total = len(results)
        print(f"  - Processed: {total} images")
        print(f"  - Successful: {successful}")
        print(f"  - Success Rate: {(successful/total)*100:.1f}%")
        
        if successful > 0:
            correct = sum(1 for r in results if r['success'] and r['is_correct'])
            accuracy = (correct / successful) * 100
            print(f"  - Accuracy: {accuracy:.1f}%")
        
        print(f"\n📁 Report saved to: {output_file}")
        
    except Exception as e:
        print(f"Error generating report: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
