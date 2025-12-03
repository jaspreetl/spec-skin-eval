#!/usr/bin/env python3
"""
Runner script for MedGemma classification using Ollama.
This is useful for local Mac runs where Ollama is easier to set up than Hugging Face.
"""

import os
import sys
import argparse
from medgemma_ollama_classifier import MedGemmaOllamaClassifier

def main():
    parser = argparse.ArgumentParser(description='Run MedGemma classification using Ollama on the balanced dataset.')
    parser.add_argument('--model', default='amsaravi/medgemma-4b-it:q8', 
                       help='Ollama model name (default: amsaravi/medgemma-4b-it:q8). Options: amsaravi/medgemma-4b-it:q8 (5GB), amsaravi/medgemma-4b-it:q6 (4GB)')
    parser.add_argument('--base-url', default='http://localhost:11434',
                       help='Ollama API base URL (default: http://localhost:11434)')
    parser.add_argument('--ground-truth', default='balanced_dataset/ground_truth_labels.csv',
                       help='Path to ground truth CSV file')
    parser.add_argument('--output', default='medgemma_ollama_classification_results.json',
                       help='Output JSON file path')
    parser.add_argument('--max-images', type=int, help='Maximum number of images to process (for testing)')
    parser.add_argument('--test', action='store_true', help='Run in test mode (10 images)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MedGemma Classification using Ollama")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Ollama URL: {args.base_url}")
    print(f"Ground truth: {args.ground_truth}")
    print(f"Output: {args.output}")
    print()
    
    # Check if Ollama model needs to be pulled
    print("⚠️  Make sure Ollama is running and the model is pulled:")
    print(f"   ollama pull {args.model}")
    print()
    
    # Determine max_images
    if args.test:
        max_images = 10
        print("Running in test mode: processing 10 images")
    elif args.max_images:
        max_images = args.max_images
        print(f"Processing {max_images} images")
    else:
        max_images = None
        print("Processing all 300 images")
    
    print()
    
    # Initialize classifier
    classifier = MedGemmaOllamaClassifier(
        model_name=args.model,
        base_url=args.base_url
    )
    
    # Process dataset
    classifier.process_dataset(
        ground_truth_csv=args.ground_truth,
        output_file=args.output,
        max_images=max_images
    )
    
    print("\n" + "=" * 60)
    print("✅ Classification complete!")
    print(f"Results saved to: {args.output}")
    print("=" * 60)

if __name__ == "__main__":
    main()

