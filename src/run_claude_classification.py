#!/usr/bin/env python3
"""
Example script to run Claude skin classification on the balanced dataset.
This script demonstrates how to use the ClaudeSkinClassifier.
"""

import os
import sys
from claude_skin_classifier import ClaudeSkinClassifier
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def main():
    """Run Claude classification on the balanced dataset."""
    
    # Check if API key is provided
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("Error: Please set the ANTHROPIC_API_KEY environment variable")
        print("You can get your API key from: https://console.anthropic.com/")
        print("\nExample:")
        print("export ANTHROPIC_API_KEY='your-api-key-here'")
        print("python src/run_claude_classification.py")
        return
    
    # Initialize classifier
    print("Initializing Claude skin classifier...")
    classifier = ClaudeSkinClassifier(api_key=api_key)
    
    # Paths
    ground_truth_csv = "balanced_dataset/ground_truth_labels.csv"
    output_file = "claude_classification_results.json"
    
    # Check if ground truth file exists
    if not os.path.exists(ground_truth_csv):
        print(f"Error: Ground truth file not found at {ground_truth_csv}")
        print("Please run the prepare_balanced_dataset.py script first.")
        return
    
    print(f"Ground truth file: {ground_truth_csv}")
    print(f"Output file: {output_file}")
    
    # Ask user for confirmation
    print("\nThis will process 300 images and may take a while.")
    print("The script will save intermediate results every 10 images.")
    
    # For testing, you can limit the number of images
    test_mode = input("\nDo you want to run in test mode (process only 10 images)? (y/n): ").lower().strip()
    max_images = 10 if test_mode == 'y' else None
    
    if max_images:
        print(f"Running in test mode: processing {max_images} images")
    else:
        print("Running full classification: processing all 300 images")
    
    confirm = input("Continue? (y/n): ").lower().strip()
    if confirm != 'y':
        print("Classification cancelled.")
        return
    
    # Process the dataset
    try:
        classifier.process_dataset(
            ground_truth_csv=ground_truth_csv,
            output_file=output_file,
            max_images=max_images
        )
        print(f"\n✅ Classification complete! Results saved to {output_file}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Classification interrupted by user")
        print(f"Partial results saved to {output_file}")
        
    except Exception as e:
        print(f"\n❌ Error during classification: {e}")
        print(f"Partial results saved to {output_file}")

if __name__ == "__main__":
    main()
