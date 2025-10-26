#!/usr/bin/env python3
"""
Non-interactive script to run Claude skin classification on the full balanced dataset.
This script processes all 300 images without user prompts.
"""

import os
import sys
from claude_skin_classifier import ClaudeSkinClassifier
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def main():
    """Run Claude classification on the full balanced dataset."""
    
    # Check if API key is provided
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("Error: Please set the ANTHROPIC_API_KEY environment variable")
        print("You can get your API key from: https://console.anthropic.com/")
        print("\nExample:")
        print("export ANTHROPIC_API_KEY='your-api-key-here'")
        return False
    
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
        return False
    
    print(f"Ground truth file: {ground_truth_csv}")
    print(f"Output file: {output_file}")
    
    print("\nStarting full classification of 300 images...")
    print("This will take approximately 25-40 minutes.")
    print("The script will save intermediate results every 10 images.")
    print("=" * 60)
    
    # Process the dataset
    try:
        classifier.process_dataset(
            ground_truth_csv=ground_truth_csv,
            output_file=output_file,
            max_images=None  # Process all images
        )
        print(f"\n✅ Classification complete! Results saved to {output_file}")
        
        # Run analysis
        print("\nRunning analysis...")
        os.system(f"python src/analyze_claude_results.py --results {output_file}")
        
        return True
        
    except KeyboardInterrupt:
        print("\n⚠️  Classification interrupted by user")
        print(f"Partial results saved to {output_file}")
        return False
        
    except Exception as e:
        print(f"\n❌ Error during classification: {e}")
        print(f"Partial results saved to {output_file}")
        return False

if __name__ == "__main__":
    main()
