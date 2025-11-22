#!/usr/bin/env python3
"""
Example script to run GPT skin classification on the balanced dataset.
This script demonstrates how to use the GPTSkinClassifier.
Matches the Claude pipeline exactly.
"""

import os
import sys
from gpt_skin_classifier import GPTSkinClassifier
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def main():
    """Run GPT classification on the balanced dataset."""
    
    # Check if API key is provided
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: Please set the OPENAI_API_KEY environment variable")
        print("You can get your API key from: https://platform.openai.com/api-keys")
        print("\nExample:")
        print("Create a .env file with:")
        print("OPENAI_API_KEY=your-api-key-here")
        print("python src/run_gpt_classification.py")
        return
    
    # Initialize classifier
    print("Initializing GPT skin classifier...")
    classifier = GPTSkinClassifier(api_key=api_key)
    
    # Paths
    ground_truth_csv = "balanced_dataset/ground_truth_labels.csv"
    output_file = "gpt_results_5.1.json"
    
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
    # Default to full run (all 300 images)
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        max_images = 10
        print("Running in test mode: processing 10 images")
    else:
        max_images = None
        print("Running full classification: processing all 300 images (use --test for 10 images)")
    
    print("Starting classification...")
    
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
