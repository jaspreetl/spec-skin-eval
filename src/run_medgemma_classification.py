#!/usr/bin/env python3
"""
Example script to run MedGemma skin classification on the balanced dataset.
This script demonstrates how to use the MedGemmaSkinClassifier.
Matches the GPT and Claude pipeline exactly.
"""

import os
import sys
from medgemma_skin_classifier import MedGemmaSkinClassifier
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def main():
    """Run MedGemma classification on the balanced dataset."""
    
    # Paths
    ground_truth_csv = "balanced_dataset/ground_truth_labels.csv"
    output_file = "medgemma_classification_results.json"
    
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
    print("\nNote: MedGemma runs locally and requires:")
    print("  - GPU recommended for faster processing (CUDA/MPS)")
    print("  - ~8GB+ VRAM for the 4B model")
    print("  - MPS (Metal) will be used on Mac if available")
    print("  - transformers, torch, and pillow packages")
    
    # For testing, you can limit the number of images
    # Default to full run (all 300 images)
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        max_images = 2
        print("\nRunning in test mode: processing 10 images")
    else:
        max_images = None
        print("\nRunning full classification: processing all 300 images (use --test for 10 images)")
    
    # Model selection
    model_name = "google/medgemma-4b-it"
    if len(sys.argv) > 2:
        model_name = sys.argv[2]
    
    # Get HF token from environment
    hf_token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN')
    if not hf_token:
        print("⚠️  Warning: No HF_TOKEN found in .env file")
        print("   Some models may require authentication. Continuing without token...")
        print("   To add token: Add 'HF_TOKEN=your-token-here' to your .env file")
    else:
        print("✅ Found Hugging Face token")
    
    print(f"Using model: {model_name}")
    print("Starting classification...")
    
    # Process the dataset
    try:
        # Initialize classifier
        print("\nInitializing MedGemma skin classifier...")
        classifier = MedGemmaSkinClassifier(model_name=model_name, token=hf_token)
        
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
        import traceback
        traceback.print_exc()
        print(f"Partial results saved to {output_file}")

if __name__ == "__main__":
    main()

