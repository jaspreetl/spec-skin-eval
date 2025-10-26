#!/usr/bin/env python3
"""
Test script to classify a single image using Claude API.
This is useful for testing the setup before running the full dataset.
"""

import os
import json
import sys
from claude_skin_classifier import ClaudeSkinClassifier
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def test_single_image():
    """Test classification on a single image."""
    
    # Check if API key is provided
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("Error: Please set the ANTHROPIC_API_KEY environment variable")
        print("You can get your API key from: https://console.anthropic.com/")
        print("\nExample:")
        print("export ANTHROPIC_API_KEY='your-api-key-here'")
        return False
    
    # Load ground truth data
    ground_truth_csv = "balanced_dataset/ground_truth_labels.csv"
    if not os.path.exists(ground_truth_csv):
        print(f"Error: Ground truth file not found at {ground_truth_csv}")
        print("Please run the prepare_balanced_dataset.py script first.")
        return False
    
    df = pd.read_csv(ground_truth_csv)
    print(f"Loaded {len(df)} images from ground truth file")
    
    # Select a test image (let's pick the first one)
    test_row = df.iloc[0]
    image_path = test_row['path']
    
    print(f"\nTesting with image: {test_row['image_id']}")
    print(f"Ground truth: {test_row['category']} - {test_row.get('acne_type', 'N/A')}")
    print(f"Image path: {image_path}")
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        print("This might be because the image paths in the CSV are from a different system.")
        print("Let's try to find an image in the current directory structure...")
        
        # Try to find an image in the data directory
        import glob
        possible_images = glob.glob("**/*.jpg", recursive=True) + glob.glob("**/*.jpeg", recursive=True) + glob.glob("**/*.png", recursive=True)
        
        if possible_images:
            print(f"Found {len(possible_images)} images in the directory:")
            for i, img in enumerate(possible_images[:5]):  # Show first 5
                print(f"  {i+1}. {img}")
            
            choice = input(f"\nSelect an image to test (1-{min(5, len(possible_images))}): ")
            try:
                selected_idx = int(choice) - 1
                if 0 <= selected_idx < min(5, len(possible_images)):
                    image_path = possible_images[selected_idx]
                    print(f"Selected: {image_path}")
                    
                    # Create a mock ground truth entry
                    test_row = {
                        'dataset_id': 0,
                        'image_id': os.path.basename(image_path),
                        'category': 'Test',
                        'acne_type': 'Test',
                        'path': image_path
                    }
                else:
                    print("Invalid selection")
                    return False
            except ValueError:
                print("Invalid input")
                return False
        else:
            print("No images found in the directory")
            return False
    
    # Initialize classifier
    print("\nInitializing Claude skin classifier...")
    try:
        classifier = ClaudeSkinClassifier(api_key=api_key)
        print("✅ Claude classifier initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing classifier: {e}")
        return False
    
    # Classify the image
    print(f"\nClassifying image: {os.path.basename(image_path)}")
    print("This may take 10-30 seconds...")
    
    try:
        result = classifier.classify_image(image_path, test_row)
        
        print("\n" + "="*60)
        print("CLASSIFICATION RESULT")
        print("="*60)
        
        if result['success']:
            print(f"✅ Classification successful!")
            print(f"Image: {result['image_filename']}")
            print(f"Ground truth: {result['ground_truth_label']} ({result['ground_truth_subtype']})")
            print(f"Predicted: {result['normalized_prediction']} ({result['predicted_subtype']})")
            print(f"Correct: {result['is_correct']}")
            print(f"Processing time: {result['processing_time_ms']}ms")
            
            if result['api_response']:
                print(f"\nClaude's response:")
                print(f"  Classification: {result['api_response'].get('classification', 'N/A')}")
                print(f"  Confidence: {result['api_response'].get('confidence', 'N/A')}")
                print(f"  Reasoning: {result['api_response'].get('reasoning', 'N/A')}")
                print(f"  Key features: {result['api_response'].get('key_features', [])}")
        else:
            print(f"❌ Classification failed: {result['error']}")
            return False
        
        # Save result to file
        output_file = "test_single_image_result.json"
        with open(output_file, 'w') as f:
            json.dump([result], f, indent=2)
        print(f"\nResult saved to: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during classification: {e}")
        return False

def main():
    """Main function."""
    print("Claude Skin Classification - Single Image Test")
    print("=" * 50)
    
    success = test_single_image()
    
    if success:
        print("\n✅ Test completed successfully!")
        print("You can now run the full classification with:")
        print("  python src/run_claude_classification.py")
    else:
        print("\n❌ Test failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
