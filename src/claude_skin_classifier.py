#!/usr/bin/env python3
"""
Script to classify skin conditions using Claude API on the balanced dataset.
Processes images and generates structured JSON output for evaluation.
"""

import os
import json
import time
import base64
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple
import anthropic
from PIL import Image
import io

class ClaudeSkinClassifier:
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-5-20250929"):
        """Initialize the Claude skin classifier."""
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.results = []
        
        # Classification mapping for normalization
        self.acne_subtypes = {
            'comedonal acne': 'comedonal',
            'papular acne': 'papular', 
            'pustular acne': 'pustular',
            'nodular acne': 'nodular',
            'cystic acne': 'cystic',
            'acne conglobata': 'conglobata'
        }
        
        self.non_acne_conditions = {
            'rosacea': 'rosacea',
            'eczema': 'eczema',
            'dermatitis': 'dermatitis',
            'not acne': 'other'
        }
    
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 for Claude API."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None
    
    def get_image_format(self, image_path: str) -> str:
        """Get the image format from file extension."""
        ext = Path(image_path).suffix.lower()
        if ext in ['.jpg', '.jpeg']:
            return 'jpeg'
        elif ext == '.png':
            return 'png'
        else:
            return 'jpeg'  # default
    
    def normalize_classification(self, classification: str) -> Tuple[str, str]:
        """
        Normalize the classification into acne/non_acne and subtype.
        
        Returns:
            Tuple of (normalized_label, subtype)
        """
        classification_lower = classification.lower()
        
        # Check for acne types
        for acne_type, subtype in self.acne_subtypes.items():
            if acne_type in classification_lower:
                return 'acne', subtype
        
        # Check for non-acne conditions
        for condition, subtype in self.non_acne_conditions.items():
            if condition in classification_lower:
                return 'non_acne', subtype
        
        # Default classification
        if 'not acne' in classification_lower or 'non-acne' in classification_lower:
            return 'non_acne', 'other'
        elif 'acne' in classification_lower:
            return 'acne', 'general'
        else:
            return 'non_acne', 'other'
    
    def classify_image(self, image_path: str, ground_truth_data: Dict) -> Dict:
        """
        Classify a single image using Claude API.
        
        Args:
            image_path: Path to the image file
            ground_truth_data: Dictionary containing ground truth information
            
        Returns:
            Dictionary with classification results
        """
        start_time = time.time()
        
        try:
            # Encode image
            image_base64 = self.encode_image(image_path)
            if not image_base64:
                return self._create_error_result(ground_truth_data, "Failed to encode image")
            
            # Prepare the prompt
            prompt = """You are an expert dermatologist analyzing skin condition images. Your task is to determine whether the image shows acne or a non-acne condition.

Analyze the image carefully and classify it into ONE of the following categories:

ACNE TYPES:
- Comedonal acne (blackheads and whiteheads)
- Papular acne (small red bumps without pus)
- Pustular acne (pus-filled pimples with white/yellow centers)
- Nodular acne (large, painful lumps beneath the skin)
- Cystic acne (deep, painful, pus-filled cysts)
- Acne conglobata (severe, interconnected lesions)

NON-ACNE:
- Not acne (if the condition is not acne, such as rosacea, eczema, dermatitis, or other skin conditions)

Provide your response in the following JSON format:
{
    "classification": "the category from the list above",
    "confidence": "high/medium/low",
    "reasoning": "brief medical explanation for your classification (2-3 sentences)",
    "key_features": ["list", "of", "specific", "visual", "features", "observed"]
}

Consider these diagnostic criteria:
- Comedones (blackheads/whiteheads) indicate comedonal acne
- Pustules with inflammation indicate pustular acne
- Deep nodules or cysts indicate severe acne
- Facial redness without comedones may indicate rosacea (not acne)
- Symmetrical rash patterns may indicate dermatitis (not acne)"""

            # Call Claude API
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=0.1,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": f"image/{self.get_image_format(image_path)}",
                                    "data": image_base64
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )
            
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Parse the response
            api_response_text = response.content[0].text
            
            # Try to extract JSON from the response
            try:
                # Look for JSON in the response
                json_match = re.search(r'\{.*\}', api_response_text, re.DOTALL)
                if json_match:
                    api_response = json.loads(json_match.group())
                else:
                    # Fallback: create a structured response
                    api_response = {
                        "classification": api_response_text.strip(),
                        "confidence": "medium",
                        "reasoning": "Unable to parse structured response",
                        "key_features": []
                    }
            except json.JSONDecodeError:
                api_response = {
                    "classification": api_response_text.strip(),
                    "confidence": "medium", 
                    "reasoning": "Unable to parse JSON response",
                    "key_features": []
                }
            
            # Store the full raw response
            full_raw_response = api_response_text
            
            # Normalize the classification
            normalized_label, predicted_subtype = self.normalize_classification(
                api_response.get('classification', '')
            )
            
            # Determine if prediction is correct
            ground_truth_label = 'acne' if ground_truth_data['category'] in ['True Acne', 'Acne-like/Confusable'] else 'non_acne'
            
            # Handle acne_type safely (might be NaN for non-acne images)
            acne_type = ground_truth_data.get('acne_type', 'other')
            if pd.isna(acne_type) or acne_type is None:
                ground_truth_subtype = 'other'
            else:
                ground_truth_subtype = str(acne_type).lower().replace(' ', '_')
            
            is_correct = (normalized_label == ground_truth_label)
            
            # Create result dictionary
            result = {
                "image_id": int(ground_truth_data['dataset_id']),
                "image_filename": ground_truth_data['image_id'],
                "image_path": image_path,
                "ground_truth_label": ground_truth_label,
                "ground_truth_subtype": ground_truth_subtype,
                "model": self.model,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "api_response": api_response,
                "full_raw_response": full_raw_response,  # Complete raw response from Claude
                "normalized_prediction": normalized_label,
                "predicted_subtype": predicted_subtype,
                "is_correct": bool(is_correct),
                "processing_time_ms": round(processing_time, 2),
                "api_cost_usd": 0.0,  # Claude pricing would need to be calculated
                "success": True,
                "error": None
            }
            
            return result
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            return self._create_error_result(ground_truth_data, str(e), processing_time)
    
    def _create_error_result(self, ground_truth_data: Dict, error_message: str, processing_time: float = 0) -> Dict:
        """Create an error result dictionary."""
        ground_truth_label = 'acne' if ground_truth_data['category'] in ['True Acne', 'Acne-like/Confusable'] else 'non_acne'
        
        # Handle acne_type safely (might be NaN for non-acne images)
        acne_type = ground_truth_data.get('acne_type', 'other')
        if pd.isna(acne_type) or acne_type is None:
            ground_truth_subtype = 'other'
        else:
            ground_truth_subtype = str(acne_type).lower().replace(' ', '_')
        
        return {
            "image_id": int(ground_truth_data['dataset_id']),
            "image_filename": ground_truth_data['image_id'],
            "image_path": ground_truth_data['path'],
            "ground_truth_label": ground_truth_label,
            "ground_truth_subtype": ground_truth_subtype,
            "model": self.model,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "api_response": None,
            "full_raw_response": None,  # No response due to error
            "normalized_prediction": None,
            "predicted_subtype": None,
            "is_correct": False,
            "processing_time_ms": round(processing_time, 2),
            "api_cost_usd": 0.0,
            "success": False,
            "error": error_message
        }
    
    def process_dataset(self, ground_truth_csv: str, output_file: str, max_images: Optional[int] = None):
        """
        Process the entire dataset and save results.
        
        Args:
            ground_truth_csv: Path to the ground truth CSV file
            output_file: Path to save the results JSON file
            max_images: Maximum number of images to process (for testing)
        """
        # Load ground truth data
        df = pd.read_csv(ground_truth_csv)
        
        if max_images:
            df = df.head(max_images)
            print(f"Processing {max_images} images for testing...")
        else:
            print(f"Processing all {len(df)} images...")
        
        total_images = len(df)
        successful = 0
        failed = 0
        
        print(f"Starting classification of {total_images} images...")
        print("=" * 60)
        
        for idx, row in df.iterrows():
            print(f"Processing image {idx + 1}/{total_images}: {row['image_id']}")
            
            # Check if image file exists
            image_path = row['path']
            if not os.path.exists(image_path):
                print(f"  ❌ Image not found: {image_path}")
                result = self._create_error_result(row, "Image file not found")
                failed += 1
            else:
                # Classify the image
                result = self.classify_image(image_path, row)
                
                if result['success']:
                    print(f"  ✅ Classification: {result['normalized_prediction']} ({result['predicted_subtype']})")
                    print(f"     Ground truth: {result['ground_truth_label']} ({result['ground_truth_subtype']})")
                    print(f"     Correct: {result['is_correct']}")
                    print(f"     Time: {result['processing_time_ms']}ms")
                    successful += 1
                else:
                    print(f"  ❌ Error: {result['error']}")
                    failed += 1
            
            self.results.append(result)
            
            # Save intermediate results every 10 images
            if (idx + 1) % 10 == 0:
                self._save_results(output_file)
                print(f"  💾 Saved intermediate results ({idx + 1} images processed)")
            
            print("-" * 40)
        
        # Save final results
        self._save_results(output_file)
        
        # Print summary
        print("\n" + "=" * 60)
        print("CLASSIFICATION SUMMARY")
        print("=" * 60)
        print(f"Total images processed: {total_images}")
        print(f"Successful classifications: {successful}")
        print(f"Failed classifications: {failed}")
        print(f"Success rate: {(successful/total_images)*100:.1f}%")
        
        if successful > 0:
            correct_predictions = sum(1 for r in self.results if r['is_correct'])
            accuracy = (correct_predictions / successful) * 100
            print(f"Accuracy: {accuracy:.1f}% ({correct_predictions}/{successful})")
        
        print(f"\nResults saved to: {output_file}")
    
    def _save_results(self, output_file: str):
        """Save results to JSON file."""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)

def main():
    """Main function to run the skin classification."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Classify skin conditions using Claude API')
    parser.add_argument('--api-key', required=True, help='Claude API key')
    parser.add_argument('--ground-truth', default='balanced_dataset/ground_truth_labels.csv', 
                       help='Path to ground truth CSV file')
    parser.add_argument('--output', default='claude_classification_results.json', 
                       help='Output JSON file path')
    parser.add_argument('--max-images', type=int, help='Maximum number of images to process (for testing)')
    parser.add_argument('--model', default='claude-sonnet-4-5-20250929', 
                       help='Claude model to use')
    
    args = parser.parse_args()
    
    # Initialize classifier
    classifier = ClaudeSkinClassifier(api_key=args.api_key, model=args.model)
    
    # Process dataset
    classifier.process_dataset(
        ground_truth_csv=args.ground_truth,
        output_file=args.output,
        max_images=args.max_images
    )

if __name__ == "__main__":
    main()
