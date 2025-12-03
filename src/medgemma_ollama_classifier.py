#!/usr/bin/env python3
"""
Script to classify skin conditions using MedGemma via Ollama API.
"""

import os
import json
import time
import pandas as pd
from datetime import datetime, timezone
import re
from typing import Dict, List, Optional, Tuple
from PIL import Image
import base64
import requests
import io

class MedGemmaOllamaClassifier:
    def __init__(self, model_name: str = "amsaravi/medgemma-4b-it:q8", base_url: str = "http://localhost:11434"):
        """
        Initialize the MedGemma skin classifier using Ollama.
        
        Args:
            model_name: Ollama model name (default: amsaravi/medgemma-4b-it:q8)
                        Options: amsaravi/medgemma-4b-it:q8 (5GB), amsaravi/medgemma-4b-it:q6 (4GB)
            base_url: Ollama API base URL (default: http://localhost:11434)
        """
        self.model_name = model_name
        self.base_url = base_url
        self.results = []
        
        print(f"Using Ollama model: {model_name}")
        print(f"Ollama API URL: {base_url}")
        
        # Check if Ollama is running
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print("Ollama is running")
                # Check if model is available
                models = response.json().get('models', [])
                model_names = [m.get('name', '') for m in models]
                if model_name in model_names:
                    print(f"Model {model_name} is available")
                else:
                    print(f"Model {model_name} not found. Available models: {model_names}")
                    print(f"   Run: ollama pull {model_name}")
            else:
                print(f"Ollama API returned status {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Cannot connect to Ollama at {base_url}")
            print(f"   Make sure Ollama is installed and running:")
            print(f"   1. Install: https://ollama.com/download")
            print(f"   2. Start Ollama service")
            print(f"   3. Pull model: ollama pull {model_name}")
            raise
        
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
    
    def load_image(self, image_path: str) -> Optional[str]:
        """Load image and convert to base64 for Ollama API.
        
        Ollama expects just the raw base64 string, not a data URI.
        We convert images to JPEG format for consistency.
        """
        try:
            if not os.path.exists(image_path):
                print(f"Image file not found: {image_path}")
                return None
            
            # Load image with PIL and convert to RGB JPEG format
            # This ensures consistent encoding regardless of input format
            try:
                img = Image.open(image_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Save to bytes buffer as JPEG
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=95)
                image_data = buffer.getvalue()
                
                if len(image_data) == 0:
                    print(f"Image conversion resulted in empty data: {image_path}")
                    return None
                
            except Exception as e:
                # Fallback: try reading raw file
                print(f"PIL conversion failed, trying raw read: {e}")
                with open(image_path, 'rb') as f:
                    image_data = f.read()
                    if len(image_data) == 0:
                        print(f"Image file is empty: {image_path}")
                        return None
            
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            try:
                decoded = base64.b64decode(image_base64, validate=True)
                if len(decoded) == 0:
                    print(f"Base64 decode resulted in empty data: {image_path}")
                    return None
            except Exception as e:
                print(f"Invalid base64 encoding for {image_path}: {e}")
                return None
            
            return image_base64
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
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
        Classify a single image using Ollama API.
        
        Args:
            image_path: Path to the image file
            ground_truth_data: Dictionary containing ground truth information
            
        Returns:
            Dictionary with classification results
        """
        start_time = time.time()
        
        try:
            # Load and encode image
            image_base64 = self.load_image(image_path)
            if not image_base64:
                return self._create_error_result(ground_truth_data, "Failed to load image")
            
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

            api_url = f"{self.base_url}/api/chat"
            
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [image_base64]
                    }
                ],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 256
                }
            }
            
            response = requests.post(api_url, json=payload, timeout=120)
            
            if response.status_code != 200:
                raise Exception(f"Ollama API returned status {response.status_code}: {response.text}")
            
            result = response.json()
            # Extract response from chat format
            if 'message' in result:
                api_response_text = result['message'].get('content', '').strip()
            elif 'response' in result:
                api_response_text = result.get('response', '').strip()
            else:
                # Fallback: try to get any text content
                api_response_text = str(result).strip()
            
            processing_time = (time.time() - start_time) * 1000
            
            # Parse JSON response
            try:
                json_match = re.search(r'\{.*\}', api_response_text, re.DOTALL)
                if json_match:
                    api_response = json.loads(json_match.group())
                else:
                    api_response = {
                        "classification": api_response_text,
                        "confidence": "medium",
                        "reasoning": "Unable to parse structured response",
                        "key_features": []
                    }
            except json.JSONDecodeError:
                api_response = {
                    "classification": api_response_text,
                    "confidence": "medium", 
                    "reasoning": "Unable to parse JSON response",
                    "key_features": []
                }
            
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
            
            return {
                "image_id": int(ground_truth_data['dataset_id']),
                "image_filename": ground_truth_data['image_id'],
                "image_path": image_path,
                "ground_truth_label": ground_truth_label,
                "ground_truth_subtype": ground_truth_subtype,
                "model": self.model_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "api_response": api_response,
                "full_raw_response": api_response_text,
                "normalized_prediction": normalized_label,
                "predicted_subtype": predicted_subtype,
                "is_correct": bool(is_correct),
                "processing_time_ms": round(processing_time, 2),
                "api_cost_usd": 0.0,
                "success": True,
                "error": None
            }
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            return self._create_error_result(ground_truth_data, str(e), processing_time)
    
    def _create_error_result(self, ground_truth_data: Dict, error_message: str, processing_time: float = 0) -> Dict:
        """Create an error result dictionary."""
        ground_truth_label = 'acne' if ground_truth_data['category'] in ['True Acne', 'Acne-like/Confusable'] else 'non_acne'
        
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
            "model": self.model_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "api_response": None,
            "full_raw_response": None,
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
        Process the entire dataset.
        
        Args:
            ground_truth_csv: Path to ground truth CSV file
            output_file: Path to output JSON file
            max_images: Maximum number of images to process (None for all)
        """
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
        
        for idx, row in df.iterrows():
            print(f"Processing image {idx + 1}/{total_images}: {row['image_id']}")
            
            image_path = row['path']
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                result = self._create_error_result(row, "Image file not found")
                failed += 1
            else:
                result = self.classify_image(image_path, row)
                
                if result['success']:
                    print(f"  ✅ Classification: {result['normalized_prediction']} ({result['predicted_subtype']})")
                    print(f"     Ground truth: {result['ground_truth_label']} ({result['ground_truth_subtype']})")
                    print(f"     Correct: {result['is_correct']}")
                    print(f"     Time: {result['processing_time_ms']}ms")
                    successful += 1
                else:
                    print(f"Error: {result['error']}")
                    failed += 1
            
            self.results.append(result)
            
            # Save intermediate results every 10 images
            if (idx + 1) % 10 == 0:
                self._save_results(output_file)
                print(f"  💾 Saved intermediate results ({idx + 1} images processed)")
            
            print("-" * 40)
        
        self._save_results(output_file)
        
        print("CLASSIFICATION SUMMARY")
        print(f"Total images processed: {len(self.results)}")
        print(f"Successful classifications: {successful}")
        print(f"Failed classifications: {failed}")
        
        if successful > 0:
            correct_predictions = sum(1 for r in self.results if r['is_correct'])
            accuracy = (correct_predictions / successful) * 100
            print(f"Accuracy: {accuracy:.1f}% ({correct_predictions}/{successful})")
    
    def _save_results(self, output_file: str):
        """Save results to JSON file."""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)

