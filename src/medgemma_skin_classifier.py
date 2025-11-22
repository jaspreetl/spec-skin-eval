#!/usr/bin/env python3
"""
Script to classify skin conditions using Google MedGemma model on the balanced dataset.
Processes images and generates structured JSON output for evaluation.
"""

import os
import json
import time
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

class MedGemmaSkinClassifier:
    def __init__(self, model_name: str = "google/medgemma-4b-it", device: str = None, token: str = None):
        """
        Initialize the MedGemma skin classifier.
        
        Args:
            model_name: Hugging Face model name (default: google/medgemma-4b-it)
            device: Device to run on ('cuda', 'mps', 'cpu', or None for auto-detect)
            token: Hugging Face API token (optional, can also use HF_TOKEN env var)
        """
        self.model_name = model_name
        self.results = []
        
        # Get token from parameter or environment variable
        if token is None:
            token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN')
        
        self.token = token
        
        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        # Validate MPS availability if requested
        if self.device == "mps" and (not hasattr(torch.backends, 'mps') or not torch.backends.mps.is_available()):
            print("⚠️  MPS requested but not available, falling back to CPU")
            self.device = "cpu"
        
        print(f"Loading MedGemma model: {model_name}")
        print(f"Using device: {self.device}")
        if self.token:
            print("✅ Using Hugging Face token for authentication")
        else:
            print("⚠️  No HF token found - some models may require authentication")
        
        try:
            # Load processor and model with token
            self.processor = AutoProcessor.from_pretrained(model_name, token=self.token)
            
            # Determine dtype based on device
            if self.device == "cuda":
                dtype = torch.float16
                device_map = "auto"
            elif self.device == "mps":
                # MPS works best with float32, and device_map should be None
                dtype = torch.float32
                device_map = None
            else:
                dtype = torch.float32
                device_map = None
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=device_map,
                token=self.token
            )
            
            # Move model to device if not using device_map
            if device_map is None:
                self.model = self.model.to(self.device)
            
            self.model.eval()
            print("✅ Model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Make sure you have installed: pip install transformers torch pillow")
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
    
    def load_image(self, image_path: str) -> Image.Image:
        """Load and preprocess image."""
        try:
            image = Image.open(image_path).convert('RGB')
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
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
        Classify a single image using MedGemma model.
        
        Args:
            image_path: Path to the image file
            ground_truth_data: Dictionary containing ground truth information
            
        Returns:
            Dictionary with classification results
        """
        start_time = time.time()
        
        try:
            import traceback
            # Load image
            image = self.load_image(image_path)
            if not image:
                return self._create_error_result(ground_truth_data, "Failed to load image")
            
            # Prepare the prompt with image token placeholder
            # MedGemma requires <image> tokens in the prompt to indicate image positions
            prompt = """<image>
You are an expert dermatologist analyzing skin condition images. Your task is to determine whether the image shows acne or a non-acne condition.

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

            # Process image and text
            # Gemma3/MedGemma processor requires specific format
            # Remove <image> token from prompt - we'll handle it properly
            prompt_text = prompt.replace("<image>\n", "").strip()
            
            inputs = None
            
            # Try to get the correct image token from processor
            # Gemma3 processors have image_token_id attribute
            image_token_str = None
            if hasattr(self.processor, 'image_token_id'):
                # Get the string representation of the image token
                if hasattr(self.processor, 'tokenizer') and self.processor.tokenizer is not None:
                    try:
                        # Decode the image token ID to get the string
                        image_token_id = self.processor.image_token_id
                        image_token_str = self.processor.tokenizer.decode([image_token_id])
                    except:
                        pass
            
            # If we couldn't get it, try common formats
            if image_token_str is None:
                # Try common Gemma3 image token formats
                for token_candidate in ["<image>", "<|image|>", "<|image_0|>"]:
                    if hasattr(self.processor, 'tokenizer') and self.processor.tokenizer is not None:
                        try:
                            # Check if this token exists in vocabulary
                            token_id = self.processor.tokenizer.convert_tokens_to_ids([token_candidate])
                            if token_id and token_id[0] != self.processor.tokenizer.unk_token_id:
                                image_token_str = token_candidate
                                break
                        except:
                            continue
            
            # Default fallback
            if image_token_str is None:
                image_token_str = "<image>"
            
            # Create prompt with image token
            prompt_with_image = f"{image_token_str}\n{prompt_text}"
            
            try:
                # First try: with detected image token
                inputs = self.processor(text=prompt_with_image, images=image, return_tensors="pt")
            except Exception as e1:
                try:
                    # Second try: image as list
                    inputs = self.processor(text=prompt_with_image, images=[image], return_tensors="pt")
                except Exception as e2:
                    try:
                        # Third try: use apply_chat_template properly
                        if hasattr(self.processor, 'apply_chat_template'):
                            # Create messages in the format Gemma3 expects
                            messages = [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "image"},
                                        {"type": "text", "text": prompt_text}
                                    ]
                                }
                            ]
                            # Apply chat template - this should add the image token automatically
                            formatted_text = self.processor.apply_chat_template(
                                messages, 
                                tokenize=False, 
                                add_generation_prompt=True
                            )
                            # Now process with the formatted text (which should have image tokens)
                            inputs = self.processor(text=formatted_text, images=image, return_tensors="pt")
                        else:
                            raise e2
                    except Exception as e3:
                        # Last resort: check processor attributes for image handling
                        print(f"  ⚠️  All standard methods failed. Trying alternative approach...")
                        print(f"  ⚠️  Errors: {e1}, {e2}, {e3}")
                        # Try using the processor's internal image handling
                        raise Exception(f"Could not process image with any method. Last error: {e3}")
            
            # Validate inputs were created
            if inputs is None:
                raise Exception("Failed to process inputs - processor returned None")
            
            # Move inputs to device (cuda, mps, or cpu)
            if self.device in ["cuda", "mps"]:
                try:
                    inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                except Exception as e:
                    print(f"  ⚠️  Error moving inputs to device: {e}")
                    # Continue with CPU if device move fails
                    self.device = "cpu"
                    print(f"  ⚠️  Falling back to CPU")
            
            # Validate inputs before generation
            if inputs is None:
                raise Exception("Failed to process inputs - inputs is None")
            
            # Get pad_token_id safely
            pad_token_id = None
            if hasattr(self.processor, 'tokenizer') and self.processor.tokenizer is not None:
                if hasattr(self.processor.tokenizer, 'eos_token_id'):
                    pad_token_id = self.processor.tokenizer.eos_token_id
                elif hasattr(self.processor.tokenizer, 'pad_token_id'):
                    pad_token_id = self.processor.tokenizer.pad_token_id
            
            # Generate response
            outputs = None
            with torch.no_grad():
                try:
                    # Build generate kwargs carefully, filtering out None values
                    generate_kwargs = {}
                    
                    # Copy inputs, but check each value
                    for k, v in inputs.items():
                        if v is not None:
                            generate_kwargs[k] = v
                    
                    # Add generation parameters
                    generate_kwargs["max_new_tokens"] = 512
                    generate_kwargs["temperature"] = 0.1
                    generate_kwargs["do_sample"] = True
                    
                    if pad_token_id is not None:
                        generate_kwargs["pad_token_id"] = pad_token_id
                    
                    outputs = self.model.generate(**generate_kwargs)
                except Exception as e:
                    # Fallback if generation fails
                    print(f"  ⚠️  Generation error: {e}, trying with default parameters...")
                    try:
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=256,
                            do_sample=False
                        )
                    except Exception as e2:
                        print(f"  ❌ Generation failed with fallback parameters: {e2}")
                        raise Exception(f"Model generation failed: {e2}")
            
            # Check if outputs is valid
            if outputs is None:
                raise Exception("Model generation returned None")
            
            if len(outputs) == 0:
                raise Exception("Model generation returned empty output")
            
            # Decode response - handle outputs carefully
            generated_text = ""
            try:
                # Check outputs structure
                if outputs is None:
                    raise Exception("Outputs is None")
                
                # Get the first output - handle different output formats
                if isinstance(outputs, torch.Tensor):
                    output_tokens = outputs[0] if len(outputs.shape) > 1 else outputs
                elif isinstance(outputs, (list, tuple)):
                    output_tokens = outputs[0] if len(outputs) > 0 else None
                else:
                    output_tokens = outputs
                
                if output_tokens is None:
                    raise Exception("Output tokens is None")
                
                # Try to decode
                if hasattr(self.processor, 'decode'):
                    generated_text = self.processor.decode(output_tokens, skip_special_tokens=True)
                elif hasattr(self.processor, 'tokenizer') and self.processor.tokenizer is not None:
                    if hasattr(self.processor.tokenizer, 'decode'):
                        generated_text = self.processor.tokenizer.decode(output_tokens, skip_special_tokens=True)
                    else:
                        generated_text = str(output_tokens)
                else:
                    generated_text = str(output_tokens)
                    
            except Exception as e:
                print(f"  ⚠️  Decoding error: {e}, outputs type: {type(outputs)}")
                import traceback
                traceback.print_exc()
                # Last resort fallback
                try:
                    generated_text = str(outputs) if outputs is not None else ""
                except:
                    generated_text = "Error decoding response"
            
            # Extract only the generated part (remove prompt)
            if prompt in generated_text:
                api_response_text = generated_text.split(prompt)[-1].strip()
            else:
                api_response_text = generated_text.strip()
            
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
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
                "model": self.model_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "api_response": api_response,
                "full_raw_response": full_raw_response,
                "normalized_prediction": normalized_label,
                "predicted_subtype": predicted_subtype,
                "is_correct": bool(is_correct),
                "processing_time_ms": round(processing_time, 2),
                "api_cost_usd": 0.0,  # Local model, no API cost
                "success": True,
                "error": None
            }
            
            return result
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            import traceback
            error_details = f"{str(e)}\n{traceback.format_exc()}"
            print(f"  ❌ Full error traceback:\n{error_details}")
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
        print(f"Total images processed: {len(self.results)}")
        print(f"Successful classifications: {successful}")
        print(f"Failed classifications: {failed}")
        
        if len(self.results) > 0:
            print(f"Success rate: {(successful/len(self.results))*100:.1f}%")
        
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
    
    parser = argparse.ArgumentParser(description='Classify skin conditions using MedGemma model')
    parser.add_argument('--model', default='google/medgemma-4b-it', 
                       help='Hugging Face model name')
    parser.add_argument('--device', default=None, choices=['cuda', 'mps', 'cpu'],
                       help='Device to use (cuda/mps/cpu, default: auto-detect)')
    parser.add_argument('--token', default=None,
                       help='Hugging Face API token (or set HF_TOKEN env var)')
    parser.add_argument('--ground-truth', default='balanced_dataset/ground_truth_labels.csv', 
                       help='Path to ground truth CSV file')
    parser.add_argument('--output', default='medgemma_classification_results.json', 
                       help='Output JSON file path')
    parser.add_argument('--max-images', type=int, help='Maximum number of images to process (for testing)')
    
    args = parser.parse_args()
    
    # Initialize classifier
    classifier = MedGemmaSkinClassifier(
        model_name=args.model, 
        device=args.device,
        token=args.token
    )
    
    # Process dataset
    classifier.process_dataset(
        ground_truth_csv=args.ground_truth,
        output_file=args.output,
        max_images=args.max_images
    )

if __name__ == "__main__":
    main()

