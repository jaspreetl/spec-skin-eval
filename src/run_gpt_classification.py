#!/usr/bin/env python3
"""
GPT skin classification script that works with metadata.csv
Provides the same workflow as Claude but uses your existing dataset.
"""

import os
import sys
import json
import time
import pandas as pd
import base64
from datetime import datetime, timezone
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# 20 normalized conditions (same as your metadata.csv)
CONDITIONS = [
    "Acne",
    "Actinic Carcinoma",
    "Atopic Dermatitis",
    "Bullous Disease",
    "Cellulitis",
    "Eczema",
    "Drug Eruptions",
    "Herpes HPV",
    "Light Diseases",
    "Lupus",
    "Melanoma",
    "Poison Ivy",
    "Psoriasis",
    "Benign Tumors",
    "Systemic Disease",
    "Ringworm",
    "Urticarial Hives",
    "Vascular Tumors",
    "Vasculitis",
    "Viral Infections"
]

def encode_image(image_path: str) -> str:
    """Convert image to base64 string for API call."""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def classify_image(client, image_path: str, max_retries: int = 3) -> dict:
    """Classify a single image using GPT."""
    start_time = time.time()
    
    try:
        base64_img = encode_image(image_path)
        if not base64_img:
            return {
                "prediction": "ERROR",
                "error": "Failed to encode image",
                "processing_time_ms": 0,
                "api_cost_usd": 0.0,
                "success": False,
                "raw_response": None
            }

        prompt = f"""
You are an expert dermatologist. 
Classify the following skin image into exactly one of these 20 conditions:
{", ".join(CONDITIONS)}.

Provide your response in the following JSON format:
{{
    "classification": "the exact condition name from the list above",
    "confidence": "high/medium/low",
    "reasoning": "brief medical explanation for your classification (2-3 sentences)",
    "key_features": ["list", "of", "specific", "visual", "features", "observed"]
}}

Choose the most appropriate condition from the list. Be precise with the condition name.
        """

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a dermatologist AI."},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
                                }
                            ]
                        }
                    ],
                    max_tokens=500,
                    temperature=0.1,
                )
                
                processing_time = (time.time() - start_time) * 1000
                raw_response = response.choices[0].message.content.strip()
                
                # Try to parse JSON response
                try:
                    import re
                    json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
                    if json_match:
                        parsed_response = json.loads(json_match.group())
                        prediction = parsed_response.get('classification', raw_response)
                    else:
                        prediction = raw_response
                        parsed_response = {"classification": raw_response}
                except json.JSONDecodeError:
                    prediction = raw_response
                    parsed_response = {"classification": raw_response}
                
                # Estimate cost
                estimated_tokens = 1000 + len(raw_response.split()) * 1.3
                api_cost_usd = (estimated_tokens * 0.15) / 1000000
                
                return {
                    "prediction": prediction,
                    "error": None,
                    "processing_time_ms": round(processing_time, 2),
                    "api_cost_usd": round(api_cost_usd, 6),
                    "success": True,
                    "raw_response": raw_response,
                    "parsed_response": parsed_response
                }
                
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "quota" in error_msg.lower():
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 60
                        print(f"  ⚠️  Rate limit hit. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}")
                        time.sleep(wait_time)
                        continue
                    else:
                        processing_time = (time.time() - start_time) * 1000
                        return {
                            "prediction": "QUOTA_EXCEEDED",
                            "error": "API quota exceeded",
                            "processing_time_ms": round(processing_time, 2),
                            "api_cost_usd": 0.0,
                            "success": False,
                            "raw_response": None,
                            "parsed_response": None
                        }
                else:
                    raise e
        
        processing_time = (time.time() - start_time) * 1000
        return {
            "prediction": "ERROR",
            "error": "All retries failed",
            "processing_time_ms": round(processing_time, 2),
            "api_cost_usd": 0.0,
            "success": False,
            "raw_response": None,
            "parsed_response": None
        }
        
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        return {
            "prediction": "ERROR",
            "error": str(e),
            "processing_time_ms": round(processing_time, 2),
            "api_cost_usd": 0.0,
            "success": False,
            "raw_response": None,
            "parsed_response": None
        }

def main():
    """Run GPT classification on the metadata dataset."""
    
    # Check if API key is provided
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: Please set the OPENAI_API_KEY environment variable")
        print("You can get your API key from: https://platform.openai.com/api-keys")
        print("\nExample:")
        print("Create a .env file with:")
        print("OPENAI_API_KEY=your-api-key-here")
        return
    
    # Initialize client
    client = OpenAI(api_key=api_key)
    
    # Paths
    metadata_csv = "data/metadata.csv"
    output_file = "results/new_gpt_classification_results.json"
    csv_output = "results/new_gpt_predictions.csv"
    
    # Check if metadata file exists
    if not os.path.exists(metadata_csv):
        print(f"Error: Metadata file not found at {metadata_csv}")
        print("Please run the ingest.py script first to create the metadata.")
        return
    
    print("🔬 GPT Skin Classification")
    print("=" * 50)
    print(f"Metadata file: {metadata_csv}")
    print(f"JSON output: {output_file}")
    print(f"CSV output: {csv_output}")
    
    # Load and prepare data
    df = pd.read_csv(metadata_csv)
    df = df.dropna(subset=["condition"])
    
    print(f"\nFound {len(df)} images with labels")
    print(f"Available conditions: {df['condition'].nunique()}")
    
    # Ask user for sample size
    test_mode = input("\nDo you want to run in test mode (process only 20 images)? (y/n): ").lower().strip()
    
    if test_mode == 'y':
        max_images = 20
        print(f"Running in test mode: processing {max_images} images")
    else:
        try:
            max_images = int(input(f"How many images to process? (max: {len(df)}): ") or "100")
            max_images = min(max_images, len(df))
        except (ValueError, KeyboardInterrupt):
            max_images = 100
    
    estimated_cost = max_images * 0.002
    print(f"Estimated cost: ~${estimated_cost:.3f}")
    
    confirm = input("Continue? (y/n): ").lower().strip()
    if confirm != 'y':
        print("Classification cancelled.")
        return
    
    # Sample images
    sample_df = df.sample(n=max_images, random_state=42)
    
    results = []
    successful = 0
    failed = 0
    total_cost = 0.0
    
    print(f"\n🚀 Starting classification of {max_images} images...")
    print("=" * 50)
    
    for i, (_, row) in enumerate(sample_df.iterrows(), 1):
        true_label = row["condition"]
        img_path = row["path"]
        
        print(f"\nProcessing {i}/{max_images}: {os.path.basename(img_path)}")
        print(f"  Ground truth: {true_label}")
        
        # Check if image exists
        if not os.path.exists(img_path):
            print(f"  ❌ Image not found")
            result_data = {
                "prediction": "FILE_NOT_FOUND",
                "error": "Image file not found",
                "processing_time_ms": 0,
                "api_cost_usd": 0.0,
                "success": False,
                "raw_response": None,
                "parsed_response": None
            }
            failed += 1
        else:
            # Classify the image
            result_data = classify_image(client, img_path)
            
            if result_data["success"]:
                print(f"  ✅ Prediction: {result_data['prediction']}")
                print(f"     Correct: {result_data['prediction'] == true_label}")
                print(f"     Time: {result_data['processing_time_ms']}ms")
                print(f"     Cost: ${result_data['api_cost_usd']:.6f}")
                successful += 1
                total_cost += result_data['api_cost_usd']
            else:
                print(f"  ❌ Error: {result_data['error']}")
                failed += 1
                
                # Stop if quota exceeded
                if result_data['prediction'] == 'QUOTA_EXCEEDED':
                    print(f"  ⚠️  Stopping due to quota limits")
                    break
        
        # Store result
        results.append({
            "image_id": row["image_id"],
            "image_filename": os.path.basename(img_path),
            "image_path": img_path,
            "ground_truth_label": true_label,
            "predicted_label": result_data["prediction"],
            "is_correct": result_data["success"] and (result_data["prediction"] == true_label),
            "processing_time_ms": result_data["processing_time_ms"],
            "api_cost_usd": result_data["api_cost_usd"],
            "success": result_data["success"],
            "error": result_data["error"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "raw_response": result_data["raw_response"],
            "parsed_response": result_data["parsed_response"]
        })
        
        # Save intermediate results every 10 images
        if i % 10 == 0:
            os.makedirs("results", exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"  💾 Saved intermediate results ({i} images processed)")
        
        # Add delay between requests
        if i < max_images:
            time.sleep(2)
    
    # Save final results
    os.makedirs("results", exist_ok=True)
    
    # Save JSON
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save CSV (clean format)
    csv_results = []
    for r in results:
        csv_results.append({
            "image_id": r["image_id"],
            "true_label": r["ground_truth_label"],
            "predicted_label": r["predicted_label"],
            "path": r["image_path"],
            "success": r["success"],
            "is_correct": r["is_correct"],
            "processing_time_ms": r["processing_time_ms"],
            "api_cost_usd": r["api_cost_usd"],
            "error": r["error"]
        })
    
    csv_df = pd.DataFrame(csv_results)
    csv_df.to_csv(csv_output, index=False)
    
    # Print summary
    print(f"\n" + "=" * 50)
    print(f"📊 CLASSIFICATION SUMMARY")
    print("=" * 50)
    print(f"Total images processed: {len(results)}")
    print(f"Successful classifications: {successful}")
    print(f"Failed classifications: {failed}")
    
    if len(results) > 0:
        success_rate = (successful / len(results)) * 100
        print(f"Success rate: {success_rate:.1f}%")
    
    if successful > 0:
        correct_predictions = sum(1 for r in results if r['is_correct'])
        accuracy = (correct_predictions / successful) * 100
        print(f"Accuracy: {accuracy:.1f}% ({correct_predictions}/{successful})")
    
    print(f"Total API cost: ${total_cost:.6f}")
    print(f"\n📁 Results saved to:")
    print(f"  - {output_file} (detailed JSON)")
    print(f"  - {csv_output} (clean CSV)")
    print(f"\n📊 Analyze results:")
    print(f"  python src/analyze_gpt_results.py --results {output_file}")

if __name__ == "__main__":
    main()
