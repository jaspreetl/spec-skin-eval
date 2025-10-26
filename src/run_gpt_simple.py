#!/usr/bin/env python3
"""
Simple GPT skin classification script that works with metadata.csv
Similar to run_llm.py but with better error handling and rate limiting.
"""

import os
import base64
import pandas as pd
import random
import time
import json
from datetime import datetime, timezone
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize client
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    print("Error: Please set OPENAI_API_KEY in your .env file")
    exit(1)

client = OpenAI(api_key=api_key)

# 20 normalized conditions (same as original run_llm.py)
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

def classify_image(image_path: str, max_retries: int = 3) -> dict:
    """Send image to GPT-4o-mini with classification prompt."""
    start_time = time.time()
    
    try:
        base64_img = encode_image(image_path)
        if not base64_img:
            return {
                "prediction": "ERROR",
                "error": "Failed to encode image",
                "processing_time_ms": 0,
                "success": False
            }

        prompt = f"""
        You are an expert dermatologist. 
        Classify the following skin image into exactly one of these 20 conditions:
        {", ".join(CONDITIONS)}.
        
        Respond with only the condition name from the list above.
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
                    max_tokens=50,
                    temperature=0.1,
                )
                
                processing_time = (time.time() - start_time) * 1000
                prediction = response.choices[0].message.content.strip()
                
                # Estimate cost (rough calculation for GPT-4o-mini)
                estimated_tokens = 1000 + len(prediction.split()) * 1.3
                api_cost_usd = (estimated_tokens * 0.15) / 1000000
                
                return {
                    "prediction": prediction,
                    "error": None,
                    "processing_time_ms": round(processing_time, 2),
                    "api_cost_usd": round(api_cost_usd, 6),
                    "success": True
                }
                
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "quota" in error_msg.lower():
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 60  # 1min, 2min, 4min
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
                            "success": False
                        }
                else:
                    # For other errors, don't retry
                    raise e
        
        # If we get here, all retries failed
        processing_time = (time.time() - start_time) * 1000
        return {
            "prediction": "ERROR",
            "error": "All retries failed",
            "processing_time_ms": round(processing_time, 2),
            "api_cost_usd": 0.0,
            "success": False
        }
        
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        return {
            "prediction": "ERROR",
            "error": str(e),
            "processing_time_ms": round(processing_time, 2),
            "api_cost_usd": 0.0,
            "success": False
        }

def main():
    print("🔬 GPT Skin Classification Tool")
    print("=" * 50)
    
    # Load metadata
    print("📂 Loading metadata...")
    df = pd.read_csv("data/metadata.csv")
    print(f"Found {len(df)} total images")
    
    # Drop missing labels if any
    df_clean = df.dropna(subset=["condition"])
    print(f"Found {len(df_clean)} images with labels")
    
    # Ask user how many images to process
    print(f"\nAvailable conditions: {df_clean['condition'].nunique()}")
    print("Sample conditions:", df_clean['condition'].value_counts().head().to_dict())
    
    # Get sample size from user
    max_available = len(df_clean)
    print(f"\nHow many images would you like to process? (max: {max_available})")
    print("Recommended: 20 for testing, 100 for evaluation")
    
    try:
        sample_size = int(input("Enter number of images: "))
        sample_size = min(sample_size, max_available)
    except (ValueError, KeyboardInterrupt):
        print("Using default: 20 images")
        sample_size = 20
    
    # Estimate cost
    estimated_cost = sample_size * 0.002  # Rough estimate per image
    print(f"\n💰 Estimated cost: ~${estimated_cost:.3f}")
    
    confirm = input("Continue? (y/n): ").lower().strip()
    if confirm != 'y':
        print("Classification cancelled.")
        return
    
    # Sample images
    print(f"\n🎯 Sampling {sample_size} random images...")
    sample_df = df_clean.sample(n=sample_size, random_state=42)
    
    results = []
    successful = 0
    failed = 0
    total_cost = 0.0
    
    print(f"\n🚀 Starting classification...")
    print("=" * 50)
    
    for i, (_, row) in enumerate(sample_df.iterrows(), 1):
        true_label = row["condition"]
        img_path = row["path"]
        
        print(f"Processing {i}/{sample_size}: {os.path.basename(img_path)}")
        print(f"  Ground truth: {true_label}")
        
        # Check if image exists
        if not os.path.exists(img_path):
            print(f"  ❌ Image not found: {img_path}")
            result_data = {
                "prediction": "FILE_NOT_FOUND",
                "error": "Image file not found",
                "processing_time_ms": 0,
                "api_cost_usd": 0.0,
                "success": False
            }
            failed += 1
        else:
            # Classify the image
            result_data = classify_image(img_path)
            
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
            "true_label": true_label,
            "predicted_label": result_data["prediction"],
            "path": img_path,
            "processing_time_ms": result_data["processing_time_ms"],
            "api_cost_usd": result_data["api_cost_usd"],
            "success": result_data["success"],
            "error": result_data["error"],
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Add delay between requests
        if i < sample_size:  # Don't delay after last image
            time.sleep(1)
        
        print("-" * 30)
    
    # Save results
    print(f"\n💾 Saving results...")
    results_df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    
    # Save CSV (like original)
    results_df.to_csv("results/gpt_predictions.csv", index=False)
    
    # Save detailed JSON
    with open("results/gpt_detailed_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 CLASSIFICATION SUMMARY")
    print("=" * 50)
    print(f"Total images processed: {len(results)}")
    print(f"Successful classifications: {successful}")
    print(f"Failed classifications: {failed}")
    
    if len(results) > 0:
        success_rate = (successful / len(results)) * 100
        print(f"Success rate: {success_rate:.1f}%")
    
    if successful > 0:
        correct_predictions = sum(1 for r in results if r['predicted_label'] == r['true_label'] and r['success'])
        accuracy = (correct_predictions / successful) * 100
        print(f"Accuracy: {accuracy:.1f}% ({correct_predictions}/{successful})")
    
    print(f"Total API cost: ${total_cost:.6f}")
    print(f"\n📁 Results saved to:")
    print(f"  - results/gpt_predictions.csv (simple format)")
    print(f"  - results/gpt_detailed_results.json (detailed format)")

if __name__ == "__main__":
    main()
