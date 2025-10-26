import os
import base64
import pandas as pd
import random
import time
import json
from datetime import datetime, timezone
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize client
client = OpenAI()

# 20 normalized conditions
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
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def classify_image(image_path: str, ground_truth_data: dict, max_retries: int = 3) -> dict:
    """Send image to GPT-4o-mini with classification prompt."""
    start_time = time.time()
    
    try:
        base64_img = encode_image(image_path)
        if not base64_img:
            return create_error_result(ground_truth_data, "Failed to encode image")

        prompt = f"""
You are an expert dermatologist analyzing skin condition images.

Classify the following skin image into exactly ONE of these 20 conditions:
{", ".join(CONDITIONS)}

Provide your response in the following JSON format:
{{
    "classification": "the exact condition name from the list above",
    "confidence": "high/medium/low",
    "reasoning": "brief medical explanation for your classification (2-3 sentences)",
    "key_features": ["list", "of", "specific", "visual", "features", "observed"]
}}

Important:
- Choose the EXACT condition name from the list above
- Be specific and precise with the condition name
- Consider all visual features carefully
- Provide a clear medical reasoning for your choice
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
                
                # Clean up the response - remove markdown formatting
                import re
                import json
                
                try:
                    # Remove markdown code blocks if present
                    cleaned_response = re.sub(r'```json\s*|\s*```', '', raw_response)
                    
                    # Try to parse as JSON
                    json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
                    if json_match:
                        api_response = json.loads(json_match.group())
                    else:
                        # Fallback: create structured response
                        api_response = {
                            "classification": raw_response.strip(),
                            "confidence": "medium",
                            "reasoning": "Unable to parse structured response",
                            "key_features": []
                        }
                except json.JSONDecodeError:
                    api_response = {
                        "classification": raw_response.strip(),
                        "confidence": "medium",
                        "reasoning": "Unable to parse JSON response",
                        "key_features": []
                    }
                
                # Extract prediction
                predicted_label = api_response.get('classification', raw_response)
                
                # Determine if prediction is correct
                is_correct = (predicted_label == ground_truth_data['condition'])
                
                # Calculate approximate cost (GPT-4o-mini pricing)
                estimated_input_tokens = 1000  # Approximate for image + prompt
                estimated_output_tokens = len(raw_response.split()) * 1.3
                api_cost_usd = (estimated_input_tokens * 0.15 + estimated_output_tokens * 0.60) / 1000000
                
                # Create result dictionary matching Claude format
                result = {
                    "image_id": ground_truth_data.get('image_id', ''),
                    "image_filename": ground_truth_data.get('image_id', ''),
                    "image_path": image_path,
                    "ground_truth_label": ground_truth_data['condition'],
                    "ground_truth_subtype": "other",  # We don't have subtypes in metadata.csv
                    "model": "gpt-4o-mini",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "api_response": api_response,
                    "full_raw_response": raw_response,
                    "normalized_prediction": predicted_label,
                    "predicted_subtype": "other",
                    "is_correct": bool(is_correct),
                    "processing_time_ms": round(processing_time, 2),
                    "api_cost_usd": round(api_cost_usd, 6),
                    "success": True,
                    "error": None
                }
                
                return result
                
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "quota" in error_msg.lower():
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 60
                        print(f"Rate limit hit. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"Max retries reached. Quota exceeded.")
                        processing_time = (time.time() - start_time) * 1000
                        return create_error_result(ground_truth_data, "API quota exceeded", processing_time)
                else:
                    raise e
        
        # If we get here, all retries failed
        processing_time = (time.time() - start_time) * 1000
        return create_error_result(ground_truth_data, "All retries failed", processing_time)
        
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        return create_error_result(ground_truth_data, str(e), processing_time)

def create_error_result(ground_truth_data: dict, error_message: str, processing_time: float = 0) -> dict:
    """Create an error result dictionary matching Claude format."""
    from datetime import datetime, timezone
    
    return {
        "image_id": ground_truth_data.get('image_id', ''),
        "image_filename": ground_truth_data.get('image_id', ''),
        "image_path": ground_truth_data.get('path', ''),
        "ground_truth_label": ground_truth_data.get('condition', ''),
        "ground_truth_subtype": "other",
        "model": "gpt-4o-mini",
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

def main():
    print("🔬 GPT Skin Classification with Smart Batching")
    print("=" * 50)
    
    # Load metadata
    print("Loading metadata...")
    df = pd.read_csv("data/metadata.csv")
    print(f"Found {len(df)} total images")

    # Drop missing labels if any
    df = df.dropna(subset=["condition"])
    print(f"Found {len(df)} images with labels")

    # Ask for sample size
    print(f"\nHow many images to process? (default: 20)")
    try:
        sample_size = int(input("Enter number: ") or "20")
        sample_size = min(sample_size, len(df))
    except (ValueError, KeyboardInterrupt):
        sample_size = 20
    
    # Smart batching configuration
    batch_size = 5  # Process 5 images per batch
    batch_delay = 30  # Wait 30 seconds between batches
    
    print(f"\n🧠 Smart Batching Configuration:")
    print(f"  • Total images: {sample_size}")
    print(f"  • Batch size: {batch_size} images")
    print(f"  • Delay between batches: {batch_delay} seconds")
    print(f"  • Estimated total time: {((sample_size // batch_size) * batch_delay) / 60:.1f} minutes")
    
    confirm = input(f"\nContinue with batched processing? (y/n): ").lower().strip()
    if confirm != 'y':
        print("Processing cancelled.")
        return
    
    # Sample images
    sample_df = df.sample(n=sample_size, random_state=42)

    results = []
    successful = 0
    batch_count = 0
    
    print(f"\n🚀 Starting batched classification...")
    print("=" * 50)
    
    for i, (_, row) in enumerate(sample_df.iterrows(), 1):
        true_label = row["condition"]
        img_path = row["path"]
        
        # Check if we need to start a new batch
        if (i - 1) % batch_size == 0 and i > 1:
            batch_count += 1
            print(f"\n⏸️  Batch {batch_count} complete. Waiting {batch_delay}s to respect rate limits...")
            time.sleep(batch_delay)
            print(f"🔄 Starting batch {batch_count + 1}...")
        
        print(f"\n[{i}/{sample_size}] {os.path.basename(img_path)}")
        print(f"Ground truth: {true_label}")
        
        # Check if file exists
        if not os.path.exists(img_path):
            print(f"❌ File not found: {img_path}")
            result = create_error_result(row.to_dict(), "Image file not found")
            failed += 1
        else:
            try:
                # Pass row data to classify_image
                result = classify_image(img_path, row.to_dict())
                
                if result["success"]:
                    successful += 1
                    print(f"✅ Prediction: {result['normalized_prediction']}")
                    print(f"Correct: {result['is_correct']}")
                    print(f"Time: {result['processing_time_ms']}ms")
                    print(f"Cost: ${result['api_cost_usd']:.6f}")
                else:
                    print(f"❌ Error: {result['error']}")
                    failed += 1
                    
                    if result.get('error') == "API quota exceeded":
                        print("⚠️  Quota exceeded. Stopping.")
                        results.append(result)
                        break
                        
            except Exception as e:
                print(f"❌ Error: {e}")
                result = create_error_result(row.to_dict(), str(e))
                failed += 1

        results.append(result)
        
        # Small delay between individual requests within batch
        if i % batch_size != 0 and i < sample_size:
            time.sleep(2)  # 2 second delay between images in same batch
        
        # Save intermediate results every batch
        if i % batch_size == 0:
            os.makedirs("results", exist_ok=True)
            with open("results/gpt_results_partial.json", 'w') as f:
                json.dump(results, f, indent=2)
            print(f"💾 Saved intermediate results ({i} images processed)")

    # Save final results
    os.makedirs("results", exist_ok=True)
    
    # Save detailed JSON (matching Claude format)
    with open("results/gpt_classification_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save simple CSV for compatibility
    csv_results = []
    for r in results:
        csv_results.append({
            "image_id": r["image_id"],
            "true_label": r["ground_truth_label"],
            "predicted_label": r["normalized_prediction"],
            "path": r["image_path"],
            "success": r["success"],
            "is_correct": r["is_correct"],
            "processing_time_ms": r["processing_time_ms"],
            "api_cost_usd": r["api_cost_usd"],
            "error": r["error"]
        })
    
    csv_df = pd.DataFrame(csv_results)
    csv_df.to_csv("results/predictions.csv", index=False)
    
    # Summary
    print(f"\n" + "=" * 50)
    print(f"📊 FINAL SUMMARY")
    print("=" * 50)
    print(f"Processed: {len(results)} images")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if successful > 0:
        correct = sum(1 for r in results if r['is_correct'])
        accuracy = (correct / successful) * 100
        print(f"Accuracy: {accuracy:.1f}% ({correct}/{successful})")
        
        total_cost = sum(r['api_cost_usd'] for r in results if r['success'])
        print(f"Total API cost: ${total_cost:.6f}")
    
    print(f"\n💾 Results saved to:")
    print(f"  - results/gpt_classification_results.json (detailed, Claude-compatible)")
    print(f"  - results/predictions.csv (simple CSV)")
    print(f"\n📊 Analyze results:")
    print(f"  python src/analyze_gpt_results.py --results results/gpt_classification_results.json")

if __name__ == "__main__":
    main()
