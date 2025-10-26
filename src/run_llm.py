import os
import base64
import pandas as pd
import random
import time
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

def classify_image(image_path: str, max_retries: int = 3) -> str:
    """Send image to GPT-4o-mini with classification prompt."""
    base64_img = encode_image(image_path)

    prompt = """
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
- Symmetrical rash patterns may indicate dermatitis (not acne)
    """

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # can swap for claude/other later
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
            )

            return response.choices[0].message.content.strip()
            
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower():
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 60  # 1min, 2min, 4min
                    print(f"Rate limit hit. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"Max retries reached. Quota exceeded.")
                    return "QUOTA_EXCEEDED"
            else:
                # For other errors, don't retry
                raise e
    
    return "ERROR"

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
            pred = "FILE_NOT_FOUND"
        else:
            try:
                pred = classify_image(img_path)
                if pred == "QUOTA_EXCEEDED":
                    print("⚠️  Quota exceeded. Stopping.")
                    break
                elif pred != "ERROR":
                    successful += 1
                    print(f"✅ Prediction: {pred}")
                    print(f"Correct: {pred == true_label}")
                else:
                    print(f"❌ Classification error")
            except Exception as e:
                print(f"❌ Error: {e}")
                pred = "ERROR"

        results.append({
            "image_id": row["image_id"],
            "true_label": true_label,
            "predicted_label": pred,
            "path": img_path,
            "success": pred not in ["ERROR", "FILE_NOT_FOUND", "QUOTA_EXCEEDED"],
            "is_correct": pred == true_label and pred not in ["ERROR", "FILE_NOT_FOUND", "QUOTA_EXCEEDED"],
            "error": None if pred not in ["ERROR", "FILE_NOT_FOUND", "QUOTA_EXCEEDED"] else pred
        })
        
        # Small delay between individual requests within batch
        if i % batch_size != 0 and i < sample_size:
            time.sleep(2)  # 2 second delay between images in same batch
        
        # Save intermediate results every batch
        if i % batch_size == 0:
            results_df = pd.DataFrame(results)
            os.makedirs("results", exist_ok=True)
            results_df.to_csv("results/predictions_partial.csv", index=False)
            print(f"💾 Saved intermediate results ({i} images processed)")

    # Save final results
    results_df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    results_df.to_csv("results/predictions.csv", index=False)
    
    # Summary
    print(f"\n" + "=" * 50)
    print(f"📊 FINAL SUMMARY")
    print("=" * 50)
    print(f"Processed: {len(results)} images")
    print(f"Successful: {successful}")
    if successful > 0:
        correct = sum(1 for r in results if r['predicted_label'] == r['true_label'] and r['predicted_label'] not in ['ERROR', 'FILE_NOT_FOUND', 'QUOTA_EXCEEDED'])
        accuracy = (correct / successful) * 100
        print(f"Accuracy: {accuracy:.1f}% ({correct}/{successful})")
    
    print(f"💾 Results saved to: results/predictions.csv")
    print(f"📊 Analyze results: python src/analyze_gpt_results.py --csv results/predictions.csv")

if __name__ == "__main__":
    main()
