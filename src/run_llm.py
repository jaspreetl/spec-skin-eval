import os
import base64
import pandas as pd
import random
from openai import OpenAI

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

def classify_image(image_path: str) -> str:
    """Send image to GPT-4o-mini with classification prompt."""
    base64_img = encode_image(image_path)

    prompt = f"""
    You are an expert dermatologist. 
    Classify the following skin image into exactly one of these 20 conditions:
    {", ".join(CONDITIONS)}.
    Respond with only the condition name.
    """

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
        max_tokens=20,
    )

    return response.choices[0].message.content.strip()

def main():
    # Load metadata
    df = pd.read_csv("data/metadata.csv")

    # Drop missing labels if any
    df = df.dropna(subset=["condition"])

    # Sample ~100 random images
    sample_df = df.sample(n=100, random_state=42)

    results = []
    for _, row in sample_df.iterrows():
        true_label = row["condition"]
        img_path = row["path"]

        try:
            pred = classify_image(img_path)
        except Exception as e:
            print(f"Error with {img_path}: {e}")
            pred = "ERROR"

        results.append({
            "image_id": row["image_id"],
            "true_label": true_label,
            "predicted_label": pred,
            "path": img_path
        })

    results_df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    results_df.to_csv("results/predictions.csv", index=False)
    print("Saved predictions to results/predictions.csv")

if __name__ == "__main__":
    main()
