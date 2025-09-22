# python src/ingest.py
import os
import pandas as pd
import kagglehub

def build_metadata():
    dataset_path = kagglehub.dataset_download("haroonalam16/20-skin-diseases-dataset")

    dataset_root = os.path.join(dataset_path, "Dataset")

    rows = []
    for split in ["train", "test"]:
        split_dir = os.path.join(dataset_root, split)
        if not os.path.isdir(split_dir):
            continue

        for disease in os.listdir(split_dir):
            disease_dir = os.path.join(split_dir, disease)
            if not os.path.isdir(disease_dir):
                continue
            for img in os.listdir(disease_dir):
                if img.lower().endswith((".jpg", ".jpeg", ".png")):
                    rows.append({
                        "image_id": img,
                        "condition": disease,
                        "path": os.path.join(disease_dir, img),
                        "split": split,
                        "skin_tone": None  # optional
                    })
    df = pd.DataFrame(rows)
    os.makedirs("data", exist_ok=True)
    
    NORMALIZED_LABELS = {
        "Acne and Rosacea Photos": "Acne",
        "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions": "Actinic Carcinoma",
        "Atopic Dermatitis Photos": "Atopic Dermatitis",
        "Bullous Disease Photos": "Bullous Disease",
        "Cellulitis Impetigo and other Bacterial Infections": "Cellulitis",
        "Eczema Photos": "Eczema",
        "Drug Eruptions Photos": "Drug Eruptions",
        "Herpes HPV and other STDs Photos": "Herpes HPV",
        "Light Diseases and Disorders of Pigmentation": "Light Diseases",
        "Lupus and other Connective Tissue diseases": "Lupus",
        "Melanoma Skin Cancer Nevi and Moles": "Melanoma",
        "Poison Ivy Photos and other Contact Dermatitis": "Poison Ivy",
        "Psoriasis pictures Lichen Planus and related diseases": "Psoriasis",
        "Benign Tumors": "Benign Tumors",
        "Systemic Disease": "Systemic Disease",
        "Tinea Ringworm Candidiasis and other Fungal Infections": "Ringworm",
        "Urticaria Hives": "Urticarial Hives",
        "Vascular Tumors": "Vascular Tumors",
        "Vasculitis Photos": "Vasculitis",
        "Viral Infections": "Viral Infections"
    }
    df["condition"] = df["condition"].map(NORMALIZED_LABELS)

    df.to_csv("data/metadata.csv", index=False)
    print(f"Saved {len(df)} rows across {df['condition'].nunique()} conditions to data/metadata.csv")

    # quick summary
    print(df.groupby(["split", "condition"]).size().head(20))

if __name__ == "__main__":
    build_metadata()
