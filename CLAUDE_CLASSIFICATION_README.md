# Claude Skin Condition Classification

This directory contains scripts to classify skin conditions using Claude API on the balanced dataset.

## Files

- `src/claude_skin_classifier.py` - Main classification script
- `src/run_claude_classification.py` - Easy-to-use runner script
- `src/analyze_claude_results.py` - Results analysis and evaluation
- `balanced_dataset/ground_truth_labels.csv` - Ground truth labels with acne types

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Get Claude API key:**
   - Visit [Anthropic Console](https://console.anthropic.com/)
   - Create an account and get your API key

3. **Set environment variable:**
   ```bash
   export ANTHROPIC_API_KEY='your-api-key-here'
   ```

## Usage

### Quick Start (Recommended)

Run the easy-to-use script:
```bash
python src/run_claude_classification.py
```

This will:
- Ask if you want to run in test mode (10 images) or full mode (300 images)
- Process the images and save results to `claude_classification_results.json`
- Show progress and intermediate results

### Advanced Usage

Run the main classifier directly:
```bash
python src/claude_skin_classifier.py \
    --api-key "your-api-key" \
    --ground-truth "balanced_dataset/ground_truth_labels.csv" \
    --output "my_results.json" \
    --max-images 50
```

### Analyze Results

After classification, analyze the results:
```bash
python src/analyze_claude_results.py --results claude_classification_results.json
```

## Dataset Structure

The balanced dataset contains 300 images:

### Acne Images (150 total)
- **True Acne (100 images):**
  - 30 Cystic Acne
  - 25 Pustular Acne  
  - 20 Open Comedo (Blackhead)
  - 15 Closed Comedo (Whitehead)
  - 10 General Acne

- **Acne-like/Confusable (50 images):**
  - 30 Rosacea
  - 20 Perioral Dermatitis

### Non-Acne Images (150 total)
- **Inflammatory/Similar (70 images):** Atopic Dermatitis, Eczema, Psoriasis, Poison Ivy
- **Infections (30 images):** Cellulitis, Ringworm, Herpes HPV
- **Serious Conditions (20 images):** Melanoma, Actinic Carcinoma
- **Other Conditions (30 images):** Vascular Tumors, Lupus, Light Diseases, Systemic Disease

## Classification Categories

### Acne Types
- Comedonal acne (blackheads and whiteheads)
- Papular acne (small red bumps without pus)
- Pustular acne (pus-filled pimples with white/yellow centers)
- Nodular acne (large, painful lumps beneath the skin)
- Cystic acne (deep, painful, pus-filled cysts)
- Acne conglobata (severe, interconnected lesions)

### Non-Acne
- Not acne (rosacea, eczema, dermatitis, or other skin conditions)

## Output Format

The script generates JSON output in this format:

```json
[
  {
    "image_id": 0,
    "image_filename": "acne-cystic-118.jpg",
    "image_path": "/path/to/image.jpg",
    "ground_truth_label": "acne",
    "ground_truth_subtype": "cystic_acne",
    "model": "claude-3-5-sonnet-20241022",
    "timestamp": "2025-01-23T10:30:45Z",
    "api_response": {
      "classification": "Cystic acne (deep, painful, pus-filled cysts)",
      "confidence": "high",
      "reasoning": "Multiple deep, inflamed cysts visible with surrounding erythema",
      "key_features": ["deep cysts", "inflammation", "erythema", "facial distribution"]
    },
    "normalized_prediction": "acne",
    "predicted_subtype": "cystic",
    "is_correct": true,
    "processing_time_ms": 1250,
    "api_cost_usd": 0.0,
    "success": true,
    "error": null
  }
]
```

## Features

- **Robust Error Handling:** Continues processing even if some images fail
- **Progress Tracking:** Shows progress and saves intermediate results
- **Flexible Testing:** Can process subset of images for testing
- **Comprehensive Analysis:** Detailed evaluation metrics and confusion matrices
- **Cost Tracking:** Monitors API usage (cost calculation can be added)

## Performance Considerations

- **Processing Time:** ~1-2 seconds per image
- **API Costs:** Claude pricing applies (check current rates)
- **Rate Limits:** Built-in error handling for API limits
- **Memory Usage:** Processes images one at a time to minimize memory usage

## Troubleshooting

### Common Issues

1. **API Key Error:**
   ```
   Error: Please set the ANTHROPIC_API_KEY environment variable
   ```
   Solution: Set your API key as shown in setup instructions

2. **Image Not Found:**
   ```
   ❌ Image not found: /path/to/image.jpg
   ```
   Solution: Ensure the ground truth CSV has correct image paths

3. **JSON Parse Error:**
   ```
   Unable to parse JSON response
   ```
   Solution: This is handled gracefully - the response is still captured

### Getting Help

- Check that all dependencies are installed: `pip install -r requirements.txt`
- Verify your API key is valid and has sufficient credits
- For large datasets, consider using `--max-images` for testing first

## Example Workflow

1. **Test with small subset:**
   ```bash
   python src/run_claude_classification.py
   # Choose 'y' for test mode (10 images)
   ```

2. **Analyze test results:**
   ```bash
   python src/analyze_claude_results.py
   ```

3. **Run full classification:**
   ```bash
   python src/run_claude_classification.py
   # Choose 'n' for full mode (300 images)
   ```

4. **Analyze full results:**
   ```bash
   python src/analyze_claude_results.py
   ```

This will give you comprehensive evaluation metrics including accuracy, confusion matrices, and detailed error analysis.
