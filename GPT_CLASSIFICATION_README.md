# GPT Skin Condition Classification

This directory contains scripts to classify skin conditions using OpenAI GPT API on your skin disease dataset.

## Files

- `src/run_llm.py` - Main classification script (simple, works with metadata.csv)
- `src/gpt_skin_classifier.py` - Advanced classification class (matches Claude structure)
- `src/run_gpt_classification.py` - Runner for advanced classifier
- `src/analyze_gpt_results.py` - Results analysis and evaluation
- `data/metadata.csv` - Your dataset metadata with image paths

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Get OpenAI API key:**
   - Visit [OpenAI Platform](https://platform.openai.com/api-keys)
   - Create an account and get your API key

3. **Set environment variable:**
   Create a `.env` file in the project root:
   ```bash
   OPENAI_API_KEY=your-api-key-here
   ```

## Usage

### Quick Start (Recommended)

Run the simple classification script:
```bash
python src/run_llm.py
```

This will:
- Load images from your `data/metadata.csv` file
- Ask how many images you want to process (default: 20)
- Process the images and save results to `results/predictions.csv`
- Show progress and handle rate limits gracefully

### Advanced Usage

For more detailed analysis, use the advanced classifier:
```bash
python src/run_gpt_classification.py
```

This requires the `balanced_dataset/ground_truth_labels.csv` file (from Claude workflow).

### Direct Script Usage

Run the main classifier directly:
```bash
python src/gpt_skin_classifier.py \
    --api-key "your-api-key" \
    --ground-truth "balanced_dataset/ground_truth_labels.csv" \
    --output "my_results.json" \
    --max-images 50
```

## Analysis

After running classification, analyze the results:

```bash
# Analyze simple results
python src/analyze_gpt_results.py --csv results/predictions.csv

# Analyze detailed results (if using advanced classifier)
python src/analyze_gpt_results.py --results gpt_classification_results.json
```

## Output Files

### Simple Workflow (`run_llm.py`)
- `results/predictions.csv` - Simple CSV with predictions
- `results/gpt_detailed_results.json` - Detailed JSON results (if available)

### Advanced Workflow (`run_gpt_classification.py`)
- `gpt_classification_results.json` - Detailed JSON with full analysis data

## Results Format

### CSV Format (Simple)
```csv
image_id,true_label,predicted_label,path,success,is_correct,error
angioedema-4.jpg,Urticarial Hives,Urticarial Hives,/path/to/image.jpg,True,True,
```

### JSON Format (Detailed)
```json
{
  "image_id": 123,
  "image_filename": "acne-sample.jpg",
  "ground_truth_label": "Acne",
  "predicted_label": "Acne", 
  "is_correct": true,
  "processing_time_ms": 1250.5,
  "api_cost_usd": 0.002,
  "success": true,
  "error": null,
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## Analysis Output

The analysis script provides:

- **Basic Statistics**: Success rate, total processed
- **Accuracy Analysis**: Overall and per-category accuracy
- **Confusion Matrix**: Detailed classification matrix
- **Misclassification Analysis**: Most common errors
- **Performance Metrics**: Processing time, API costs
- **Error Analysis**: Breakdown of failures
- **Sample Predictions**: Examples of incorrect classifications

## Models Supported

- `gpt-4o-mini` (default) - Fast and cost-effective
- `gpt-4o` - Higher accuracy, more expensive
- `gpt-4-turbo` - Good balance of speed and accuracy

Change model in the script:
```python
model="gpt-4o"  # or gpt-4-turbo
```

## Cost Estimation

Approximate costs per image:
- GPT-4o-mini: ~$0.002
- GPT-4o: ~$0.01
- GPT-4-turbo: ~$0.005

For 1000 images:
- GPT-4o-mini: ~$2.00
- GPT-4o: ~$10.00
- GPT-4-turbo: ~$5.00

## Rate Limits

The scripts handle OpenAI rate limits automatically:
- Exponential backoff retry (1min, 2min, 4min)
- Graceful quota exceeded handling
- Progress saving every 10 images

## Skin Conditions Classified

The system classifies into 20 normalized conditions:

1. Acne
2. Actinic Carcinoma
3. Atopic Dermatitis
4. Bullous Disease
5. Cellulitis
6. Eczema
7. Drug Eruptions
8. Herpes HPV
9. Light Diseases
10. Lupus
11. Melanoma
12. Poison Ivy
13. Psoriasis
14. Benign Tumors
15. Systemic Disease
16. Ringworm
17. Urticarial Hives
18. Vascular Tumors
19. Vasculitis
20. Viral Infections

## Example Workflow

1. **Test with small subset:**
   ```bash
   python src/run_llm.py
   # Enter 10 when asked for number of images
   ```

2. **Analyze test results:**
   ```bash
   python src/analyze_gpt_results.py --csv results/predictions.csv
   ```

3. **Run larger evaluation:**
   ```bash
   python src/run_llm.py
   # Enter 100 when asked for number of images
   ```

4. **Analyze full results:**
   ```bash
   python src/analyze_gpt_results.py --csv results/predictions.csv
   ```

## Comparison with Claude

To compare GPT and Claude results:

1. Run both classifiers on the same dataset
2. Use the analysis scripts to generate metrics
3. Compare accuracy, processing time, and costs
4. Analyze confusion matrices for different error patterns

## Troubleshooting

### Common Issues

1. **API Key Error**: Make sure `OPENAI_API_KEY` is set in `.env`
2. **Rate Limit**: Script will automatically retry with delays
3. **Quota Exceeded**: Check your OpenAI billing and usage
4. **File Not Found**: Verify image paths in `metadata.csv` are correct
5. **Import Errors**: Run `pip install -r requirements.txt`

### Debug Mode

Add debug prints by modifying the scripts or run with verbose output.

This gives you comprehensive evaluation metrics including accuracy, confusion matrices, and detailed error analysis comparable to the Claude workflow.
