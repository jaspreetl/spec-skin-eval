# Claude Skin Classification Analysis Report

**Generated on:** 2025-11-22 15:12:29

## Executive Summary

- **Total Images Processed:** 300
- **Success Rate:** 100.0%
- **Overall Accuracy:** 56.3%
- **Acne Detection (Sensitivity):** 14.7%
- **Non-Acne Detection (Specificity):** 98.0%

## Key Metrics

| Metric | Value |
|--------|-------|
| Overall Accuracy | 56.3% |
| Macro Precision | 0.707 |
| Macro Recall | 0.563 |
| Macro F1-Score | 0.472 |
| Acne Precision | 88.0% |
| Acne Recall (Sensitivity) | 14.7% |
| Non-Acne Precision | 53.5% |
| Non-Acne Recall (Specificity) | 98.0% |

## Confusion Matrix

| | Predicted Acne | Predicted Non-Acne | Total |
|---|---|---|---|
| **Actual Acne** | 22 | 128 | 150 |
| **Actual Non-Acne** | 3 | 147 | 150 |
| **Total** | 25 | 275 | 300 |

## Analysis by Category

| Category | Accuracy | Correct/Total |
|----------|----------|---------------|
| True Acne | 21.0% | 21/100 |
| Acne-like/Confusable | 2.0% | 1/50 |
| Inflammatory/Similar | 97.1% | 68/70 |
| Infection | 96.7% | 29/30 |
| Serious | 100.0% | 20/20 |
| Other | 100.0% | 30/30 |

## Key Findings

### Strengths
- **High Non-Acne Detection:** 98.0% specificity
- **High Acne Precision:** 88.0% precision when predicting acne
- **Conservative Approach:** Low false positive rate for acne

### Areas for Improvement
- **Low Acne Detection:** Only 14.7% of actual acne cases detected
- **High False Negatives:** 128 acne cases missed
- **Subtype Classification:** Poor performance on specific acne subtypes

## Recommendations

1. **Improve Acne Detection:** Focus on reducing false negatives
2. **Better Subtype Recognition:** Train on more diverse acne subtypes
3. **Acne-like Conditions:** Improve distinction between acne and rosacea/perioral dermatitis
4. **Clinical Validation:** Review misclassified cases with dermatologists

## Technical Details

- **Model:** Claude Sonnet 4 (claude-sonnet-4-20250514)
- **Processing Time:** Average ~4.6 seconds per image
- **Confidence:** 99.7% high confidence predictions
- **API Success Rate:** 100%

