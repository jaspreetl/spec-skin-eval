# Claude Skin Classification Analysis Report

**Generated on:** 2025-10-25 23:43:40

## Executive Summary

- **Total Images Processed:** 300
- **Success Rate:** 100.0%
- **Overall Accuracy:** 60.3%
- **Acne Detection (Sensitivity):** 23.3%
- **Non-Acne Detection (Specificity):** 97.3%

## Key Metrics

| Metric | Value |
|--------|-------|
| Overall Accuracy | 60.3% |
| Macro Precision | 0.728 |
| Macro Recall | 0.603 |
| Macro F1-Score | 0.540 |
| Acne Precision | 89.7% |
| Acne Recall (Sensitivity) | 23.3% |
| Non-Acne Precision | 55.9% |
| Non-Acne Recall (Specificity) | 97.3% |

## Confusion Matrix

| | Predicted Acne | Predicted Non-Acne | Total |
|---|---|---|---|
| **Actual Acne** | 35 | 115 | 150 |
| **Actual Non-Acne** | 4 | 146 | 150 |
| **Total** | 39 | 261 | 300 |

## Analysis by Category

| Category | Accuracy | Correct/Total |
|----------|----------|---------------|
| True Acne | 34.0% | 34/100 |
| Acne-like/Confusable | 2.0% | 1/50 |
| Inflammatory/Similar | 95.7% | 67/70 |
| Infection | 96.7% | 29/30 |
| Serious | 100.0% | 20/20 |
| Other | 100.0% | 30/30 |

## Key Findings

### Strengths
- **High Non-Acne Detection:** 97.3% specificity
- **High Acne Precision:** 89.7% precision when predicting acne
- **Conservative Approach:** Low false positive rate for acne

### Areas for Improvement
- **Low Acne Detection:** Only 23.3% of actual acne cases detected
- **High False Negatives:** 115 acne cases missed
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

