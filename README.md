# BERT Financial Sentiment Analysis

A financial sentiment classifier demonstrating proper ML practices through systematic improvement. This project showcases the evolution from a basic implementation to a production-quality model with comprehensive validation and class imbalance handling.

## Key Achievement

**Improved minority class detection by 206%** (negative recall: 28.6% → 87.4%) through class weights, early stopping, and proper validation metrics.

## Results

### Final Performance (Validation Set)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Negative | 52.8% | **87.4%** | 65.8% | 175 |
| Neutral | 94.3% | 71.2% | 81.1% | 622 |
| Positive | 81.2% | 89.3% | 85.0% | 372 |
| **Overall** | - | - | **79.4%** | 1,169 |

### Before vs After

| Metric | Original | Improved | Change |
|--------|----------|----------|---------|
| Negative Recall | 28.6% | 87.4% | **+206%** |
| Negative F1 | 0.41 | 0.66 | **+61%** |
| Overall Accuracy | 81.5%* | 79.4% | -2.1% |

*Original accuracy was misleading - model ignored minority class despite appearing successful.

## Technical Improvements

### 1. Class Weights for Imbalanced Data
Dataset was severely imbalanced (54% neutral, 32% positive, 15% negative). Implemented weighted loss:
```python
class_weights = torch.FloatTensor([2.27, 0.62, 1.05])  # [negative, neutral, positive]
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

### 2. Early Stopping
Prevented overfitting by monitoring validation loss with patience=3:

| Epoch | Train Loss | Val Loss | Status |
|-------|-----------|----------|---------|
| 2 | 0.375 | **0.447** | Best (saved) |
| 3 | 0.265 | 0.567 |  Worse |
| 4 | 0.207 | 0.628 | Worse |
| 5 | 0.189 | 0.745 | Stopped |

Model automatically reverted to Epoch 2 checkpoint.

### 3. Comprehensive Metrics
Added precision, recall, and F1-score tracking per class - revealed the original model was missing 72% of negative examples despite 81% overall accuracy.

### 4. Learning Rate Scheduling
Implemented linear warmup (10%) + decay for better convergence.

## Dataset

- **Size**: 5,842 financial headlines
- **Split**: 4,673 train / 1,169 validation
- **Classes**: Negative (15%), Neutral (54%), Positive (32%)

## Quick Start

```bash
# Clone and install
git clone https://github.com/M-Rodani1/bert-financial-sentiment.git
cd bert-financial-sentiment
pip install -r requirements.txt

# Train model
python final_training.py
```

Model automatically stops at optimal point and displays per-class metrics.

## What This Project Demonstrates

**ML Best Practices:**
- Proper train/validation split with comprehensive evaluation
- Handling class imbalance with weighted loss
- Early stopping to prevent overfitting
- Per-class metrics (F1-score) over accuracy alone
- Honest reporting of model limitations

**Key Insight:** Dropped 2% overall accuracy to gain 58 percentage points in minority class detection - the right tradeoff for financial applications where missing negative sentiment is costly.

## Example Predictions

- "Tesla stock surged 15% after beating earnings" → **Positive** 
- "Company announces massive layoffs amid struggles" → **Negative** 
- "Market remains stable with modest volumes" → **Neutral** 

High negative recall (87%) makes it suitable for applications where detecting bad news is critical.

## Technologies

- **PyTorch** + **Transformers** - BERT fine-tuning
- **scikit-learn** - Metrics and evaluation
- **Pandas** - Data processing

## Documentation

See [IMPROVEMENTS.md](IMPROVEMENTS.md) for detailed technical documentation of all enhancements.

## Future Work

- K-fold cross-validation
- Confusion matrix visualization
- Try FinBERT (domain-specific model)
- Data augmentation for minority class
- REST API deployment

## Key Takeaway

*This project demonstrates that real ML is about solving actual problems, not achieving vanity metrics. A 79% accuracy model that catches 87% of negative examples is more valuable than an 81% accuracy model that misses 72% of them.*

---

**Author:** M-Rodani1 | [GitHub](https://github.com/M-Rodani1)
