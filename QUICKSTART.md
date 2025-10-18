# Fake News Detection - Quick Start Guide

## Installation

```bash
# Clone repository
git clone https://github.com/KMTanVeer/FakeNewsDetect.git
cd FakeNewsDetect

# Install dependencies
pip install -r requirements.txt
```

## Quick Usage

### 1. Run Example Demo
```bash
python example.py
```

### 2. Train a Model
```bash
# With sample data
python train.py

# With your own data
python train.py --data path/to/your/data.csv --model logistic_regression
```

### 3. Make Predictions
```bash
# Interactive mode
python predict.py

# Single prediction
python predict.py --text "Your news text here"

# Batch predictions from file
python predict.py --file texts.txt
```

### 4. Compare Models
```bash
python train.py --compare
```

## Model Options

- `logistic_regression` - Fast, good baseline (default)
- `naive_bayes` - Fast, works well with text
- `random_forest` - Ensemble method, robust
- `svm` - Support Vector Machine, accurate

## Data Format

CSV file with columns:
- `text` - News article text
- `label` - 0 for real, 1 for fake

Example:
```csv
text,label
"Government announces new policy",0
"You won't believe this!",1
```

## Project Structure

```
FakeNewsDetect/
├── src/               # Core modules
│   ├── preprocessing.py
│   ├── models.py
│   └── data_loader.py
├── train.py          # Training script
├── predict.py        # Prediction script
├── example.py        # Demo script
├── tests/            # Test suite
└── requirements.txt  # Dependencies
```

## Run Tests

```bash
python tests/test_fake_news_detection.py
```

## Common Tasks

### Train with Custom Parameters
```python
from src.models import FakeNewsDetector
from src.preprocessing import FeatureExtractor

# Create custom feature extractor
features = FeatureExtractor(max_features=10000)

# Train specific model
detector = FakeNewsDetector('random_forest')
detector.train(X_train, y_train)
```

### Batch Predictions
```python
from predict import NewsPredictor

predictor = NewsPredictor()
texts = ["Text 1", "Text 2", "Text 3"]
results = predictor.predict_batch(texts)
```

## Tips

1. **Larger Dataset** = Better Performance
2. **Clean Text** = Better Features
3. **Try Multiple Models** = Find Best Fit
4. **Use test_size=0.2** for smaller datasets
5. **Cross-validation** for robust evaluation

## Performance

Typical metrics with sample data:
- Accuracy: ~85-100% (sample synthetic data)
- F1-Score: ~85-100% (sample synthetic data)

**Note**: With real-world datasets, expect 75-90% accuracy depending on:
- Dataset quality and size
- Text length and complexity
- Class balance (real vs fake distribution)
- Domain specificity

*Performance varies significantly with dataset characteristics*

## Troubleshooting

**NLTK Data Error**
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

**Module Not Found**
```bash
pip install -r requirements.txt
```

**Low Accuracy**
- Use more training data
- Clean your text better
- Try different models
- Adjust feature parameters

## Next Steps

1. ✓ Run example.py
2. ✓ Train model with sample data
3. ✓ Test predictions
4. → Use your own dataset
5. → Experiment with parameters
6. → Deploy your model

## Support

- GitHub Issues: Report bugs
- Documentation: README.md
- Tests: tests/ directory

Happy detecting! 🔍
