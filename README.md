# Fake News Detection System

A comprehensive machine learning project for detecting fake news using Natural Language Processing (NLP) and various classification algorithms.

## Overview

This project implements a complete pipeline for fake news detection, including:
- Text preprocessing and cleaning
- TF-IDF feature extraction
- Multiple machine learning models
- Training and evaluation scripts
- Real-time prediction capabilities

## Features

- **Multiple ML Models**: Logistic Regression, Naive Bayes, Random Forest, SVM, and Gradient Boosting
- **Text Preprocessing**: Advanced NLP preprocessing including stemming, stopword removal, and cleaning
- **Feature Extraction**: TF-IDF vectorization with n-grams
- **Model Comparison**: Compare performance across different algorithms
- **Easy to Use**: Simple command-line interface for training and prediction
- **Modular Design**: Clean, maintainable code structure

## Project Structure

```
FakeNewsDetect/
├── data/                   # Data directory
├── models/                 # Saved models directory
├── src/                    # Source code
│   ├── data_loader.py     # Data loading utilities
│   ├── preprocessing.py   # Text preprocessing
│   └── models.py          # ML model implementations
├── train.py               # Training script
├── predict.py             # Prediction script
├── example.py             # Example usage
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/KMTanVeer/FakeNewsDetect.git
cd FakeNewsDetect
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data (automatic on first run):
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

## Quick Start

### Run the Example

See the complete workflow in action:
```bash
python example.py
```

### Train a Model

Train with sample data:
```bash
python train.py --data data/sample_news.csv --model logistic_regression
```

Train with your own data:
```bash
python train.py --data path/to/your/data.csv --model naive_bayes
```

Available models:
- `logistic_regression` (default)
- `naive_bayes`
- `random_forest`
- `svm`

### Compare Models

Compare all models:
```bash
python train.py --compare --data data/sample_news.csv
```

### Make Predictions

Interactive mode:
```bash
python predict.py
```

Predict single text:
```bash
python predict.py --text "BREAKING: Scientists discover cure for all diseases!"
```

Predict from file:
```bash
python predict.py --file news_texts.txt
```

## Usage Examples

### Training a Model

```python
from src.data_loader import DataLoader
from src.preprocessing import TextPreprocessor, FeatureExtractor
from src.models import FakeNewsDetector

# Load data
loader = DataLoader()
texts, labels = loader.load_csv('data/news.csv')
X_train, X_test, y_train, y_test = loader.split_data(texts, labels)

# Preprocess
preprocessor = TextPreprocessor()
X_train_clean = [preprocessor.preprocess(text) for text in X_train]
X_test_clean = [preprocessor.preprocess(text) for text in X_test]

# Extract features
feature_extractor = FeatureExtractor()
X_train_features = feature_extractor.fit_transform(X_train_clean)
X_test_features = feature_extractor.transform(X_test_clean)

# Train model
detector = FakeNewsDetector('logistic_regression')
detector.train(X_train_features, y_train)

# Evaluate
metrics = detector.evaluate(X_test_features, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}")
```

### Making Predictions

```python
from predict import NewsPredictor

# Initialize predictor
predictor = NewsPredictor(
    model_path='models/logistic_regression_model.pkl',
    vectorizer_path='models/vectorizer.pkl'
)

# Predict
text = "Breaking news: Major event happened today!"
result = predictor.predict(text)
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## Data Format

Your CSV data should have at least two columns:
- `text`: The news article text
- `label`: 0 for real news, 1 for fake news

Example:
```csv
text,label
"Government announces new policy changes",0
"You won't believe this shocking secret!",1
```

## Model Performance

Using the sample dataset, typical performance metrics:

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Logistic Regression | ~0.85 | ~0.85 |
| Naive Bayes | ~0.82 | ~0.82 |
| Random Forest | ~0.83 | ~0.83 |
| SVM | ~0.84 | ~0.84 |

*Note: Performance varies based on dataset quality and size*

## Preprocessing Pipeline

1. **Text Cleaning**:
   - Convert to lowercase
   - Remove URLs, mentions, hashtags
   - Remove punctuation and numbers
   - Remove extra whitespace

2. **Tokenization and Stemming**:
   - Split text into tokens
   - Remove stopwords
   - Apply Porter Stemmer
   - Filter short tokens

3. **Feature Extraction**:
   - TF-IDF vectorization
   - Unigrams and bigrams
   - Maximum 5000 features

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is a class project and is available for educational purposes.

## Acknowledgments

- NLTK for natural language processing tools
- Scikit-learn for machine learning algorithms
- The open-source community for inspiration and resources

## Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This is an educational project. For production use, consider:
- Using larger, real-world datasets
- Implementing deep learning models (LSTM, BERT)
- Adding more sophisticated feature engineering
- Implementing cross-validation
- Adding API endpoints for integration
