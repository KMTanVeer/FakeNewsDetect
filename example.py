"""
Example usage of the fake news detection system.
Demonstrates basic workflow.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import create_sample_dataset
from src.preprocessing import TextPreprocessor
from src.models import FakeNewsDetector
from src.data_loader import DataLoader


def main():
    """Demonstrate basic usage of the fake news detection system."""
    
    print("=" * 70)
    print("FAKE NEWS DETECTION SYSTEM - EXAMPLE USAGE")
    print("=" * 70)
    
    # 1. Create sample dataset
    print("\n1. Creating sample dataset...")
    data_path = 'data/sample_news.csv'
    os.makedirs('data', exist_ok=True)
    create_sample_dataset(data_path, n_samples=1000)
    
    # 2. Load data
    print("\n2. Loading data...")
    loader = DataLoader()
    texts, labels = loader.load_csv(data_path)
    print(f"   Loaded {len(texts)} samples")
    
    # 3. Display sample
    print("\n3. Sample data:")
    for i in range(3):
        label_str = "FAKE" if labels[i] == 1 else "REAL"
        print(f"   [{label_str}] {texts[i][:60]}...")
    
    # 4. Split data
    print("\n4. Splitting data into train/test sets...")
    X_train_raw, X_test_raw, y_train, y_test = loader.split_data(texts, labels)
    print(f"   Training: {len(X_train_raw)} samples")
    print(f"   Testing: {len(X_test_raw)} samples")
    
    # 5. Preprocess text
    print("\n5. Preprocessing text...")
    preprocessor = TextPreprocessor()
    
    # Show preprocessing example
    sample_text = X_train_raw[0]
    cleaned_text = preprocessor.preprocess(sample_text)
    print(f"   Original: {sample_text[:50]}...")
    print(f"   Cleaned:  {cleaned_text[:50]}...")
    
    X_train_clean = [preprocessor.preprocess(text) for text in X_train_raw]
    X_test_clean = [preprocessor.preprocess(text) for text in X_test_raw]
    
    # 6. Extract features
    print("\n6. Extracting TF-IDF features...")
    from src.preprocessing import FeatureExtractor
    feature_extractor = FeatureExtractor(max_features=3000)
    X_train = feature_extractor.fit_transform(X_train_clean)
    X_test = feature_extractor.transform(X_test_clean)
    print(f"   Feature matrix shape: {X_train.shape}")
    
    # 7. Train model
    print("\n7. Training Logistic Regression model...")
    detector = FakeNewsDetector('logistic_regression')
    detector.train(X_train, y_train)
    print("   Model trained successfully")
    
    # 8. Evaluate model
    print("\n8. Evaluating model on test set...")
    metrics = detector.evaluate(X_test, y_test)
    print("\n   Performance Metrics:")
    print(f"   ├─ Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   ├─ Precision: {metrics['precision']:.4f}")
    print(f"   ├─ Recall:    {metrics['recall']:.4f}")
    print(f"   └─ F1-Score:  {metrics['f1_score']:.4f}")
    
    # 9. Make predictions on new examples
    print("\n9. Making predictions on sample texts...")
    
    test_examples = [
        "BREAKING: Scientists discover cure for all diseases overnight!",
        "Local government announces new infrastructure development plan",
        "You won't believe this shocking celebrity secret!",
        "Research team publishes findings in peer-reviewed journal"
    ]
    
    for text in test_examples:
        cleaned = preprocessor.preprocess(text)
        features = feature_extractor.transform([cleaned])
        prediction = detector.predict(features)[0]
        label = "FAKE" if prediction == 1 else "REAL"
        
        print(f"\n   Text: {text[:50]}...")
        print(f"   Prediction: {label}")
    
    # 10. Save model
    print("\n10. Saving model...")
    os.makedirs('models', exist_ok=True)
    detector.save('models/example_model.pkl')
    import joblib
    joblib.dump(feature_extractor, 'models/example_vectorizer.pkl')
    print("    Model saved to: models/example_model.pkl")
    print("    Vectorizer saved to: models/example_vectorizer.pkl")
    
    print("\n" + "=" * 70)
    print("EXAMPLE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nNext steps:")
    print("  - Use train.py to train with your own data")
    print("  - Use predict.py to make predictions on new texts")
    print("  - Experiment with different models and parameters")
    print("=" * 70)


if __name__ == '__main__':
    main()
