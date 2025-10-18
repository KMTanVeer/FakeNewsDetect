"""
Main training script for fake news detection model.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DataLoader, create_sample_dataset
from src.preprocessing import TextPreprocessor, FeatureExtractor
from src.models import FakeNewsDetector, compare_models
import argparse


def train_model(data_path, model_type='logistic_regression', output_dir='models'):
    """
    Train a fake news detection model.
    
    Args:
        data_path (str): Path to training data CSV
        model_type (str): Type of model to train
        output_dir (str): Directory to save trained model
    """
    print("=" * 60)
    print("FAKE NEWS DETECTION - MODEL TRAINING")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data...")
    loader = DataLoader()
    
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}")
        print("Creating sample dataset...")
        create_sample_dataset(data_path)
    
    texts, labels = loader.load_csv(data_path)
    print(f"   Loaded {len(texts)} samples")
    
    # Split data
    print("\n2. Splitting data...")
    X_train_raw, X_test_raw, y_train, y_test = loader.split_data(texts, labels)
    print(f"   Training samples: {len(X_train_raw)}")
    print(f"   Test samples: {len(X_test_raw)}")
    
    # Preprocess text
    print("\n3. Preprocessing text...")
    preprocessor = TextPreprocessor()
    X_train_clean = [preprocessor.preprocess(text) for text in X_train_raw]
    X_test_clean = [preprocessor.preprocess(text) for text in X_test_raw]
    print("   Text preprocessing complete")
    
    # Extract features
    print("\n4. Extracting features...")
    feature_extractor = FeatureExtractor(max_features=5000)
    X_train = feature_extractor.fit_transform(X_train_clean)
    X_test = feature_extractor.transform(X_test_clean)
    print(f"   Feature shape: {X_train.shape}")
    
    # Train model
    print(f"\n5. Training {model_type} model...")
    detector = FakeNewsDetector(model_type)
    detector.train(X_train, y_train)
    print("   Training complete")
    
    # Evaluate
    print("\n6. Evaluating model...")
    metrics = detector.evaluate(X_test, y_test)
    print("\n   Performance Metrics:")
    print(f"   - Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   - Precision: {metrics['precision']:.4f}")
    print(f"   - Recall:    {metrics['recall']:.4f}")
    print(f"   - F1-Score:  {metrics['f1_score']:.4f}")
    
    # Save model and feature extractor
    print(f"\n7. Saving model...")
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f'{model_type}_model.pkl')
    vectorizer_path = os.path.join(output_dir, 'vectorizer.pkl')
    
    detector.save(model_path)
    import joblib
    joblib.dump(feature_extractor, vectorizer_path)
    
    print(f"   Model saved to: {model_path}")
    print(f"   Vectorizer saved to: {vectorizer_path}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    return detector, feature_extractor


def compare_all_models(data_path):
    """
    Compare performance of different models.
    
    Args:
        data_path (str): Path to training data CSV
    """
    print("=" * 60)
    print("COMPARING MULTIPLE MODELS")
    print("=" * 60)
    
    # Load and prepare data
    print("\nPreparing data...")
    loader = DataLoader()
    
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}")
        print("Creating sample dataset...")
        create_sample_dataset(data_path)
    
    texts, labels = loader.load_csv(data_path)
    X_train_raw, X_test_raw, y_train, y_test = loader.split_data(texts, labels)
    
    # Preprocess
    preprocessor = TextPreprocessor()
    X_train_clean = [preprocessor.preprocess(text) for text in X_train_raw]
    X_test_clean = [preprocessor.preprocess(text) for text in X_test_raw]
    
    # Extract features
    feature_extractor = FeatureExtractor(max_features=5000)
    X_train = feature_extractor.fit_transform(X_train_clean)
    X_test = feature_extractor.transform(X_test_clean)
    
    # Compare models
    print("\nTraining and evaluating models...")
    results = compare_models(X_train, X_test, y_train, y_test)
    
    # Print summary
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print("\n" + "=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train fake news detection model'
    )
    parser.add_argument(
        '--data', 
        type=str, 
        default='data/sample_news.csv',
        help='Path to training data CSV'
    )
    parser.add_argument(
        '--model', 
        type=str, 
        default='logistic_regression',
        choices=[
            'logistic_regression', 
            'naive_bayes', 
            'random_forest', 
            'svm'
        ],
        help='Type of model to train'
    )
    parser.add_argument(
        '--compare', 
        action='store_true',
        help='Compare all models'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='models',
        help='Directory to save trained model'
    )
    
    args = parser.parse_args()
    
    if args.compare:
        compare_all_models(args.data)
    else:
        train_model(args.data, args.model, args.output)
