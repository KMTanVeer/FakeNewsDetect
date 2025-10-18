"""
Basic tests for the fake news detection system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import TextPreprocessor, FeatureExtractor
from src.models import FakeNewsDetector
from src.data_loader import DataLoader, create_sample_dataset
import tempfile
import shutil


def test_text_preprocessor():
    """Test text preprocessing functionality."""
    print("Testing TextPreprocessor...")
    
    preprocessor = TextPreprocessor()
    
    # Test cleaning
    text = "BREAKING: Check out this link! http://example.com #FakeNews @user123"
    cleaned = preprocessor.clean_text(text)
    assert "http" not in cleaned.lower()
    assert "@" not in cleaned
    assert "#" not in cleaned
    
    # Test full preprocessing
    processed = preprocessor.preprocess(text)
    assert len(processed) > 0
    assert isinstance(processed, str)
    
    print("  ✓ Text preprocessing works correctly")


def test_feature_extractor():
    """Test feature extraction."""
    print("Testing FeatureExtractor...")
    
    texts = [
        "This is a sample text",
        "Another sample document",
        "Third text for testing"
    ]
    
    extractor = FeatureExtractor(max_features=100)
    features = extractor.fit_transform(texts)
    
    assert features.shape[0] == 3
    assert features.shape[1] > 0
    
    # Test transform on new data
    new_features = extractor.transform(["New text to transform"])
    assert new_features.shape[0] == 1
    
    print("  ✓ Feature extraction works correctly")


def test_model_training():
    """Test model training and prediction."""
    print("Testing FakeNewsDetector...")
    
    # Create sample data
    X_train = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]
    y_train = [0, 1, 0, 1]
    X_test = [[0.2, 0.3], [0.6, 0.7]]
    y_test = [0, 1]
    
    # Test with Naive Bayes (works with sparse features)
    detector = FakeNewsDetector('naive_bayes')
    detector.train(X_train, y_train)
    
    assert detector.trained
    
    # Test prediction
    predictions = detector.predict(X_test)
    assert len(predictions) == 2
    
    # Test evaluation
    metrics = detector.evaluate(X_test, y_test)
    assert 'accuracy' in metrics
    assert 'f1_score' in metrics
    
    print("  ✓ Model training and prediction works correctly")


def test_data_loader():
    """Test data loading functionality."""
    print("Testing DataLoader...")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create sample dataset
        csv_path = os.path.join(temp_dir, 'test_data.csv')
        create_sample_dataset(csv_path, n_samples=100)
        
        # Load data
        loader = DataLoader()
        texts, labels = loader.load_csv(csv_path)
        
        assert len(texts) == 100
        assert len(labels) == 100
        
        # Test split
        X_train, X_test, y_train, y_test = loader.split_data(texts, labels)
        assert len(X_train) + len(X_test) == 100
        
        print("  ✓ Data loading works correctly")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def test_end_to_end():
    """Test complete pipeline."""
    print("Testing end-to-end pipeline...")
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create sample data
        csv_path = os.path.join(temp_dir, 'test_data.csv')
        create_sample_dataset(csv_path, n_samples=100)
        
        # Load and split
        loader = DataLoader()
        texts, labels = loader.load_csv(csv_path)
        X_train_raw, X_test_raw, y_train, y_test = loader.split_data(texts, labels, test_size=0.3)
        
        # Preprocess
        preprocessor = TextPreprocessor()
        X_train_clean = [preprocessor.preprocess(text) for text in X_train_raw]
        X_test_clean = [preprocessor.preprocess(text) for text in X_test_raw]
        
        # Extract features
        feature_extractor = FeatureExtractor(max_features=100)
        X_train = feature_extractor.fit_transform(X_train_clean)
        X_test = feature_extractor.transform(X_test_clean)
        
        # Train model
        detector = FakeNewsDetector('logistic_regression')
        detector.train(X_train, y_train)
        
        # Evaluate
        metrics = detector.evaluate(X_test, y_test)
        
        assert metrics['accuracy'] >= 0.5  # Should be better than random
        
        print("  ✓ End-to-end pipeline works correctly")
        
    finally:
        shutil.rmtree(temp_dir)


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("RUNNING TESTS FOR FAKE NEWS DETECTION SYSTEM")
    print("=" * 60 + "\n")
    
    tests = [
        test_text_preprocessor,
        test_feature_extractor,
        test_model_training,
        test_data_loader,
        test_end_to_end
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ Test failed: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60 + "\n")
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
