"""
Prediction script for fake news detection.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import TextPreprocessor
import joblib
import argparse


class NewsPredictor:
    """Predicts whether news is fake or real."""
    
    def __init__(self, model_path='models/logistic_regression_model.pkl', 
                 vectorizer_path='models/vectorizer.pkl'):
        """
        Initialize predictor.
        
        Args:
            model_path (str): Path to trained model
            vectorizer_path (str): Path to fitted vectorizer
        """
        self.preprocessor = TextPreprocessor()
        
        # Load model and vectorizer
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Vectorizer not found at {vectorizer_path}")
        
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        print(f"Model loaded from: {model_path}")
        print(f"Vectorizer loaded from: {vectorizer_path}")
    
    def predict(self, text):
        """
        Predict if text is fake or real news.
        
        Args:
            text (str): News text to classify
            
        Returns:
            dict: Prediction result with label and confidence
        """
        # Preprocess
        cleaned_text = self.preprocessor.preprocess(text)
        
        # Extract features
        features = self.vectorizer.transform([cleaned_text])
        
        # Predict
        prediction = self.model.predict(features)[0]
        
        # Get probability if available
        confidence = None
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(features)[0]
            confidence = float(max(proba))
        
        result = {
            'prediction': 'FAKE' if prediction == 1 else 'REAL',
            'label': int(prediction),
            'confidence': confidence,
            'text': text[:100] + '...' if len(text) > 100 else text
        }
        
        return result
    
    def predict_batch(self, texts):
        """
        Predict for multiple texts.
        
        Args:
            texts (list): List of news texts
            
        Returns:
            list: List of prediction results
        """
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(
        description='Predict fake news'
    )
    parser.add_argument(
        '--text', 
        type=str,
        help='News text to classify'
    )
    parser.add_argument(
        '--file', 
        type=str,
        help='File containing news texts (one per line)'
    )
    parser.add_argument(
        '--model', 
        type=str, 
        default='models/logistic_regression_model.pkl',
        help='Path to trained model'
    )
    parser.add_argument(
        '--vectorizer', 
        type=str, 
        default='models/vectorizer.pkl',
        help='Path to fitted vectorizer'
    )
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = NewsPredictor(args.model, args.vectorizer)
    
    print("\n" + "=" * 60)
    print("FAKE NEWS DETECTION - PREDICTION")
    print("=" * 60)
    
    # Predict single text
    if args.text:
        print(f"\nInput: {args.text}")
        result = predictor.predict(args.text)
        print(f"\nPrediction: {result['prediction']}")
        if result['confidence']:
            print(f"Confidence: {result['confidence']:.2%}")
    
    # Predict from file
    elif args.file:
        with open(args.file, 'r') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        results = predictor.predict_batch(texts)
        
        print(f"\nProcessed {len(results)} texts:\n")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['text']}")
            print(f"   Prediction: {result['prediction']}", end='')
            if result['confidence']:
                print(f" (Confidence: {result['confidence']:.2%})")
            else:
                print()
            print()
    
    # Interactive mode
    else:
        print("\nInteractive mode - Enter news text (or 'quit' to exit):")
        while True:
            print("\n" + "-" * 60)
            text = input("Enter news text: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("Exiting...")
                break
            
            if not text:
                print("Please enter some text.")
                continue
            
            result = predictor.predict(text)
            print(f"\n✓ Prediction: {result['prediction']}")
            if result['confidence']:
                print(f"  Confidence: {result['confidence']:.2%}")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
