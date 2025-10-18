"""
Machine learning models for fake news detection.
Implements multiple classifiers for comparison.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, classification_report, confusion_matrix
)
import joblib
import numpy as np


class FakeNewsDetector:
    """Fake news detection model wrapper."""
    
    def __init__(self, model_type='logistic_regression'):
        """
        Initialize detector with specified model.
        
        Args:
            model_type (str): Type of model to use
                Options: 'logistic_regression', 'naive_bayes', 
                        'random_forest', 'svm', 'gradient_boosting'
        """
        self.model_type = model_type
        self.model = self._get_model(model_type)
        self.trained = False
    
    def _get_model(self, model_type):
        """
        Get model instance based on type.
        
        Args:
            model_type (str): Type of model
            
        Returns:
            sklearn model: Model instance
        """
        models = {
            'logistic_regression': LogisticRegression(
                max_iter=1000, 
                random_state=42
            ),
            'naive_bayes': MultinomialNB(),
            'random_forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=42
            ),
            'svm': LinearSVC(
                max_iter=1000, 
                random_state=42
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100, 
                random_state=42
            )
        }
        
        if model_type not in models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return models[model_type]
    
    def train(self, X_train, y_train):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        self.model.fit(X_train, y_train)
        self.trained = True
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Features
            
        Returns:
            array: Predictions
        """
        if not self.trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities.
        
        Args:
            X: Features
            
        Returns:
            array: Prediction probabilities
        """
        if not self.trained:
            raise ValueError("Model must be trained before prediction")
        
        # Some models don't have predict_proba
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        elif hasattr(self.model, 'decision_function'):
            # For SVM, use decision function
            decision = self.model.decision_function(X)
            # Convert to probability-like scores
            proba = np.exp(decision) / (1 + np.exp(decision))
            return np.column_stack([1 - proba, proba])
        else:
            raise NotImplementedError(
                f"Model {self.model_type} doesn't support probability predictions"
            )
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            dict: Evaluation metrics
        """
        y_pred = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        
        return metrics
    
    def save(self, filepath):
        """
        Save model to disk.
        
        Args:
            filepath (str): Path to save model
        """
        if not self.trained:
            raise ValueError("Cannot save untrained model")
        joblib.dump(self.model, filepath)
    
    def load(self, filepath):
        """
        Load model from disk.
        
        Args:
            filepath (str): Path to load model from
        """
        self.model = joblib.load(filepath)
        self.trained = True


def compare_models(X_train, X_test, y_train, y_test):
    """
    Compare performance of multiple models.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        
    Returns:
        dict: Results for each model
    """
    model_types = [
        'logistic_regression', 
        'naive_bayes', 
        'random_forest', 
        'svm'
    ]
    
    results = {}
    
    for model_type in model_types:
        print(f"\nTraining {model_type}...")
        detector = FakeNewsDetector(model_type)
        detector.train(X_train, y_train)
        metrics = detector.evaluate(X_test, y_test)
        results[model_type] = metrics
        
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
    
    return results
