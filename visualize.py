"""
Visualization utilities for fake news detection.
Creates plots and charts for model performance.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from src.data_loader import DataLoader, create_sample_dataset
from src.preprocessing import TextPreprocessor, FeatureExtractor
from src.models import FakeNewsDetector
import argparse


def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    plt.title('Confusion Matrix - Fake News Detection')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {save_path}")
    plt.close()


def plot_model_comparison(results, save_path='model_comparison.png'):
    """
    Plot comparison of different models.
    
    Args:
        results: Dictionary of model results
        save_path: Path to save plot
    """
    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for idx, metric in enumerate(metrics):
        values = [results[model][metric] for model in models]
        
        axes[idx].bar(range(len(models)), values, color='skyblue', edgecolor='navy')
        axes[idx].set_xticks(range(len(models)))
        axes[idx].set_xticklabels([m.replace('_', ' ').title() for m in models], 
                                   rotation=45, ha='right')
        axes[idx].set_ylabel('Score')
        axes[idx].set_title(f'{metric.replace("_", " ").title()}')
        axes[idx].set_ylim(0, 1.1)
        axes[idx].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(values):
            axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
    
    plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Model comparison saved to: {save_path}")
    plt.close()


def plot_feature_importance(model, feature_names, top_n=20, save_path='feature_importance.png'):
    """
    Plot feature importance for models that support it.
    
    Args:
        model: Trained model
        feature_names: Names of features
        top_n: Number of top features to show
        save_path: Path to save plot
    """
    if hasattr(model, 'coef_'):
        # For linear models
        importance = np.abs(model.coef_[0])
    elif hasattr(model, 'feature_importances_'):
        # For tree-based models
        importance = model.feature_importances_
    else:
        print("Model doesn't support feature importance")
        return
    
    # Get top features
    indices = np.argsort(importance)[-top_n:]
    top_features = [feature_names[i] for i in indices]
    top_importance = importance[indices]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), top_importance, color='lightcoral', edgecolor='darkred')
    plt.yticks(range(top_n), top_features)
    plt.xlabel('Importance Score')
    plt.title(f'Top {top_n} Most Important Features')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Feature importance saved to: {save_path}")
    plt.close()


def visualize_predictions(data_path='data/sample_news.csv', model_type='logistic_regression'):
    """
    Create comprehensive visualizations for model performance.
    
    Args:
        data_path: Path to data
        model_type: Type of model to train
    """
    print("=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)
    
    # Prepare data
    print("\n1. Loading and preparing data...")
    if not os.path.exists(data_path):
        create_sample_dataset(data_path)
    
    loader = DataLoader()
    texts, labels = loader.load_csv(data_path)
    X_train_raw, X_test_raw, y_train, y_test = loader.split_data(texts, labels)
    
    # Preprocess
    print("2. Preprocessing...")
    preprocessor = TextPreprocessor()
    X_train_clean = [preprocessor.preprocess(text) for text in X_train_raw]
    X_test_clean = [preprocessor.preprocess(text) for text in X_test_raw]
    
    # Extract features
    print("3. Extracting features...")
    feature_extractor = FeatureExtractor(max_features=1000)
    X_train = feature_extractor.fit_transform(X_train_clean)
    X_test = feature_extractor.transform(X_test_clean)
    
    # Train model
    print(f"4. Training {model_type} model...")
    detector = FakeNewsDetector(model_type)
    detector.train(X_train, y_train)
    
    # Make predictions
    print("5. Making predictions...")
    y_pred = detector.predict(X_test)
    
    # Create visualizations directory
    viz_dir = 'visualizations'
    os.makedirs(viz_dir, exist_ok=True)
    
    # Plot confusion matrix
    print("6. Creating confusion matrix...")
    plot_confusion_matrix(y_test, y_pred, 
                         os.path.join(viz_dir, f'{model_type}_confusion_matrix.png'))
    
    # Plot feature importance
    print("7. Creating feature importance plot...")
    if hasattr(detector.model, 'coef_') or hasattr(detector.model, 'feature_importances_'):
        feature_names = feature_extractor.vectorizer.get_feature_names_out()
        plot_feature_importance(detector.model, feature_names, 
                              save_path=os.path.join(viz_dir, f'{model_type}_feature_importance.png'))
    
    # Print classification report
    print("\n8. Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
    
    print("\n" + "=" * 60)
    print(f"Visualizations saved in '{viz_dir}/' directory")
    print("=" * 60)


def compare_and_visualize(data_path='data/sample_news.csv'):
    """
    Compare multiple models and create comparison visualization.
    
    Args:
        data_path: Path to data
    """
    print("=" * 60)
    print("COMPARING MODELS AND CREATING VISUALIZATIONS")
    print("=" * 60)
    
    # Prepare data
    print("\nPreparing data...")
    if not os.path.exists(data_path):
        create_sample_dataset(data_path)
    
    loader = DataLoader()
    texts, labels = loader.load_csv(data_path)
    X_train_raw, X_test_raw, y_train, y_test = loader.split_data(texts, labels)
    
    preprocessor = TextPreprocessor()
    X_train_clean = [preprocessor.preprocess(text) for text in X_train_raw]
    X_test_clean = [preprocessor.preprocess(text) for text in X_test_raw]
    
    feature_extractor = FeatureExtractor(max_features=1000)
    X_train = feature_extractor.fit_transform(X_train_clean)
    X_test = feature_extractor.transform(X_test_clean)
    
    # Train and evaluate multiple models
    model_types = ['logistic_regression', 'naive_bayes', 'random_forest', 'svm']
    results = {}
    
    for model_type in model_types:
        print(f"\nTraining {model_type}...")
        detector = FakeNewsDetector(model_type)
        detector.train(X_train, y_train)
        metrics = detector.evaluate(X_test, y_test)
        results[model_type] = metrics
    
    # Create visualization
    viz_dir = 'visualizations'
    os.makedirs(viz_dir, exist_ok=True)
    
    print("\nCreating comparison plot...")
    plot_model_comparison(results, os.path.join(viz_dir, 'model_comparison.png'))
    
    print("\n" + "=" * 60)
    print(f"Visualizations saved in '{viz_dir}/' directory")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize model performance')
    parser.add_argument('--data', type=str, default='data/sample_news.csv',
                       help='Path to data')
    parser.add_argument('--model', type=str, default='logistic_regression',
                       help='Model type')
    parser.add_argument('--compare', action='store_true',
                       help='Compare all models')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_and_visualize(args.data)
    else:
        visualize_predictions(args.data, args.model)
