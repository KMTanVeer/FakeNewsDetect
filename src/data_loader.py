"""
Data loading utilities for fake news detection.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class DataLoader:
    """Loads and manages datasets for fake news detection."""
    
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def load_csv(self, filepath, text_column='text', label_column='label'):
        """
        Load data from CSV file.
        
        Args:
            filepath (str): Path to CSV file
            text_column (str): Name of text column
            label_column (str): Name of label column
            
        Returns:
            tuple: (texts, labels)
        """
        self.data = pd.read_csv(filepath)
        
        if text_column not in self.data.columns:
            raise ValueError(f"Column '{text_column}' not found in data")
        if label_column not in self.data.columns:
            raise ValueError(f"Column '{label_column}' not found in data")
        
        texts = self.data[text_column].fillna('').tolist()
        labels = self.data[label_column].tolist()
        
        return texts, labels
    
    def split_data(self, texts, labels, test_size=0.2, random_state=42):
        """
        Split data into train and test sets.
        
        Args:
            texts (list): List of text documents
            labels (list): List of labels
            test_size (float): Proportion of test set
            random_state (int): Random seed
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_data_info(self):
        """
        Get information about loaded data.
        
        Returns:
            dict: Data information
        """
        if self.data is None:
            return {"error": "No data loaded"}
        
        info = {
            'total_samples': len(self.data),
            'columns': list(self.data.columns),
            'missing_values': self.data.isnull().sum().to_dict()
        }
        
        if self.X_train is not None:
            info['train_size'] = len(self.X_train)
            info['test_size'] = len(self.X_test)
        
        return info


def create_sample_dataset(output_path='data/sample_news.csv', n_samples=1000):
    """
    Create a sample dataset for testing.
    
    Args:
        output_path (str): Path to save the dataset
        n_samples (int): Number of samples to generate
    """
    np.random.seed(42)
    
    # Sample fake news headlines/texts
    fake_samples = [
        "BREAKING: Scientists discover cure for all diseases overnight!",
        "SHOCKING: Celebrity caught in unbelievable scandal!",
        "You won't believe what happened next in this amazing story!",
        "URGENT: Government hiding the truth about major conspiracy!",
        "ALERT: New study proves everything you knew was wrong!",
        "EXCLUSIVE: Insider reveals shocking details about secret plan!",
        "CONFIRMED: Miracle product solves all problems instantly!",
        "WARNING: This one trick will change your life forever!",
        "BREAKING NEWS: Unbelievable event shocks the entire world!",
        "REVEALED: The truth they don't want you to know about!",
    ]
    
    # Sample real news headlines/texts
    real_samples = [
        "Local government announces new infrastructure development plan for city",
        "Research team publishes findings on climate change impact in peer-reviewed journal",
        "Economic indicators show moderate growth in third quarter according to official data",
        "University opens new research facility focused on renewable energy",
        "Health officials recommend continued preventive measures during flu season",
        "City council approves budget for upcoming fiscal year after public hearings",
        "Transportation department schedules maintenance work on major highway",
        "International conference brings together experts to discuss global issues",
        "Company reports quarterly earnings in line with market expectations",
        "Scientists present new findings at annual conference on environmental studies",
    ]
    
    texts = []
    labels = []
    
    # Generate samples
    for _ in range(n_samples // 2):
        texts.append(np.random.choice(fake_samples))
        labels.append(1)  # 1 for fake
    
    for _ in range(n_samples // 2):
        texts.append(np.random.choice(real_samples))
        labels.append(0)  # 0 for real
    
    # Create DataFrame
    df = pd.DataFrame({
        'text': texts,
        'label': labels
    })
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save
    df.to_csv(output_path, index=False)
    print(f"Sample dataset created with {n_samples} samples at {output_path}")
    
    return df
