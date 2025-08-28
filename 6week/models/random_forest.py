#!/usr/bin/env python3
"""
Random Forest Training Script
Trains a Random Forest model with mixed features (numeric + text via TF-IDF) for classification.
"""

import argparse
import os
import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import json
from datetime import datetime

warnings.filterwarnings('ignore')

class TextSelector(BaseEstimator, TransformerMixin):
    """Custom transformer to select text columns"""
    def __init__(self, text_cols):
        self.text_cols = text_cols
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if self.text_cols:
            # Concatenate all text columns with space separator
            text_data = X[self.text_cols].fillna('').apply(
                lambda x: ' '.join(x.astype(str)), axis=1
            )
            return text_data.values
        return np.array([''] * len(X))

class NumericSelector(BaseEstimator, TransformerMixin):
    """Custom transformer to select numeric columns"""
    def __init__(self, numeric_cols):
        self.numeric_cols = numeric_cols
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if self.numeric_cols:
            return X[self.numeric_cols].values
        return np.empty((len(X), 0))

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Random Forest model for classification')
    parser.add_argument('--data-path', type=str, required=True, 
                       help='Path to CSV file containing the dataset')
    parser.add_argument('--target-col', type=str, required=True,
                       help='Name of the target/label column')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save trained model and metrics')
    
    # Hyperparameters
    parser.add_argument('--n-estimators', type=int, default=100,
                       help='Number of trees in the forest (default: 100)')
    parser.add_argument('--max-depth', type=int, default=10,
                       help='Maximum depth of the trees (default: 10)')
    parser.add_argument('--min-samples-split', type=int, default=2,
                       help='Minimum samples required to split a node (default: 2)')
    parser.add_argument('--min-samples-leaf', type=int, default=1,
                       help='Minimum samples required at a leaf node (default: 1)')
    parser.add_argument('--max-features', type=int, default=10000,
                       help='Maximum number of TF-IDF features (default: 10000)')
    
    return parser.parse_args()

def load_and_split_data(data_path, target_col):
    """
    Load data from CSV and perform stratified train/val/test split
    Returns: X_train, X_val, X_test, y_train, y_val, y_test
    """
    print(f"[DATA] Loading data from {data_path}")
    
    # Check file extension and load accordingly
    if data_path.endswith('.json'):
        import json
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        # If it's an EDA summary JSON, this is an error
        if 'analysis_type' in data or 'results' in data:
            raise ValueError(
                "Received EDA summary JSON file instead of dataset CSV. "
                "Model training requires the original dataset, not EDA results."
            )
        
        # If it's actual data in JSON format, convert to DataFrame
        df = pd.DataFrame(data)
    else:
        # Assume CSV format
        df = pd.read_csv(data_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Target column: {target_col}")
    
    # Check if target column exists
    if target_col not in df.columns:
        # Try to suggest appropriate target columns based on common names
        potential_targets = []
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['target', 'label', 'class', 'y', 'outcome', 'result']):
                potential_targets.append(col)
        
        # For financial data, suggest creating a target from Close price
        if not potential_targets and 'Close' in df.columns:
            print(f"[INFO] Creating binary target from 'Close' price movements")
            # Create a simple binary target: 1 if Close > previous Close, 0 otherwise
            df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
            df = df.dropna()  # Remove last row with NaN
            target_col = 'target'
            print(f"[INFO] Created target column with {df['target'].sum()} positive cases out of {len(df)} total")
        elif potential_targets:
            # Use the first potential target found
            target_col = potential_targets[0]
            print(f"[INFO] Using '{target_col}' as target column")
        else:
            # If still no target found, raise error with suggestions
            raise ValueError(
                f"Target column '{target_col}' not found in dataset. "
                f"Available columns: {list(df.columns)}. "
                f"Please specify a valid target column or ensure your dataset has a 'target' column."
            )
    
    print(f"Target distribution:\n{df[target_col].value_counts()}")
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # First split: 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Second split: 15% val, 15% test from the 30% temp
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples") 
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def identify_column_types(X):
    """
    Identify numeric and text columns in the dataset
    Returns: numeric_cols, text_cols
    """
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    text_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    # Filter out columns that are actually categorical with few unique values
    actual_text_cols = []
    for col in text_cols:
        unique_ratio = X[col].nunique() / len(X)
        avg_length = X[col].astype(str).str.len().mean()
        if unique_ratio > 0.1 and avg_length > 10:  # Likely text columns
            actual_text_cols.append(col)
    
    print(f"Numeric columns: {numeric_cols}")
    print(f"Text columns: {actual_text_cols}")
    
    return numeric_cols, actual_text_cols

def create_feature_pipeline(numeric_cols, text_cols, max_features=10000):
    """
    Create feature pipeline using FeatureUnion for numeric and text features
    """
    feature_pipelines = []
    
    # Numeric feature pipeline
    if numeric_cols:
        numeric_pipeline = Pipeline([
            ('selector', NumericSelector(numeric_cols)),
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        feature_pipelines.append(('numeric', numeric_pipeline))
    
    # Text feature pipeline  
    if text_cols:
        text_pipeline = Pipeline([
            ('selector', TextSelector(text_cols)),
            ('tfidf', TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            ))
        ])
        feature_pipelines.append(('text', text_pipeline))
    
    if not feature_pipelines:
        raise ValueError("No valid feature pipelines could be created")
    
    # Combine features using FeatureUnion
    feature_union = FeatureUnion(feature_pipelines)
    return feature_union

def train_random_forest(X_train, X_val, y_train, y_val, numeric_cols, text_cols,
                       n_estimators=100, max_depth=10, min_samples_split=2, 
                       min_samples_leaf=1, max_features=10000):
    """
    Train Random Forest with hyperparameter tuning
    """
    print("[PIPELINE] Creating feature pipeline and Random Forest model...")
    
    # Create feature pipeline
    feature_pipeline = create_feature_pipeline(numeric_cols, text_cols, max_features)
    
    # Create full pipeline with Random Forest using provided hyperparameters
    pipeline = Pipeline([
        ('features', feature_pipeline),
        ('classifier', RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    # Hyperparameter grid for tuning (reduced grid around provided values)
    param_grid = {
        'classifier__n_estimators': [n_estimators],
        'classifier__max_depth': [max_depth, max_depth * 2] if max_depth else [None],
        'classifier__min_samples_split': [min_samples_split, min_samples_split * 2],
        'classifier__min_samples_leaf': [min_samples_leaf, min_samples_leaf * 2]
    }
    
    print("[TUNING] Performing hyperparameter tuning with GridSearchCV...")
    
    # GridSearchCV with stratified k-fold
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    
    # Combine train and validation for hyperparameter tuning
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])
    
    # Fit the grid search
    grid_search.fit(X_train_val, y_train_val)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Feature importance analysis
    best_rf = grid_search.best_estimator_.named_steps['classifier']
    feature_names = []
    
    # Add numeric feature names
    if numeric_cols:
        feature_names.extend(numeric_cols)
    
    # Add text feature names (top TF-IDF terms)
    if text_cols:
        tfidf_transformer = grid_search.best_estimator_.named_steps['features'].transformer_list
        for name, transformer in tfidf_transformer:
            if name == 'text':
                tfidf_vectorizer = transformer.named_steps['tfidf']
                if hasattr(tfidf_vectorizer, 'feature_names_out'):
                    text_features = tfidf_vectorizer.get_feature_names_out()
                    feature_names.extend([f"text_{feat}" for feat in text_features[:20]])  # Top 20
    
    print("\n Top 10 Most Important Features:")
    if len(feature_names) >= len(best_rf.feature_importances_):
        feature_importance = list(zip(feature_names[:len(best_rf.feature_importances_)], 
                                    best_rf.feature_importances_))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        for i, (feat, importance) in enumerate(feature_importance[:10]):
            print(f"{i+1:2d}. {feat}: {importance:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on test set
    """
    print(" Evaluating model on test set...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1-Score: {f1:.4f}")
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }

def save_model_and_metrics(model, metrics, best_params, output_dir):
    """
    Save the trained model and evaluation metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the model
    model_path = os.path.join(output_dir, 'random_forest_model.joblib')
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    # Save metrics and parameters
    results = {
        'model_type': 'Random Forest',
        'timestamp': datetime.now().isoformat(),
        'best_parameters': best_params,
        'test_metrics': metrics
    }
    
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")

def main():
    """Main training pipeline"""
    print("Starting Random Forest Training Pipeline")
    print("=" * 50)
    
    # Parse arguments
    args = parse_arguments()
    
    # Load and split data
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(
        args.data_path, args.target_col
    )
    
    # Identify column types
    numeric_cols, text_cols = identify_column_types(X_train)
    
    if not numeric_cols and not text_cols:
        raise ValueError("No suitable columns found for training")
    
    # Train model
    best_model, best_params = train_random_forest(
        X_train, X_val, y_train, y_val, numeric_cols, text_cols,
        n_estimators=args.n_estimators, max_depth=args.max_depth,
        min_samples_split=args.min_samples_split, min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features
    )
    
    # Evaluate model
    metrics = evaluate_model(best_model, X_test, y_test)
    
    # Save model and metrics
    save_model_and_metrics(best_model, metrics, best_params, args.output_dir)
    
    print("=" * 50)
    print(f" Model and metrics saved to {args.output_dir}")
    print(" Training completed successfully!")

if __name__ == "__main__":
    main()
