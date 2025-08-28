#!/usr/bin/env python3
"""
Logistic Regression Training Script
Trains a logistic regression model with TF-IDF features for text classification.
"""

import argparse
import os
import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
import json
from datetime import datetime

warnings.filterwarnings('ignore')

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Logistic Regression model for text classification')
    parser.add_argument('--data-path', type=str, required=True, 
                       help='Path to CSV file containing the dataset')
    parser.add_argument('--target-col', type=str, required=True,
                       help='Name of the target/label column')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save trained model and metrics')
    
    # Hyperparameters
    parser.add_argument('--C', type=float, default=1.0,
                       help='Regularization parameter (default: 1.0)')
    parser.add_argument('--penalty', type=str, default='l2', choices=['l1', 'l2', 'elasticnet', 'none'],
                       help='Penalty norm (default: l2)')
    parser.add_argument('--max-iter', type=int, default=1000,
                       help='Maximum number of iterations (default: 1000)')
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

def create_preprocessing_pipeline(numeric_cols, text_cols, max_features=10000):
    """
    Create preprocessing pipeline for numeric and text features
    """
    transformers = []
    
    # Numeric preprocessing: impute + scale
    if numeric_cols:
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('numeric', numeric_transformer, numeric_cols))
    
    # Text preprocessing: TF-IDF vectorization
    if text_cols:
        # Combine all text columns into one for TF-IDF
        text_transformer = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            ))
        ])
        # For multiple text columns, we'll concatenate them
        transformers.append(('text', text_transformer, text_cols[0]))  # Use first text column
    
    if not transformers:
        raise ValueError("No valid columns found for preprocessing")
    
    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    return preprocessor

def train_logistic_regression(X_train, X_val, y_train, y_val, numeric_cols, text_cols, 
                            C=1.0, penalty='l2', max_iter=1000, max_features=10000):
    """
    Train logistic regression with hyperparameter tuning
    """
    print("[PIPELINE] Creating preprocessing and training pipeline...")
    
    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(numeric_cols, text_cols, max_features)
    
    # Create full pipeline with logistic regression using provided hyperparameters
    solver = 'liblinear' if penalty in ['l1', 'l2'] else 'lbfgs'
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            C=C, 
            penalty=penalty, 
            max_iter=max_iter, 
            random_state=42,
            solver=solver
        ))
    ])
    
    # Hyperparameter grid for tuning (reduced grid around provided values)
    param_grid = {
        'classifier__C': [C * 0.1, C, C * 10],
        'classifier__penalty': [penalty],
        'classifier__solver': [solver]
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
    
    print(f"[SUCCESS] Best parameters: {grid_search.best_params_}")
    print(f"[SUCCESS] Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on test set
    """
    print("[EVALUATION] Evaluating model on test set...")
    
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
    model_path = os.path.join(output_dir, 'logistic_regression_model.joblib')
    joblib.dump(model, model_path)
    print(f"[SAVED] Model saved to: {model_path}")
    
    # Save metrics and parameters
    results = {
        'model_type': 'Logistic Regression',
        'timestamp': datetime.now().isoformat(),
        'best_parameters': best_params,
        'test_metrics': metrics
    }
    
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[SAVED] Metrics saved to: {metrics_path}")

def main():
    """Main training pipeline"""
    print("[STARTING] Logistic Regression Training Pipeline")
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
    best_model, best_params = train_logistic_regression(
        X_train, X_val, y_train, y_val, numeric_cols, text_cols,
        C=args.C, penalty=args.penalty, max_iter=args.max_iter, max_features=args.max_features
    )
    
    # Evaluate model
    metrics = evaluate_model(best_model, X_test, y_test)
    
    # Save model and metrics
    save_model_and_metrics(best_model, metrics, best_params, args.output_dir)
    
    print("=" * 50)
    print(f"[SUCCESS] Model and metrics saved to {args.output_dir}")
    print("[COMPLETED] Training completed successfully!")

if __name__ == "__main__":
    main()
