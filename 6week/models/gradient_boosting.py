#!/usr/bin/env python3
"""
Gradient Boosting Training Script

Trains a Gradient Boosting model with early stopping and hyperparameter tuning for classification.
"""

import argparse
import os
import pandas as pd
import numpy as np
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib
import json
from datetime import datetime
import itertools


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Gradient Boosting model')
    
    # Required arguments - match the API calling convention
    parser.add_argument('--data-path', type=str, required=True, help='Path to the CSV data file')
    parser.add_argument('--target-col', type=str, required=True, help='Name of the target/label column')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save model and metrics')
    
    # Optional hyperparameters for Gradient Boosting
    parser.add_argument('--n-estimators', type=int, default=100, help='Number of boosting stages (default: 100)')
    parser.add_argument('--learning-rate', type=float, default=0.1, help='Learning rate (default: 0.1)')
    parser.add_argument('--max-depth', type=int, default=3, help='Maximum depth of trees (default: 3)')
    parser.add_argument('--min-samples-split', type=int, default=2, help='Minimum samples to split (default: 2)')
    parser.add_argument('--min-samples-leaf', type=int, default=1, help='Minimum samples per leaf (default: 1)')
    parser.add_argument('--subsample', type=float, default=1.0, help='Subsample ratio (default: 1.0)')
    
    # Validation strategy
    parser.add_argument('--validation-split', type=float, default=0.2, help='Validation split ratio (default: 0.2)')
    parser.add_argument('--random-state', type=int, default=42, help='Random state for reproducibility (default: 42)')
    
    return parser.parse_args()


def create_target_column(df):
    """Create target column for classification based on available data."""
    if 'target' in df.columns:
        return df, 'target'
    
    # For financial data, create target based on profit/loss
    if 'profit_loss' in df.columns:
        df['target'] = (df['profit_loss'] > 0).astype(int)
        return df, 'target'
    
    # For stock/financial data, look for price change indicators
    price_columns = [col for col in df.columns if any(word in col.lower() for word in ['price', 'close', 'value', 'amount'])]
    if price_columns:
        # Use the first price column to create a binary target
        price_col = price_columns[0]
        df['target'] = (df[price_col] > df[price_col].median()).astype(int)
        return df, 'target'
    
    # As fallback, use the last numeric column
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        target_col = numeric_cols[-1]
        df['target'] = (df[target_col] > df[target_col].median()).astype(int)
        return df, 'target'
    
    raise ValueError("No suitable column found for creating target variable")


def load_and_preprocess_data(data_path, target_col=None):
    """Load and preprocess the data."""
    print("[LOADING] Loading data from:", data_path)
    df = pd.read_csv(data_path)
    print(f"[INFO] Data shape: {df.shape}")
    
    # Use provided target column or create one if needed
    if target_col and target_col in df.columns:
        print(f"[INFO] Using provided target column: {target_col}")
    else:
        # Create target column if needed
        df, target_col = create_target_column(df)
        print(f"[INFO] Created target column: {target_col}")
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Identify numeric and text columns
    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    text_columns = X.select_dtypes(include=['object']).columns.tolist()
    
    print(f"[INFO] Numeric columns: {len(numeric_columns)}")
    print(f"[INFO] Text columns: {len(text_columns)}")
    
    # Handle numeric features
    if numeric_columns:
        # Impute missing values
        numeric_imputer = SimpleImputer(strategy='median')
        X_numeric = pd.DataFrame(
            numeric_imputer.fit_transform(X[numeric_columns]),
            columns=numeric_columns,
            index=X.index
        )
        
        # Scale numeric features
        scaler = StandardScaler()
        X_numeric_scaled = pd.DataFrame(
            scaler.fit_transform(X_numeric),
            columns=numeric_columns,
            index=X.index
        )
    else:
        X_numeric_scaled = pd.DataFrame(index=X.index)
        numeric_imputer = None
        scaler = None
    
    # Handle text features
    if text_columns:
        # Combine all text columns
        X_text_combined = X[text_columns].fillna('').agg(' '.join, axis=1)
        
        # Vectorize text
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        X_text_vectorized = vectorizer.fit_transform(X_text_combined)
        X_text_df = pd.DataFrame(
            X_text_vectorized.toarray(),
            columns=[f'text_feature_{i}' for i in range(X_text_vectorized.shape[1])],
            index=X.index
        )
    else:
        X_text_df = pd.DataFrame(index=X.index)
        vectorizer = None
    
    # Combine features
    X_processed = pd.concat([X_numeric_scaled, X_text_df], axis=1)
    
    print(f"[INFO] Final feature shape: {X_processed.shape}")
    
    return X_processed, y, {
        'numeric_imputer': numeric_imputer,
        'scaler': scaler,
        'vectorizer': vectorizer,
        'numeric_columns': numeric_columns,
        'text_columns': text_columns
    }


def train_gradient_boosting(X_train, X_val, y_train, y_val, args):
    """Train Gradient Boosting model with validation."""
    print("[TRAINING] Training Gradient Boosting model...")
    
    # Initialize model with hyperparameters (argparse converts hyphens to underscores)
    model = GradientBoostingClassifier(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        subsample=args.subsample,
        random_state=args.random_state,
        validation_fraction=0.1,
        n_iter_no_change=10,
        tol=1e-4
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average='weighted')
    
    print(f"[VALIDATION] Accuracy: {val_accuracy:.4f}")
    print(f"[VALIDATION] F1-score: {val_f1:.4f}")
    
    # Feature importance
    feature_importance = model.feature_importances_
    
    return model, {
        'validation_accuracy': val_accuracy,
        'validation_f1': val_f1,
        'feature_importance': feature_importance.tolist(),
        'n_estimators_used': model.n_estimators_
    }


def save_model_and_metrics(model, preprocessors, metrics, args, feature_names):
    """Save the trained model and metrics."""
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(args.output_dir, 'gradient_boosting_model.pkl')
    joblib.dump(model, model_path)
    print(f"[SAVE] Model saved to: {model_path}")
    
    # Save preprocessors
    if preprocessors['numeric_imputer'] is not None or preprocessors['scaler'] is not None or preprocessors['vectorizer'] is not None:
        preprocessor_path = os.path.join(args.output_dir, 'preprocessors.pkl')
        joblib.dump(preprocessors, preprocessor_path)
        print(f"[CONFIG] Preprocessors saved to: {preprocessor_path}")
    
    # Prepare comprehensive metrics
    detailed_metrics = {
        'model_type': 'gradient_boosting',
        'timestamp': datetime.now().isoformat(),
        'hyperparameters': {
            'n_estimators': args.n_estimators,
            'learning_rate': args.learning_rate,
            'max_depth': args.max_depth,
            'min_samples_split': args.min_samples_split,
            'min_samples_leaf': args.min_samples_leaf,
            'subsample': args.subsample,
            'random_state': args.random_state
        },
        'performance': {
            'validation_accuracy': metrics['validation_accuracy'],
            'validation_f1': metrics['validation_f1'],
            'n_estimators_used': metrics['n_estimators_used']
        },
        'feature_info': {
            'n_features': len(feature_names),
            'feature_names': feature_names[:50],  # Limit for JSON size
        }
    }
    
    # Add feature importance for top features
    if 'feature_importance' in metrics:
        importance_pairs = list(zip(feature_names, metrics['feature_importance']))
        importance_pairs.sort(key=lambda x: x[1], reverse=True)
        detailed_metrics['top_features'] = importance_pairs[:10]
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(detailed_metrics, f, indent=2)
    print(f"[METRICS] Metrics saved to: {metrics_path}")


def main():
    """Main training function."""
    print("[STARTING] Gradient Boosting Training Script")
    
    # Parse arguments
    args = parse_args()
    
    # Load and preprocess data (argparse converts hyphens to underscores)
    X, y, preprocessors = load_and_preprocess_data(args.data_path, args.target_col)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=args.validation_split, 
        random_state=args.random_state,
        stratify=y
    )
    
    print(f"[INFO] Training set: {X_train.shape}")
    print(f"[INFO] Validation set: {X_val.shape}")
    
    # Train model
    model, metrics = train_gradient_boosting(X_train, X_val, y_train, y_val, args)
    
    # Display feature importance
    if hasattr(model, 'feature_importances_'):
        feature_names = X.columns.tolist()
        importance_pairs = list(zip(feature_names, model.feature_importances_))
        importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        print("\n[TARGET] Top 10 Most Important Features:")
        for i, (feature, importance) in enumerate(importance_pairs[:10]):
            print(f"{i+1}. {feature}: {importance:.4f}")
    
    # Save everything
    save_model_and_metrics(model, preprocessors, metrics, args, X.columns.tolist())
    
    print(f"[SUCCESS] Model and metrics saved to {args.output_dir}")
    print("[COMPLETE] Training completed successfully!")


if __name__ == "__main__":
    main()
