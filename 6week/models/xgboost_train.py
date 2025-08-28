#!/usr/bin/env python3
"""
XGBoost Training Script
Trains an XGBoost model with early stopping and hyperparameter tuning for classification.
"""

import argparse
import os
import pandas as pd
import numpy as np
import warnings
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import json
from datetime import datetime
import itertools

warnings.filterwarnings('ignore')

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train XGBoost model for classification')
    parser.add_argument('--data-path', type=str, required=True, 
                       help='Path to CSV file containing the dataset')
    parser.add_argument('--target-col', type=str, required=True,
                       help='Name of the target/label column')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save trained model and metrics')
    
    # Hyperparameters
    parser.add_argument('--n-estimators', type=int, default=100,
                       help='Number of gradient boosted trees (default: 100)')
    parser.add_argument('--learning-rate', type=float, default=0.1,
                       help='Boosting learning rate (default: 0.1)')
    parser.add_argument('--max-depth', type=int, default=6,
                       help='Maximum tree depth for base learners (default: 6)')
    parser.add_argument('--subsample', type=float, default=1.0,
                       help='Subsample ratio of the training instances (default: 1.0)')
    parser.add_argument('--colsample-bytree', type=float, default=1.0,
                       help='Subsample ratio of columns when constructing each tree (default: 1.0)')
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

def preprocess_features(X_train, X_val, X_test, numeric_cols, text_cols):
    """
    Preprocess numeric and text features for XGBoost
    Returns: processed train, val, test sets
    """
    print("[PREPROCESSING] Preprocessing features...")
    
    processed_features = []
    feature_names = []
    
    # Process numeric columns
    if numeric_cols:
        print(f"Processing {len(numeric_cols)} numeric columns...")
        
        # Impute missing values
        numeric_imputer = SimpleImputer(strategy='mean')
        X_train_numeric = numeric_imputer.fit_transform(X_train[numeric_cols])
        X_val_numeric = numeric_imputer.transform(X_val[numeric_cols])
        X_test_numeric = numeric_imputer.transform(X_test[numeric_cols])
        
        # Standard scaling
        scaler = StandardScaler()
        X_train_numeric = scaler.fit_transform(X_train_numeric)
        X_val_numeric = scaler.transform(X_val_numeric)
        X_test_numeric = scaler.transform(X_test_numeric)
        
        processed_features.append((X_train_numeric, X_val_numeric, X_test_numeric))
        feature_names.extend([f"numeric_{col}" for col in numeric_cols])
        
        print(f"[SUCCESS] Processed numeric features shape: {X_train_numeric.shape}")
    
    # Process text columns
    if text_cols:
        print(f"Processing {len(text_cols)} text columns...")
        
        # Combine text columns
        def combine_text(df):
            text_data = df[text_cols].fillna('').apply(
                lambda x: ' '.join(x.astype(str)), axis=1
            )
            return text_data
        
        train_text = combine_text(X_train)
        val_text = combine_text(X_val)
        test_text = combine_text(X_test)
        
        # TF-IDF vectorization
        tfidf = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        X_train_text = tfidf.fit_transform(train_text).toarray()
        X_val_text = tfidf.transform(val_text).toarray()
        X_test_text = tfidf.transform(test_text).toarray()
        
        processed_features.append((X_train_text, X_val_text, X_test_text))
        text_feature_names = tfidf.get_feature_names_out()
        feature_names.extend([f"text_{name}" for name in text_feature_names])
        
        print(f"[SUCCESS] Processed text features shape: {X_train_text.shape}")
    
    # Combine all features
    if len(processed_features) > 1:
        X_train_processed = np.hstack([feat[0] for feat in processed_features])
        X_val_processed = np.hstack([feat[1] for feat in processed_features])
        X_test_processed = np.hstack([feat[2] for feat in processed_features])
    else:
        X_train_processed, X_val_processed, X_test_processed = processed_features[0]
    
    print(f"[SUCCESS] Final processed features shape: {X_train_processed.shape}")
    
    return (X_train_processed, X_val_processed, X_test_processed, 
            feature_names, tfidf if text_cols else None, 
            numeric_imputer if numeric_cols else None,
            scaler if numeric_cols else None)

def prepare_labels(y_train, y_val, y_test):
    """
    Prepare labels for XGBoost (encode if necessary)
    """
    # Check if labels are already numeric
    if y_train.dtype == 'object':
        print("[LABEL] Encoding string labels to numeric...")
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_val_encoded = label_encoder.transform(y_val)
        y_test_encoded = label_encoder.transform(y_test)
        
        print(f"Label mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
        return y_train_encoded, y_val_encoded, y_test_encoded, label_encoder
    else:
        return y_train.values, y_val.values, y_test.values, None

def hyperparameter_tuning(dtrain, dval, num_classes):
    """
    Perform hyperparameter tuning for XGBoost
    """
    print("[TUNING] Performing hyperparameter tuning...")
    
    # Define parameter grid
    param_grid = {
        'eta': [0.01, 0.1, 0.3],
        'max_depth': [3, 6, 9],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    best_score = 0
    best_params = None
    best_num_rounds = None
    
    # Grid search
    param_combinations = [dict(zip(param_grid.keys(), v)) 
                         for v in itertools.product(*param_grid.values())]
    
    print(f"Testing {len(param_combinations)} parameter combinations...")
    
    for i, params in enumerate(param_combinations[:12]):  # Limit to 12 combinations for speed
        print(f"Testing combination {i+1}/12: {params}")
        
        # Set base parameters
        xgb_params = {
            'objective': 'multi:softprob' if num_classes > 2 else 'binary:logistic',
            'num_class': num_classes if num_classes > 2 else None,
            'eval_metric': 'mlogloss' if num_classes > 2 else 'logloss',
            'seed': 42,
            'verbosity': 0
        }
        xgb_params.update(params)
        
        # Remove None values
        xgb_params = {k: v for k, v in xgb_params.items() if v is not None}
        
        # Train with early stopping
        evals = [(dtrain, 'train'), (dval, 'val')]
        bst = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=1000,
            evals=evals,
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        # Get best score
        best_iteration = bst.best_iteration
        val_score = bst.best_score
        
        if val_score > best_score or (num_classes <= 2 and val_score < best_score):
            # For binary classification, lower logloss is better
            if num_classes > 2:
                is_better = val_score > best_score
            else:
                is_better = val_score < best_score if best_score > 0 else True
            
            if is_better:
                best_score = val_score
                best_params = xgb_params.copy()
                best_num_rounds = best_iteration
        
    print(f"[SUCCESS] Best parameters: {best_params}")
    print(f"[SUCCESS] Best validation score: {best_score:.4f}")
    print(f"[SUCCESS] Best number of rounds: {best_num_rounds}")
    
    return best_params, best_num_rounds

def train_xgboost(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Train XGBoost model with hyperparameter tuning and early stopping
    """
    print("[STARTING] Starting XGBoost training...")
    
    # Prepare labels
    y_train_enc, y_val_enc, y_test_enc, label_encoder = prepare_labels(y_train, y_val, y_test)
    
    # Create DMatrix objects
    dtrain = xgb.DMatrix(X_train, label=y_train_enc)
    dval = xgb.DMatrix(X_val, label=y_val_enc)
    dtest = xgb.DMatrix(X_test, label=y_test_enc)
    
    # Determine number of classes
    num_classes = len(np.unique(y_train_enc))
    print(f"Number of classes: {num_classes}")
    
    # Hyperparameter tuning
    best_params, best_num_rounds = hyperparameter_tuning(dtrain, dval, num_classes)
    
    # Train final model with best parameters
    print("[TARGET] Training final model with best parameters...")
    evals = [(dtrain, 'train'), (dval, 'val')]
    
    final_model = xgb.train(
        best_params,
        dtrain,
        num_boost_round=best_num_rounds,
        evals=evals,
        verbose_eval=False
    )
    
    return final_model, best_params, label_encoder, dtest, y_test_enc

def evaluate_model(model, dtest, y_test_enc, label_encoder):
    """
    Evaluate the trained XGBoost model
    """
    print("[EVALUATION] Evaluating model on test set...")
    
    # Make predictions
    y_pred_proba = model.predict(dtest)
    
    # Convert probabilities to class predictions
    if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
        # Multi-class classification
        y_pred_enc = np.argmax(y_pred_proba, axis=1)
    else:
        # Binary classification
        y_pred_enc = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_enc, y_pred_enc)
    f1 = f1_score(y_test_enc, y_pred_enc, average='weighted')
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1-Score: {f1:.4f}")
    
    # Convert back to original labels if necessary
    if label_encoder:
        y_test_orig = label_encoder.inverse_transform(y_test_enc)
        y_pred_orig = label_encoder.inverse_transform(y_pred_enc)
        
        print("\nClassification Report:")
        print(classification_report(y_test_orig, y_pred_orig))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test_orig, y_pred_orig))
        
        report_dict = classification_report(y_test_orig, y_pred_orig, output_dict=True)
    else:
        print("\nClassification Report:")
        print(classification_report(y_test_enc, y_pred_enc))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test_enc, y_pred_enc))
        
        report_dict = classification_report(y_test_enc, y_pred_enc, output_dict=True)
    
    # Feature importance
    print("\n[TARGET] Top 10 Most Important Features:")
    importance_dict = model.get_score(importance_type='weight')
    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    for i, (feature, importance) in enumerate(sorted_importance[:10]):
        print(f"{i+1:2d}. {feature}: {importance}")
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'classification_report': report_dict,
        'feature_importance': importance_dict
    }

def save_model_and_metrics(model, metrics, best_params, output_dir, preprocessors=None):
    """
    Save the trained XGBoost model and evaluation metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the model
    model_path = os.path.join(output_dir, 'xgboost_model.model')
    model.save_model(model_path)
    print(f"[SAVED] XGBoost model saved to: {model_path}")
    
    # Save preprocessors if available
    if preprocessors:
        import joblib
        preprocessor_path = os.path.join(output_dir, 'preprocessors.joblib')
        joblib.dump(preprocessors, preprocessor_path)
        print(f"[CONFIG] Preprocessors saved to: {preprocessor_path}")
    
    # Save metrics and parameters
    results = {
        'model_type': 'XGBoost',
        'timestamp': datetime.now().isoformat(),
        'best_parameters': best_params,
        'test_metrics': metrics
    }
    
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[METRICS] Metrics saved to: {metrics_path}")

def main():
    """Main training pipeline"""
    print("[STARTING] XGBoost Training Pipeline")
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
    
    # Preprocess features
    (X_train_processed, X_val_processed, X_test_processed, 
     feature_names, tfidf, imputer, scaler) = preprocess_features(
        X_train, X_val, X_test, numeric_cols, text_cols
    )
    
    # Train model
    model, best_params, label_encoder, dtest, y_test_enc = train_xgboost(
        X_train_processed, X_val_processed, X_test_processed, 
        y_train, y_val, y_test
    )
    
    # Evaluate model
    metrics = evaluate_model(model, dtest, y_test_enc, label_encoder)
    
    # Prepare preprocessors for saving
    preprocessors = {
        'tfidf': tfidf,
        'imputer': imputer,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'numeric_cols': numeric_cols,
        'text_cols': text_cols
    }
    
    # Save model and metrics
    save_model_and_metrics(model, metrics, best_params, args.output_dir, preprocessors)
    
    print("=" * 50)
    print(f"[SUCCESS] Model and metrics saved to {args.output_dir}")
    print("[COMPLETE] Training completed successfully!")

if __name__ == "__main__":
    main()
