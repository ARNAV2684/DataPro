#!/usr/bin/env python3
"""
Model Training Runner
Execute individual model training scripts with proper error handling and logging.
"""

import subprocess
import sys
import os
import argparse
import json
from datetime import datetime

# Available models mapping
AVAILABLE_MODELS = {
    'logistic_regression': {
        'script': 'logistic_regression.py',
        'name': 'Logistic Regression',
        'description': 'Fast probabilistic classifier with mixed features'
    },
    'random_forest': {
        'script': 'random_forest.py', 
        'name': 'Random Forest',
        'description': 'Ensemble method with mixed numeric/text features'
    },
    'gradient_boosting': {
        'script': 'gradient_boosting.py',
        'name': 'Gradient Boosting',
        'description': 'Scikit-learn gradient boosting with mixed features'
    },
    'xgboost': {
        'script': 'xgboost_train.py',
        'name': 'XGBoost',
        'description': 'XGBoost gradient boosting with early stopping'
    },
    'distilbert': {
        'script': 'distilbert_finetune.py',
        'name': 'DistilBERT',
        'description': 'Transformer model fine-tuning'
    }
}

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run text classification model training')
    parser.add_argument('--model', type=str, required=True, 
                       choices=list(AVAILABLE_MODELS.keys()),
                       help='Model to train')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to CSV file containing the dataset')
    parser.add_argument('--target-col', type=str, required=True,
                       help='Name of the target/label column')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save trained model and metrics')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    return parser.parse_args()

def check_requirements():
    """Check if required packages are installed"""
    required_packages = {
        'logistic_regression': ['sklearn', 'pandas', 'numpy'],
        'random_forest': ['sklearn', 'pandas', 'numpy'],
        'gradient_boosting': ['sklearn', 'pandas', 'numpy'],
        'xgboost': ['xgboost', 'sklearn', 'pandas', 'numpy'],
        'distilbert': ['torch', 'transformers', 'datasets', 'sklearn']
    }
    
    print("🔍 Checking package requirements...")
    
    missing_packages = []
    for package in ['pandas', 'numpy', 'sklearn']:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {missing_packages}")
        print("💡 Install with: pip install -r requirements.txt")
        return False
    
    print("✅ Core packages available")
    return True

def run_model_training(model_key, data_path, target_col, output_dir, verbose=False):
    """
    Run the specified model training script
    """
    model_info = AVAILABLE_MODELS[model_key]
    script_path = os.path.join(os.path.dirname(__file__), model_info['script'])
    
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Training script not found: {script_path}")
    
    print(f"🚀 Starting {model_info['name']} training...")
    print(f"📝 {model_info['description']}")
    print(f"📂 Data: {data_path}")
    print(f"🎯 Target: {target_col}")
    print(f"💾 Output: {output_dir}")
    print("-" * 50)
    
    # Prepare command
    cmd = [
        sys.executable,
        script_path,
        '--data-path', data_path,
        '--target-col', target_col,
        '--output-dir', output_dir
    ]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Log training start
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'model': model_key,
        'model_name': model_info['name'],
        'data_path': data_path,
        'target_col': target_col,
        'output_dir': output_dir,
        'status': 'started'
    }
    
    log_path = os.path.join(output_dir, 'training_log.json')
    with open(log_path, 'w') as f:
        json.dump(log_entry, f, indent=2)
    
    try:
        # Run the training script
        if verbose:
            result = subprocess.run(cmd, check=True, text=True)
        else:
            result = subprocess.run(cmd, check=True, text=True, 
                                  capture_output=True)
            
        print("✅ Training completed successfully!")
        
        # Log success
        log_entry['status'] = 'completed'
        log_entry['end_timestamp'] = datetime.now().isoformat()
        
        with open(log_path, 'w') as f:
            json.dump(log_entry, f, indent=2)
            
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with error code: {e.returncode}")
        if hasattr(e, 'stdout') and e.stdout:
            print("STDOUT:", e.stdout)
        if hasattr(e, 'stderr') and e.stderr:
            print("STDERR:", e.stderr)
        
        # Log failure
        log_entry['status'] = 'failed'
        log_entry['error'] = str(e)
        log_entry['end_timestamp'] = datetime.now().isoformat()
        
        with open(log_path, 'w') as f:
            json.dump(log_entry, f, indent=2)
            
        return False
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        
        # Log failure
        log_entry['status'] = 'error'
        log_entry['error'] = str(e)
        log_entry['end_timestamp'] = datetime.now().isoformat()
        
        with open(log_path, 'w') as f:
            json.dump(log_entry, f, indent=2)
            
        return False

def list_available_models():
    """List all available models"""
    print("📋 Available Models:")
    print("=" * 50)
    for key, info in AVAILABLE_MODELS.items():
        print(f"🔸 {key}")
        print(f"   Name: {info['name']}")
        print(f"   Description: {info['description']}")
        print(f"   Script: {info['script']}")
        print()

def main():
    """Main execution function"""
    print("🤖 Text Classification Model Training Runner")
    print("=" * 50)
    
    # Parse arguments
    args = parse_arguments()
    
    # List available models if requested
    if args.model == 'list':
        list_available_models()
        return
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Validate inputs
    if not os.path.exists(args.data_path):
        print(f"❌ Data file not found: {args.data_path}")
        sys.exit(1)
    
    # Run training
    success = run_model_training(
        args.model, 
        args.data_path, 
        args.target_col, 
        args.output_dir,
        args.verbose
    )
    
    if success:
        print("🎉 Training pipeline completed successfully!")
        sys.exit(0)
    else:
        print("💥 Training pipeline failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
