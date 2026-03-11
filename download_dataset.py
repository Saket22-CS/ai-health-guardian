"""
download_dataset.py
Downloads the disease prediction dataset from Kaggle.
Run: python download_dataset.py
"""

import os
import subprocess
import sys
from pathlib import Path
import zipfile

def download_via_kaggle_api():
    """Download using Kaggle API (requires kaggle.json or env vars)."""
    print("📦 Method 1: Trying Kaggle API...")
    try:
        import kaggle
        from kaggle.api.kaggle_api_extended import KaggleApiExtended
        api = KaggleApiExtended()
        api.authenticate()
        
        os.makedirs("data", exist_ok=True)
        
        # Primary dataset: Disease Symptom Prediction
        print("   Downloading: itachi9604/disease-symptom-description-dataset...")
        api.dataset_download_files(
            "itachi9604/disease-symptom-description-dataset",
            path="data/",
            unzip=True
        )
        print("✅ Downloaded disease-symptom dataset!")
        
        # Also get the larger dataset
        print("   Downloading: kaushil268/disease-prediction-using-machine-learning...")
        api.dataset_download_files(
            "kaushil268/disease-prediction-using-machine-learning",
            path="data/",
            unzip=True
        )
        print("✅ Downloaded ML disease prediction dataset!")
        return True
        
    except Exception as e:
        print(f"   ❌ Kaggle API failed: {e}")
        return False


def download_manually():
    """Print manual download instructions."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║         MANUAL DATASET DOWNLOAD INSTRUCTIONS                 ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  DATASET 1 (Main - Disease Symptom):                         ║
║  https://www.kaggle.com/datasets/                            ║
║         itachi9604/disease-symptom-description-dataset       ║
║                                                              ║
║  Download → Extract → Copy 'dataset.csv' to data/ folder    ║
║                                                              ║
║  DATASET 2 (Training/Testing split):                         ║
║  https://www.kaggle.com/datasets/                            ║
║         kaushil268/disease-prediction-using-machine-learning ║
║                                                              ║
║  Download → Extract → Copy 'Training.csv' to data/ folder   ║
║                                                              ║
║  STEPS:                                                      ║
║  1. Go to the Kaggle URL above                               ║
║  2. Click "Download" button                                  ║
║  3. Extract the ZIP file                                     ║
║  4. Place the CSV file inside the 'data/' folder             ║
║  5. Run: python train_model.py                               ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)


def setup_kaggle_credentials():
    """Help user set up Kaggle API credentials."""
    print("""
📋 SETTING UP KAGGLE API:
  1. Go to https://www.kaggle.com → Your Profile → Settings
  2. Scroll to 'API' section → Click 'Create New Token'
  3. A file 'kaggle.json' will be downloaded
  4. Place it at:
     - Linux/Mac:  ~/.kaggle/kaggle.json
     - Windows:    C:\\Users\\<username>\\.kaggle\\kaggle.json
  5. Run this script again
  
  OR set environment variables:
     KAGGLE_USERNAME=your_username
     KAGGLE_KEY=your_api_key
    """)


def check_existing_data():
    """Check if dataset already exists."""
    possible_files = [
        "data/dataset.csv",
        "data/Training.csv",
        "Training.csv",
        "dataset.csv"
    ]
    for f in possible_files:
        if os.path.exists(f):
            import pandas as pd
            df = pd.read_csv(f)
            print(f"✅ Found existing dataset: {f}")
            print(f"   Shape: {df.shape}")
            return True
    return False


def create_sample_and_proceed():
    """If no dataset found, train_model.py will create a sample."""
    print("""
⚠️  No dataset found. The training script will create a built-in
   sample dataset with 20 diseases for demonstration.
   
   For the full dataset (130+ diseases, 400+ symptoms):
   → Follow the manual download instructions above
   → Then re-run: python train_model.py
    """)


if __name__ == "__main__":
    print("=" * 60)
    print("  AI Health Guardian — Dataset Downloader")
    print("=" * 60)
    
    # Check if data already exists
    if check_existing_data():
        print("\n✅ Dataset already present! Run: python train_model.py")
        sys.exit(0)
    
    os.makedirs("data", exist_ok=True)
    
    # Try Kaggle API first
    success = download_via_kaggle_api()
    
    if not success:
        setup_kaggle_credentials()
        download_manually()
        create_sample_and_proceed()
    
    print("\n▶  Next step: python train_model.py")
