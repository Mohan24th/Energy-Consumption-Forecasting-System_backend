# app/model_loader.py
import joblib
from pathlib import Path
import pandas as pd
from config import MODEL_PATH, FEATURES
import os

MODEL = None

def load_model():
    """Load the trained model. Called at startup."""
    global MODEL
    try:
        if MODEL_PATH.exists():
            MODEL = joblib.load(MODEL_PATH)
            print(f"✅ Model loaded successfully from {MODEL_PATH}")
            print(f"✅ Model expects {MODEL.n_features_in_} features")
            # Quick test
            test_input = pd.DataFrame([[28, 55, 750, 4, 1, 1, 2, 0, 14, 15, 6, 0, 1]],
                                      columns=FEATURES)
            test_pred = MODEL.predict(test_input)[0]
            print(f"✅ Test prediction successful")
        else:
            print(f"❌ Model file not found at {MODEL_PATH}")
            print(f"   Current working directory: {os.getcwd()}")
            print("   Please check the path and run train_model.py first")
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")

def get_model():
    """Return the loaded model (may be None if loading failed)."""
    return MODEL