# tests/test_inference.py
import sys
import os
from pathlib import Path

import pandas as pd
import pytest

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.inference_pipeline.inference import predict


@pytest.fixture(scope="session")
def sample_df():
    """Load a small sample from cleaning_eval.csv for inference testing."""
    sample_path = ROOT / "data/processed/feature_engineered_eval.csv"
    df = pd.read_csv(sample_path).sample(5, random_state=42).reset_index(drop=True)
    return df


def test_inference_runs_and_returns_predictions(sample_df):
    """Ensure inference pipeline runs and returns predicted_price column."""
    preds_df = predict(sample_df)

    # Check output is not empty
    assert not preds_df.empty

    # Must include prediction column
    assert "predicted_price" in preds_df.columns

    # Predictions should be numeric
    assert pd.api.types.is_numeric_dtype(preds_df["predicted_price"])

    print("✅ Inference pipeline test passed. Predictions:")
    print(preds_df[["predicted_price"]].head())
