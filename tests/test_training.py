import math
from pathlib import Path
from joblib import load

from src.training_pipeline.train import train_model
from src.training_pipeline.eval import evaluate_model
from src.training_pipeline.tune import tune_model


# Assumes you already ran feature engineering so the processed CSVs exist.
TRAIN_PATH = Path("data/processed/feature_engineered_train.csv")
EVAL_PATH = Path("data/processed/feature_engineered_eval.csv")

# Ensuring we have the same keys in metrics dict.
def _assert_metrics(m):
    assert set(m.keys()) == {"mae", "rmse", "r2"}
    assert all(isinstance(v, float) and math.isfinite(v) for v in m.values())


# TRAIN: Trains a quick model (with tiny sample + params to keep tests fast).
def test_train_creates_model_and_metrics(tmp_path):
    out_path = tmp_path / "xgb_model.pkl"
    # small params + sampling for speed
    _, metrics = train_model(
        train_path=TRAIN_PATH,
        eval_path=EVAL_PATH,
        model_output=out_path,
        model_params={"n_estimators": 20, "max_depth": 4, "learning_rate": 0.1},
        sample_frac=0.02,
    )
    assert out_path.exists()
    _assert_metrics(metrics)
    model = load(out_path)
    assert model is not None
    print("✅ train_model test passed")

# EVAL: Trains a model first, then evaluates it on eval set. Confirms evaluation metrics are valid.
def test_eval_works_with_saved_model(tmp_path):
    # train quick model
    model_path = tmp_path / "xgb_model.pkl"
    train_model(
        train_path=TRAIN_PATH,
        eval_path=EVAL_PATH,
        model_output=model_path,
        model_params={"n_estimators": 20},
        sample_frac=0.02,
    )
    metrics = evaluate_model(model_path=model_path, eval_path=EVAL_PATH, sample_frac=0.02)
    _assert_metrics(metrics)
    print("✅ evaluate_model test passed")

# TUNE: Runs tune_model with only 2 trials (fast for CI).
def test_tune_saves_best_model(tmp_path):
    model_out = tmp_path / "xgb_best.pkl"
    tracking_dir = tmp_path / "mlruns"
    # pass a URI rather than raw path to avoid platform-specific issues
    tracking_uri = tracking_dir.as_uri()
    best_params, best_metrics = tune_model(
        train_path=TRAIN_PATH,
        eval_path=EVAL_PATH,
        model_output=model_out,
        n_trials=2,
        sample_frac=0.02,
        tracking_uri=tracking_uri,
        experiment_name="test_xgb_optuna",
    )
    assert model_out.exists()
    assert isinstance(best_params, dict) and best_params
    _assert_metrics(best_metrics)
    print("✅ tune_model test passed")
