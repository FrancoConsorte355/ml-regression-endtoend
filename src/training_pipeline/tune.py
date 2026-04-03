"""
Hyperparameter tuning with Optuna + MLflow.

- Optimizes XGB params on eval set RMSE.
- Logs trials to MLflow.
- Retrains best model and saves to `model_output`.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from joblib import dump
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

import mlflow
import mlflow.xgboost

DEFAULT_TRAIN = Path("data/processed/feature_engineered_train.csv")
DEFAULT_EVAL = Path("data/processed/feature_engineered_eval.csv")
DEFAULT_OUT = Path("models/xgb_best_model.pkl")


def _maybe_sample(df: pd.DataFrame, sample_frac: Optional[float], random_state: int) -> pd.DataFrame:
    if sample_frac is None:
        return df
    sample_frac = float(sample_frac)
    if sample_frac <= 0 or sample_frac >= 1:
        return df
    return df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)


def _load_data(
    train_path: Path | str,
    eval_path: Path | str,
    sample_frac: Optional[float],
    random_state: int,
):
    train_df = pd.read_csv(train_path)
    eval_df = pd.read_csv(eval_path)
    train_df = _maybe_sample(train_df, sample_frac, random_state)
    eval_df = _maybe_sample(eval_df, sample_frac, random_state)

    target = "price"
    X_train, y_train = train_df.drop(columns=[target]), train_df[target]
    X_eval, y_eval = eval_df.drop(columns=[target]), eval_df[target]
    return X_train, y_train, X_eval, y_eval


def tune_model(
    train_path: Path | str = DEFAULT_TRAIN,
    eval_path: Path | str = DEFAULT_EVAL,
    model_output: Path | str = DEFAULT_OUT,
    n_trials: int = 15,
    sample_frac: Optional[float] = None,
    tracking_uri: Optional[str] = None,
    experiment_name: str = "xgboost_optuna_housing",
    random_state: int = 42,
) -> Tuple[Dict, Dict]:
    """Run Optuna tuning; save best model; return (best_params, best_metrics)."""
    if tracking_uri:
        # MLflow expects a URI scheme. When users supply a bare filesystem path
        # (common in tests and local runs), convert it into a proper file://
        # URI. ``Path.as_uri()`` handles windows vs unix automatically.
        if "://" not in tracking_uri:
            try:
                tracking_uri = Path(tracking_uri).resolve().as_uri()
            except Exception:
                # fallback: keep original string if conversion fails
                pass
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    X_train, y_train, X_eval, y_eval = _load_data(train_path, eval_path, sample_frac, random_state)

    def objective(trial: optuna.Trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": random_state,
            "n_jobs": -1,
            "tree_method": "hist",
        }

        with mlflow.start_run(nested=True):
            model = XGBRegressor(**params)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_eval)
            rmse = float(np.sqrt(mean_squared_error(y_eval, y_pred)))
            mae = float(mean_absolute_error(y_eval, y_pred))
            r2 = float(r2_score(y_eval, y_pred))

            mlflow.log_params(params)
            mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})

        return rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_trial.params
    print("✅ Best params from Optuna:", best_params)

    # Retrain best model
    best_model = XGBRegressor(**{**best_params, "random_state": random_state, "n_jobs": -1, "tree_method": "hist"})
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_eval)
    best_metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_eval, y_pred))),
        "mae": float(mean_absolute_error(y_eval, y_pred)),
        "r2": float(r2_score(y_eval, y_pred)),
    }
    print("📊 Best tuned model metrics:", best_metrics)

    # Save to models/
    out = Path(model_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    dump(best_model, out)
    print(f"✅ Best model saved to {out}")

    # Log final best model to MLflow
    with mlflow.start_run(run_name="best_xgb_model"):
        mlflow.log_params(best_params)
        mlflow.log_metrics(best_metrics)
        mlflow.xgboost.log_model(best_model, "model")

    return best_params, best_metrics


if __name__ == "__main__":
    tune_model()
