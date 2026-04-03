# Goal: Create a FastAPI app to serve your trained ML model into a web service that anyone 
# (or any system) can call over HTTP.

from fastapi import FastAPI            # Web framework for APIs
from pathlib import Path               # For handling file paths cleanly
from typing import List, Dict, Any     # For type hints (clarity in endpoints)
import pandas as pd                    # To handle incoming JSON as DataFrames
import boto3, os                       # AWS SDK for Python + env variables

# Import inference pipeline
from src.inference_pipeline.inference import predict

# ----------------------------
# Config
# ----------------------------
S3_BUCKET = os.getenv("S3_BUCKET", "housing-regression-data-fc")
REGION = os.getenv("AWS_REGION", "us-east-1")
s3 = boto3.client("s3", region_name=REGION)

# Ensures your app always has the latest model/data locally, 
# but avoids re-downloading every time it starts.
def load_from_s3(key, local_path):
    """Download from S3 if not already cached locally."""
    local_path = Path(local_path)
    if not local_path.exists():
        os.makedirs(local_path.parent, exist_ok=True)
        print(f"📥 Downloading {key} from S3…")
        s3.download_file(S3_BUCKET, key, str(local_path))
    return str(local_path)

# ----------------------------
# Paths
# Este fragmento de código resuelve uno de los problemas más traicioneros y comunes al desplegar modelos en producción:
# La desalineación de características (Feature Misalignment).
# XGBoost es increíblemente estricto. Si entrenaste el modelo con 39 columnas, 
# la API debe enviarle exactamente 39 columnas, en el mismo orden exacto y con los mismos nombres. 
# Si un usuario envía un JSON con las columnas en distinto orden, 
# XGBoost podría usar el valor de latitud creyendo que es la cantidad_de_habitaciones, prediciendo un desastre sin arrojar error.
# ----------------------------
# Downloads model + training features from S3 if not cached.
MODEL_PATH = Path(load_from_s3("models/xgb_best_model.pkl", "models/xgb_best_model.pkl"))
TRAIN_FE_PATH = Path(load_from_s3("processed/feature_engineered_train.csv", "data/processed/feature_engineered_train.csv"))

# Load expected training features for alignment
if TRAIN_FE_PATH.exists():
    _train_cols = pd.read_csv(TRAIN_FE_PATH, nrows=1)
    TRAIN_FEATURE_COLUMNS = [c for c in _train_cols.columns if c != "price"]
else:
    TRAIN_FEATURE_COLUMNS = None

# ----------------------------
# App
# ----------------------------
# Instantiates the FastAPI app.
app = FastAPI(title="Housing Regression API")

# / → simple landing endpoint to confirm API is alive.
@app.get("/")
def root():
    return {"message": "Housing Regression API is running 🚀"}

# /health → checks if model exists, returns status info (like expected feature count).
@app.get("/health")
def health():
    status: Dict[str, Any] = {"model_path": str(MODEL_PATH)}
    if not MODEL_PATH.exists():
        status["status"] = "unhealthy"
        status["error"] = "Model not found"
    else:
        status["status"] = "healthy"
        if TRAIN_FEATURE_COLUMNS:
            status["n_features_expected"] = len(TRAIN_FEATURE_COLUMNS)
    return status

# Prediction Endpoint: This is the core ML serving endpoint.
@app.post("/predict")
def predict_batch(data: List[dict]):
    if not MODEL_PATH.exists():
        return {"error": f"Model not found at {str(MODEL_PATH)}"}

    df = pd.DataFrame(data)
    if df.empty:
        return {"error": "No data provided"}

    preds_df = predict(df, model_path=MODEL_PATH)

    resp = {"predictions": preds_df["predicted_price"].astype(float).tolist()}
    if "actual_price" in preds_df.columns:
        resp["actuals"] = preds_df["actual_price"].astype(float).tolist()

    return resp

# Batch runner
from src.batch.run_monthly import run_monthly_predictions

# Trigger a monthly batch job via API.
@app.post("/run_batch")
def run_batch():
    preds = run_monthly_predictions()
    return {
        "status": "success",
        "rows_predicted": int(len(preds)),
        "output_dir": "data/predictions/"
    }

# Returns a preview of the most recent batch predictions.
@app.get("/latest_predictions")
def latest_predictions(limit: int = 5):
    pred_dir = Path("data/predictions")
    files = sorted(pred_dir.glob("preds_*.csv"))
    if not files:
        return {"error": "No predictions found"}

    latest_file = files[-1]
    df = pd.read_csv(latest_file)
    return {
        "file": latest_file.name,
        "rows": int(len(df)),
        "preview": df.head(limit).to_dict(orient="records")
    }


"""
🔹 Execution Order / Module Flow

1. Imports (FastAPI, pandas, boto3, your inference function).
2. Config setup (env vars → bucket/region).
3. S3 utility (load_from_s3).
4. Download + load model/artifacts (MODEL_PATH, TRAIN_FE_PATH).
5. Infer schema (TRAIN_FEATURE_COLUMNS).
6. Create FastAPI app (app = FastAPI).
7. Declare endpoints (/, /health, /predict, /run_batch, /latest_predictions).
"""