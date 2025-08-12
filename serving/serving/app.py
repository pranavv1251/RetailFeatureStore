from __future__ import annotations
from pathlib import Path
from typing import Optional
import time
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .feature_service import FeatureService, row_to_X
from .pred_logger import log_prediction
import os
from .feature_service import FeatureService, RedisFeatureService, row_to_X

MODEL_PATH = Path("serving/models/lgbm_churn.pkl")
FEATS_PATH = Path("serving/models/feature_names.txt")

app = FastAPI(title="Retail Churn Model", version="1.0")

backend = os.getenv("FEATURE_BACKEND", "offline").lower()
feature_service = RedisFeatureService() if backend == "redis" else FeatureService()
model = joblib.load(MODEL_PATH)
feature_names = FEATS_PATH.read_text().splitlines()

class PredictRequest(BaseModel):
    customer_id: int
    t_ref: Optional[str] = None
    latest: bool = False
    threshold: float = Field(0.5, ge=0, le=1)

class PredictResponse(BaseModel):
    customer_id: int
    t_ref: str
    probability: float
    prediction: int
    log_path: Optional[str] = None
    log_error: Optional[str] = None

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    t0 = time.perf_counter()
    try:
        row = feature_service.get_snapshot(req.customer_id, t_ref=req.t_ref, latest=req.latest)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    X = row_to_X(row, feature_names)
    proba = float(model.predict_proba(X)[:, 1][0])
    label = int(proba >= req.threshold)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    log_path = None
    log_error = None
    try:
        log_path = log_prediction(
            payload=req.model_dump(),
            features_row=row,
            X_row=X,
            proba=proba,
            label=label,
            model_artifact=str(MODEL_PATH.name),
            latency_ms=latency_ms,
        )
    except Exception as e:
        log_error = str(e)
        print(f"[warn] logging failed: {e}")

    return PredictResponse(
        customer_id=req.customer_id,
        t_ref=row["t_ref"].isoformat(),
        probability=proba,
        prediction=label,
        log_path=log_path,
        log_error=log_error,
    )


@app.get("/health")
def health():
    info = {"backend": backend, "model": str(MODEL_PATH.name)}
    try:
        if backend == "redis" and hasattr(feature_service, "r"):
            pong = feature_service.r.ping()
            n_keys = sum(1 for _ in feature_service.r.scan_iter(match=f"{feature_service.prefix}*"))
            info.update({"redis_ping": bool(pong), "online_keys": n_keys})
        else:
            n_rows = len(feature_service.df)
            info.update({"offline_rows": n_rows})
        return {"status": "ok", **info}
    except Exception as e:
        return {"status": "error", "detail": str(e), **info}


@app.post("/refresh")
def refresh():
    n = feature_service.refresh()
    return {"reloaded_rows": n}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        row = feature_service.get_snapshot(req.customer_id, t_ref=req.t_ref, latest=req.latest)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    X = row_to_X(row, feature_names)
    proba = float(model.predict_proba(X)[:, 1][0])
    label = int(proba >= req.threshold)

    return PredictResponse(
        customer_id=req.customer_id,
        t_ref=row["t_ref"].isoformat(),
        probability=proba,
        prediction=label,
    )

from datetime import datetime, timezone
import pandas as pd
from .pred_logger import _partition_dir   # reuse helper

@app.post("/debug/write_test")
def write_test():
    ts = datetime.now(timezone.utc)
    out_dir = _partition_dir(ts)
    from pyarrow import Table
    from pyarrow import parquet as pq
    df = pd.DataFrame([{
        "ts_served": ts.isoformat(),
        "latency_ms": 0.0,
        "customer_id": 0,
        "t_ref": ts.isoformat(),
        "threshold": 0.5,
        "probability": 0.0,
        "prediction": 0,
        "model_artifact": "test",
    }])
    path = out_dir / "test.parquet"
    pq.write_table(Table.from_pandas(df, preserve_index=False), path)
    return {"wrote": str(path)}


@app.get("/debug/log_state")
def log_state():
    from pathlib import Path
    preds = list(Path("data/preds").rglob("*.parquet"))
    feats = list(Path("data/gold/features_customer").rglob("*.parquet"))
    return {"pred_files": len(preds), "feature_files": len(feats)}



# inside predict() just before returning the response:
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    t0 = time.perf_counter()
    try:
        row = feature_service.get_snapshot(req.customer_id, t_ref=req.t_ref, latest=req.latest)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    X = row_to_X(row, feature_names)
    proba = float(model.predict_proba(X)[:, 1][0])
    label = int(proba >= req.threshold)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    log_path = None
    try:
        log_path = log_prediction(
            payload=req.model_dump(),
            features_row=row,
            X_row=X,
            proba=proba,
            label=label,
            model_artifact=str(MODEL_PATH.name),
            latency_ms=latency_ms,
        )
    except Exception as e:
        print(f"[warn] logging failed: {e}")

    return PredictResponse(
        customer_id=req.customer_id,
        t_ref=row["t_ref"].isoformat(),
        probability=proba,
        prediction=label,
        log_path=log_path,
    )

    
