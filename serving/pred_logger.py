from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import uuid

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

BASE_DIR = Path(__file__).resolve().parents[1]          # repo root
DATA_DIR = BASE_DIR / "data"
LOG_DIR  = DATA_DIR / "preds"

def _partition_dir(ts: datetime) -> Path:
    d = ts.astimezone(timezone.utc).date().isoformat()
    p = LOG_DIR / f"date={d}"
    p.mkdir(parents=True, exist_ok=True)
    return p

def _ensure_utc_iso(val) -> str:
    ts = pd.to_datetime(val, errors="coerce")
    if ts is pd.NaT:
        # fall back to "now" to avoid crashing logging
        ts = datetime.now(timezone.utc)
        return ts.isoformat()
    # localize / convert to UTC
    if getattr(ts, "tzinfo", None) is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat()

def log_prediction(*, payload: dict, features_row: pd.Series, X_row: pd.DataFrame,
                   proba: float, label: int, model_artifact: str, latency_ms: float) -> str:
    ts = datetime.now(timezone.utc)

    base = {
        "ts_served": ts.isoformat(),
        "latency_ms": float(latency_ms),
        "customer_id": int(payload["customer_id"]),
        "t_ref": _ensure_utc_iso(features_row["t_ref"]),
        "threshold": float(payload.get("threshold", 0.5)),
        "probability": float(proba),
        "prediction": int(label),
        "model_artifact": model_artifact,
    }
    # flatten numeric features
    feat_vals = {f"feat__{c}": float(X_row.iloc[0][c]) for c in X_row.columns}
    row = {**base, **feat_vals}
    df = pd.DataFrame([row])

    out_dir = _partition_dir(ts)
    fname = f"part-{ts.strftime('%H%M%S')}-{uuid.uuid4().hex[:8]}.parquet"
    out_path = out_dir / fname

    try:
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), out_path)
    except Exception as e:
        raise RuntimeError(f"write failed to {out_path}: {e}")

    return str(out_path)
