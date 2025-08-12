import json
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import joblib
import typer

GOLD = Path("data/gold/features_customer")
MODEL_PATH = Path("serving/models/lgbm_churn.pkl")
FEAT_NAMES_PATH = Path("serving/models/feature_names.txt")

app = typer.Typer(help="Local inference for churn model without FastAPI.")


def _load_latest_features() -> pd.DataFrame:
    parts = [pd.read_parquet(p) for p in GOLD.rglob("*.parquet")]
    feats = pd.concat(parts, ignore_index=True)
    feats["t_ref"] = pd.to_datetime(feats["t_ref"], utc=True)
    feats = feats.sort_values(["customer_id", "t_ref"])
    return feats


def _select_row(feats: pd.DataFrame, customer_id: int, t_ref: Optional[str], latest: bool):
    g = feats[feats["customer_id"] == customer_id]
    if g.empty:
        raise ValueError(f"No features for customer_id={customer_id}")
    if latest:
        return g.iloc[-1]
    if t_ref:
        tr = pd.to_datetime(t_ref, utc=True)
        g = g[g["t_ref"] <= tr]
        if g.empty:
            raise ValueError(f"No feature snapshot at or before t_ref={t_ref} for customer_id={customer_id}")
        return g.iloc[-1]
    # default: latest
    return g.iloc[-1]


def _row_to_X(row: pd.Series, feature_names: List[str]) -> pd.DataFrame:
    # Start with a single-row frame from the row
    df = pd.DataFrame([row])

    # Build country one-hots according to feature_names we trained with
    # Any column like "country__{VAL}" should be 1 iff df['country'] == VAL else 0
    for fn in feature_names:
        if fn.startswith("country__"):
            val = fn.split("country__", 1)[1]
            df[fn] = (df["country"].astype(str) == val).astype("int8")

    # Drop non-feature cols if present
    drop_cols = {"t_ref", "country", "churn_30d"}  # churn_30d won't exist here, but safe to drop
    existing_drop = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=existing_drop, errors="ignore")

    # Align to saved feature_names strictly
    for c in feature_names:
        if c not in df.columns:
            df[c] = 0
    df = df[feature_names].astype(float)

    return df


@app.command()
def predict(
    customer_id: int = typer.Option(..., help="Customer ID to score"),
    t_ref: Optional[str] = typer.Option(None, help='ISO time like "2011-11-01T00:00:00Z"'),
    latest: bool = typer.Option(False, help="Ignore t_ref and use the latest snapshot"),
    proba_threshold: float = typer.Option(0.5, help="Decision threshold for class label"),
):
    """Score a single customer snapshot and print JSON with prediction + probability + feature time."""
    # Load model + feature names
    model = joblib.load(MODEL_PATH)
    feature_names = FEAT_NAMES_PATH.read_text().splitlines()

    feats = _load_latest_features()
    row = _select_row(feats, customer_id, t_ref, latest)
    X = _row_to_X(row, feature_names)

    proba = float(model.predict_proba(X)[:, 1][0])
    label = int(proba >= proba_threshold)

    out = {
        "customer_id": int(customer_id),
        "t_ref": row["t_ref"].isoformat(),
        "prediction": label,
        "probability": proba,
    }
    print(json.dumps(out, indent=2))


@app.command()
def predict_batch(
    n: int = typer.Option(20, help="How many customers (latest snapshot each)"),
    out: Optional[str] = typer.Option(None, help="Optional CSV path to write outputs"),
    proba_threshold: float = typer.Option(0.5, help="Decision threshold"),
):
    """Score the latest snapshot for N customers (by customer_id ascending)."""
    model = joblib.load(MODEL_PATH)
    feature_names = FEAT_NAMES_PATH.read_text().splitlines()

    feats = _load_latest_features()
    latest_rows = (
        feats.sort_values(["customer_id", "t_ref"])
             .groupby("customer_id", as_index=False)
             .tail(1)
             .sort_values("customer_id")
             .head(n)
    )

    outputs = []
    for _, row in latest_rows.iterrows():
        X = _row_to_X(row, feature_names)
        proba = float(model.predict_proba(X)[:, 1][0])
        label = int(proba >= proba_threshold)
        outputs.append({
            "customer_id": int(row["customer_id"]),
            "t_ref": row["t_ref"].isoformat(),
            "prediction": label,
            "probability": proba,
        })

    df_out = pd.DataFrame(outputs)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(out, index=False)
        print(f"Wrote {len(df_out)} rows → {out}")
    else:
        print(df_out.to_string(index=False))


if __name__ == "__main__":
    app()
