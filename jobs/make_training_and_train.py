import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from lightgbm import LGBMClassifier
import mlflow

BRONZE = Path("data/bronze/orders")
GOLD   = Path("data/gold/features_customer")
ARTIFACTS = Path("serving/models"); ARTIFACTS.mkdir(parents=True, exist_ok=True)

def load_orders() -> pd.DataFrame:
    parts = [pd.read_parquet(p) for p in BRONZE.rglob("*.parquet")]
    df = pd.concat(parts, ignore_index=True)
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], utc=True)
    df["is_purchase"]  = (~df["is_cancel"]).astype(int) & (df["quantity"] > 0)
    return df[["customer_id","invoice_date","is_purchase","country"]]

def load_features() -> pd.DataFrame:
    parts = [pd.read_parquet(p) for p in GOLD.rglob("*.parquet")]
    feats = pd.concat(parts, ignore_index=True)
    feats["t_ref"] = pd.to_datetime(feats["t_ref"], utc=True)
    return feats

def make_labels(orders: pd.DataFrame, feats: pd.DataFrame, horizon="30D") -> pd.DataFrame:
    # churn_30d = 1 iff NO purchase in (t_ref, t_ref + 30D]
    orders = orders[orders["is_purchase"] == 1][["customer_id","invoice_date"]]
    orders = orders.rename(columns={"invoice_date":"event_ts"}).sort_values(["customer_id","event_ts"])
    h = pd.Timedelta(horizon)

    # group order timestamps per customer for quick search
    ts_by_cid = {cid: g["event_ts"].to_numpy() for cid, g in orders.groupby("customer_id", sort=False)}

    labels = []
    for cid, g in feats.groupby("customer_id", sort=False):
        trefs = g["t_ref"].to_numpy()
        events = ts_by_cid.get(cid, np.array([], dtype="datetime64[ns]"))
        for tr in trefs:
            # earliest event strictly after t_ref
            idx = np.searchsorted(events, tr, side="right")
            has_future = idx < len(events) and events[idx] <= (tr + h)
            labels.append((cid, tr, 0 if has_future else 1))

    lab = pd.DataFrame(labels, columns=["customer_id","t_ref","churn_30d"])
    return feats.merge(lab, on=["customer_id","t_ref"], how="inner").sort_values(["customer_id","t_ref"])

def encode_and_split(df: pd.DataFrame):
    # one-hot top 10 countries (simple, fast)
    top_c = df["country"].value_counts().head(10).index.tolist()
    for c in top_c:
        df[f"country__{c}"] = (df["country"] == c).astype("int8")
    df = df.drop(columns=["country","t_ref"])

    y = df["churn_30d"].astype(int).to_numpy()
    X = df.drop(columns=["churn_30d"])
    # time-aware split (no shuffle)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    return X_train, X_test, y_train, y_test, X.columns

def train_log_model(X_train, X_test, y_train, y_test, feature_names):
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("retail_churn")

    params = dict(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
    )
    with mlflow.start_run():
        model = LGBMClassifier(**params)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)
        auc = roc_auc_score(y_test, proba)
        f1  = f1_score(y_test, preds)

        mlflow.log_params(params)
        mlflow.log_metric("auc", float(auc))
        mlflow.log_metric("f1", float(f1))

        # persist artifacts for serving
        import joblib
        ARTIFACTS = Path("serving/models"); ARTIFACTS.mkdir(parents=True, exist_ok=True)
        model_path = ARTIFACTS / "lgbm_churn.pkl"
        joblib.dump(model, model_path)
        (ARTIFACTS / "feature_names.txt").write_text("\n".join(feature_names))
        mlflow.log_artifact(str(model_path), artifact_path="model")
        mlflow.log_artifact(str(ARTIFACTS / "feature_names.txt"), artifact_path="model")
        print(f"✅ Trained. AUC={auc:.3f}  F1={f1:.3f}.  Saved → {model_path}")

if __name__ == "__main__":
    orders = load_orders()
    feats   = load_features()
    df      = make_labels(orders, feats, horizon="30D")
    X_train, X_test, y_train, y_test, feat_names = encode_and_split(df)
    train_log_model(X_train, X_test, y_train, y_test, feat_names)
