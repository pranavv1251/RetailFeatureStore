import os, json, time
from pathlib import Path
import pandas as pd
import redis

BASE = Path(__file__).resolve().parents[1]
GOLD = BASE / "data" / "gold" / "features_customer"
MODEL_FEATURES = BASE / "serving" / "models" / "feature_names.txt"

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
KEY_PREFIX = os.getenv("REDIS_KEY_PREFIX", "fs:customer:")

def load_latest_snapshots(n_limit: int | None = None) -> pd.DataFrame:
    parts = sorted(GOLD.rglob("*.parquet"))
    if not parts:
        raise SystemExit("No feature parquet files found; build features first.")
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["t_ref"] = pd.to_datetime(df["t_ref"], utc=True)
    latest = df.sort_values(["customer_id","t_ref"]).groupby("customer_id").tail(1)
    if n_limit: latest = latest.head(n_limit)
    return latest

def row_to_online_fields(row: pd.Series, feature_names: list[str]) -> dict[str,str]:
    # one-hot countries exactly like training
    fields = {}
    for fn in feature_names:
        if fn.startswith("country__"):
            val = fn.split("country__",1)[1]
            fields[fn] = "1" if str(row.get("country","")) == val else "0"
    # copy numeric features
    for c in feature_names:
        if c in fields:  # already set (country one-hot)
            continue
        if c in row:
            fields[c] = str(float(row[c])) if pd.notna(row[c]) else "0"
        else:
            fields[c] = "0"
    # metadata
    fields["meta:t_ref"] = row["t_ref"].isoformat()
    fields["meta:country"] = str(row.get("country",""))
    fields["meta:updated_at"] = pd.Timestamp.utcnow().isoformat()
    return fields

if __name__ == "__main__":
    r = redis.from_url(REDIS_URL, decode_responses=True)
    feat_names = MODEL_FEATURES.read_text().splitlines()
    latest = load_latest_snapshots()

    t0 = time.time()
    pipe = r.pipeline()
    for _, row in latest.iterrows():
        key = f"{KEY_PREFIX}{int(row['customer_id'])}"
        fields = row_to_online_fields(row, feat_names)
        pipe.hset(key, mapping=fields)
        ttl = os.getenv("REDIS_TTL_SECONDS")
        if ttl: pipe.expire(key, int(ttl))
    pipe.execute()
    print(f"✅ Materialized {len(latest)} customers to Redis in {time.time()-t0:.2f}s")
