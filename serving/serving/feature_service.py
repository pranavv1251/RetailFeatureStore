from __future__ import annotations
from pathlib import Path
from typing import Optional, List
import os
import pandas as pd
import redis

# Absolute paths so Docker + host both work
BASE_DIR = Path(__file__).resolve().parents[1]
GOLD = BASE_DIR / "data" / "gold" / "features_customer"

class FeatureService:
    """Offline feature reader from Parquet (used as fallback)."""
    def __init__(self, gold_dir: Path = GOLD):
        self.gold_dir = gold_dir
        self._df: Optional[pd.DataFrame] = None

    def _load(self) -> pd.DataFrame:
        parts = list(self.gold_dir.rglob("*.parquet"))
        if not parts:
            return pd.DataFrame()
        feats = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        feats["t_ref"] = pd.to_datetime(feats["t_ref"], utc=True)
        return feats.sort_values(["customer_id", "t_ref"])

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            self._df = self._load()
        return self._df

    def refresh(self) -> int:
        self._df = self._load()
        return len(self._df)

    def get_snapshot(self, customer_id: int, *, t_ref: Optional[str], latest: bool) -> pd.Series:
        g = self.df[self.df["customer_id"] == customer_id]
        if g.empty:
            raise KeyError(f"No features for customer_id={customer_id}")
        if latest or t_ref is None:
            return g.iloc[-1]
        tr = pd.to_datetime(t_ref, utc=True)
        g = g[g["t_ref"] <= tr]
        if g.empty:
            raise KeyError(f"No feature snapshot at/before {t_ref} for customer_id={customer_id}")
        return g.iloc[-1]

def row_to_X(row: pd.Series, feature_names: List[str]) -> pd.DataFrame:
    """Turn a single snapshot row into a 1xN frame matching trained columns."""
    df = pd.DataFrame([row])
    for fn in feature_names:
        if fn.startswith("country__"):
            val = fn.split("country__", 1)[1]
            df[fn] = (df.get("country", "").astype(str) == val).astype("int8")
    df = df.drop(columns=[c for c in ["t_ref", "country", "churn_30d"] if c in df.columns], errors="ignore")
    for c in feature_names:
        if c not in df.columns:
            df[c] = 0
    return df[feature_names].astype(float)

class RedisFeatureService(FeatureService):
    """Online feature reader from Redis hashes (latest snapshot per customer)."""
    def __init__(self):
        super().__init__()
        url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.r = redis.from_url(url, decode_responses=True)
        self.prefix = os.getenv("REDIS_KEY_PREFIX", "fs:customer:")

    def get_snapshot(self, customer_id: int, *, t_ref: Optional[str], latest: bool) -> pd.Series:
        # Redis only stores "latest"; if latest requested (or no t_ref), try Redis first
        if latest or t_ref is None:
            key = f"{self.prefix}{customer_id}"
            h = self.r.hgetall(key)
            if h:
                row = {
                    "customer_id": customer_id,
                    "t_ref": pd.to_datetime(h.get("meta:t_ref"), utc=True, errors="coerce"),
                    "country": h.get("meta:country"),
                }
                for k, v in h.items():
                    if k.startswith("meta:"):
                        continue
                    try:
                        row[k] = float(v)
                    except Exception:
                        row[k] = v
                return pd.Series(row)
        # Fallback to offline Parquet if Redis miss or historical t_ref needed
        return super().get_snapshot(customer_id, t_ref=t_ref, latest=True)
