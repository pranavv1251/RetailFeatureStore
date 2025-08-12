import pandas as pd
import pyarrow as pa, pyarrow.parquet as pq
from pathlib import Path

BRONZE = Path("data/bronze/orders")
GOLD = Path("data/gold/features_customer")
WINDOWS = ["1D","7D","30D"]   # you can add "90D" later

def read_bronze() -> pd.DataFrame:
    parts = [pd.read_parquet(p) for p in BRONZE.rglob("*.parquet")]
    df = pd.concat(parts, ignore_index=True)
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], utc=True)
    # basic per-line signals (txns exclude cancels)
    df["txn"] = (~df["is_cancel"]).astype("int32")
    df["spend"] = df["line_amount"].where(~df["is_cancel"], 0.0).astype("float32")
    df["sku"] = df["stock_code"].astype(str)
    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # one feature row per (customer_id, event time at each invoice row)
    df = df.sort_values(["customer_id","invoice_date"])

    out_frames = []
    for cid, g in df.groupby("customer_id", sort=False):
        g = g.set_index("invoice_date")

        # ensure numeric dtypes for rolling ops
        g["txn"]    = g["txn"].astype("int32")
        g["spend"]  = g["spend"].astype("float32")
        g["cancel"] = g["is_cancel"].astype("int8")

        # map SKU strings -> per-customer category codes (ints)
        sku_codes = g["sku"].astype("category").cat.codes.astype("int32")

        rolls = {w: g.rolling(w, closed="both") for w in WINDOWS}
        out = pd.DataFrame(index=g.index)

        for w, rw in rolls.items():
            wl = w.lower()
            # numeric rollups
            out[f"txn_count_{wl}"] = rw["txn"].sum()
            out[f"spend_{wl}"]     = rw["spend"].sum()

            # nunique over sku codes (works because it's numeric)
            out[f"unique_skus_{wl}"] = sku_codes.rolling(w, closed="both").apply(
                lambda x: pd.Series(x).nunique(), raw=False
            ).astype("float32")

            denom = out[f"txn_count_{wl}"].where(lambda x: x > 0, other=1)
            out[f"avg_basket_value_{wl}"] = (out[f"spend_{wl}"] / denom).astype("float32")

            cancels = rw["cancel"].sum()
            total   = (cancels + rw["txn"].sum()).where(lambda x: x > 0, other=1)
            out[f"cancel_rate_{wl}"] = (cancels / total).clip(0, 1).astype("float32")

        # simple profile features
        first_ts = g.index.min()
        out["tenure_days"] = ((g.index - first_ts).days).astype("int32")
        out["country"] = g["country"].mode().iloc[0] if not g["country"].isna().all() else None

        out["customer_id"] = cid
        out = out.reset_index().rename(columns={"invoice_date": "t_ref"})
        out_frames.append(out)

    feats = pd.concat(out_frames, ignore_index=True)

    # clean NaNs
    for col in feats.columns:
        if feats[col].dtype.kind in "fc":
            feats[col] = feats[col].fillna(0)

    return feats


def write_partitioned(df: pd.DataFrame):
    GOLD.mkdir(parents=True, exist_ok=True)
    df["date"] = pd.to_datetime(df["t_ref"]).dt.tz_convert("UTC").dt.date.astype("string")
    for d, g in df.groupby("date"):
        out_dir = GOLD / f"date={d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(g.drop(columns=["date"]), preserve_index=False),
                       out_dir / "part-000.parquet")

if __name__ == "__main__":
    orders = read_bronze()
    feats = build_features(orders)
    write_partitioned(feats)
    print("✅ Features → data/gold/features_customer/")
