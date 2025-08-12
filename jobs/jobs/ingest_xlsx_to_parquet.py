import pandas as pd
import pyarrow as pa, pyarrow.parquet as pq
from pathlib import Path

RAW_XLSX = "data/raw/online_retail_ii.xlsx"
BRONZE_DIR = Path("data/bronze/orders")

def read_all_sheets(path: str) -> pd.DataFrame:
    xl = pd.ExcelFile(path)
    dfs = [xl.parse(s) for s in xl.sheet_names]  # 'Year 2009-2010', 'Year 2010-2011'
    return pd.concat(dfs, ignore_index=True)

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    mapper = {
        "Invoice": "invoice",
        "StockCode": "stock_code",
        "Description": "description",
        "Quantity": "quantity",
        "InvoiceDate": "invoice_date",
        "Price": "unit_price",           # <-- explicit mapping for your file
        "Customer ID": "customer_id",    # <-- explicit mapping for your file
        "Country": "country",
    }
    df = df.rename(columns=mapper)

    df["invoice_date"] = pd.to_datetime(df["invoice_date"], utc=True, errors="coerce")
    df["description"]  = df["description"].fillna("").astype(str).str.strip()
    df["stock_code"]   = df["stock_code"].astype(str).str.upper().str.strip()
    df["country"]      = df["country"].astype(str).str.strip()

    df["quantity"]     = pd.to_numeric(df["quantity"], errors="coerce").astype("Int32")
    df["unit_price"]   = pd.to_numeric(df["unit_price"], errors="coerce").astype("float32")
    df["customer_id"]  = pd.to_numeric(df["customer_id"], errors="coerce").astype("Int32")

    df["invoice"]      = df["invoice"].astype(str).str.strip()
    df["is_cancel"]    = df["invoice"].str.startswith("C", na=False)
    df["line_amount"]  = (df["quantity"].astype("float32") * df["unit_price"]).astype("float32")
    return df

def drop_or_quarantine(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["customer_id", "invoice_date"])
    return df.astype({"customer_id": "int32", "quantity": "int32"})

def write_partitioned(df: pd.DataFrame):
    BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    df["date"] = df["invoice_date"].dt.date.astype("string")
    for d, g in df.groupby("date"):
        out_dir = BRONZE_DIR / f"date={d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(g.drop(columns=["date"]), preserve_index=False),
                       out_dir / "part-000.parquet")

if __name__ == "__main__":
    df = read_all_sheets(RAW_XLSX)
    df = normalize_cols(df)
    df = drop_or_quarantine(df)
    write_partitioned(df)
    print("✅ Ingest complete → data/bronze/orders/")
