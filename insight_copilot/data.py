"""Dataset loading + a compact 'data card' the LLM can read (schema, ranges, valid values)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "train.csv"                  # real Kaggle file (commit this to the repo)
SAMPLE_PATH = ROOT / "data" / "sample_superstore.csv"    # synthetic fallback, same schema

# user-facing dimension name -> dataframe column
DIMS = {
    "region": "Region", "category": "Category", "sub_category": "Sub-Category", "segment": "Segment",
    "state": "State", "city": "City", "ship_mode": "Ship Mode", "product_name": "Product Name",
    "customer_name": "Customer Name",
}
REQUIRED = ["Order ID", "Order Date", "Region", "Category", "Sub-Category", "Segment", "State", "Sales"]


def using_sample_data() -> bool:
    return not DATA_PATH.exists()


@lru_cache(maxsize=1)
def load_df() -> pd.DataFrame:
    if using_sample_data() and not SAMPLE_PATH.exists():
        from .sample_data import generate
        generate(SAMPLE_PATH)
    df = pd.read_csv(SAMPLE_PATH if using_sample_data() else DATA_PATH)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")
    df = df.drop_duplicates().copy()
    for c in ("Order Date", "Ship Date"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Order Date", "Sales"])
    for c in DIMS.values():
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    if "Customer ID" not in df.columns:
        df["Customer ID"] = df["Customer Name"] if "Customer Name" in df.columns else "unknown"
    if "Row ID" not in df.columns:
        df["Row ID"] = range(1, len(df) + 1)
    return df.reset_index(drop=True)


def data_card() -> str:
    df = load_df()
    return (
        f"Dataset: Superstore Sales (US retailer), {len(df):,} order line-items, {df['Order ID'].nunique():,} orders, "
        f"{df['Customer ID'].nunique():,} customers.\n"
        f"Order dates: {df['Order Date'].min():%Y-%m-%d} to {df['Order Date'].max():%Y-%m-%d} "
        f"(the LATEST date in the data is {df['Order Date'].max():%Y-%m-%d}).\n"
        f"Columns: Row ID, Order ID, Order Date, Ship Date, Ship Mode, Customer ID, Customer Name, Segment, Country, "
        f"City, State, Postal Code, Region, Product ID, Category, Sub-Category, Product Name, Sales.\n"
        f"The ONLY numeric measure is Sales (USD, per line item). There is NO profit, cost, quantity, discount, "
        f"margin, inventory or returns data.\n"
        f"Regions: {', '.join(sorted(df['Region'].unique()))}. Categories: {', '.join(sorted(df['Category'].unique()))}. "
        f"Segments: {', '.join(sorted(df['Segment'].unique()))}. "
        f"Ship modes: {', '.join(sorted(df['Ship Mode'].unique()))}.\n"
        f"Sub-categories: {', '.join(sorted(df['Sub-Category'].unique()))}. States: {df['State'].nunique()} distinct."
    )


def validate_dataset() -> list[dict]:
    """Startup health check. Returns [{check, ok, detail}] - shown in the UI sidebar and tested offline."""
    checks: list[dict] = []

    def add(check: str, ok: bool, detail: str) -> None:
        checks.append({"check": check, "ok": bool(ok), "detail": detail})

    try:
        raw = pd.read_csv(SAMPLE_PATH if using_sample_data() else DATA_PATH)
        df = load_df()
    except Exception as e:  # noqa: BLE001
        return [{"check": "Dataset loads", "ok": False, "detail": f"{type(e).__name__}: {e}"}]
    add("Rows loaded", len(df) > 0, f"{len(df):,} rows" + (f" ({len(raw) - len(df):,} duplicate/invalid dropped)" if len(raw) != len(df) else ""))
    add("Order Date parsed", df["Order Date"].notna().all(),
        f"{df['Order Date'].min():%Y-%m-%d} -> {df['Order Date'].max():%Y-%m-%d}")
    add("Sales is numeric", pd.api.types.is_numeric_dtype(df["Sales"]), f"dtype {df['Sales'].dtype}")
    add("No missing Sales", df["Sales"].notna().all(), f"{int(df['Sales'].isna().sum())} missing")
    add("No negative Sales", (df["Sales"] >= 0).all(), f"min = {df['Sales'].min():,.2f}")
    add("Regions detected", df["Region"].nunique() >= 2, f"{df['Region'].nunique()}: {', '.join(sorted(df['Region'].unique()))}")
    add("Categories detected", df["Category"].nunique() >= 2, f"{df['Category'].nunique()}: {', '.join(sorted(df['Category'].unique()))}")
    return checks


def dataset_badge() -> dict:
    """Small summary for the UI header."""
    df = load_df()
    return {"name": "Superstore Sales" + (" (synthetic sample)" if using_sample_data() else ""), "rows": len(df),
            "start": int(df["Order Date"].dt.year.min()), "end": int(df["Order Date"].dt.year.max())}
