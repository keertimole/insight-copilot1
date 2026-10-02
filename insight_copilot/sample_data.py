"""Synthetic stand-in that mirrors the schema of Kaggle's `rohitsahoo/salesforecasting` train.csv.

Only used when data/train.csv is absent, so the app never crashes on a fresh clone.
It has seasonality, growth, regional differences and one planted outlier order.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CATALOG = {  # (category, sub-category): (median unit price, sigma)
    ("Furniture", "Bookcases"): (220, 0.6), ("Furniture", "Chairs"): (260, 0.7),
    ("Furniture", "Furnishings"): (40, 0.8), ("Furniture", "Tables"): (400, 0.7),
    ("Office Supplies", "Appliances"): (90, 0.9), ("Office Supplies", "Art"): (15, 0.8),
    ("Office Supplies", "Binders"): (25, 1.0), ("Office Supplies", "Envelopes"): (20, 0.7),
    ("Office Supplies", "Fasteners"): (8, 0.6), ("Office Supplies", "Labels"): (12, 0.7),
    ("Office Supplies", "Paper"): (18, 0.8), ("Office Supplies", "Storage"): (60, 0.9),
    ("Office Supplies", "Supplies"): (30, 1.0), ("Technology", "Accessories"): (55, 0.9),
    ("Technology", "Copiers"): (900, 0.9), ("Technology", "Machines"): (350, 1.1),
    ("Technology", "Phones"): (200, 0.8),
}
STATES = {
    "West": ["California", "Washington", "Oregon", "Colorado", "Arizona"],
    "East": ["New York", "Pennsylvania", "Massachusetts", "Ohio", "New Jersey"],
    "Central": ["Texas", "Illinois", "Michigan", "Minnesota", "Wisconsin"],
    "South": ["Florida", "North Carolina", "Virginia", "Georgia", "Tennessee"],
}
REGION_P = {"West": 0.32, "East": 0.28, "Central": 0.24, "South": 0.16}
SEASON = [0.55, 0.50, 0.80, 0.75, 0.75, 0.85, 0.75, 0.85, 1.35, 1.00, 1.45, 1.60]
GROWTH = {2015: 1.0, 2016: 1.05, 2017: 1.25, 2018: 1.45}
SHIP = ["Standard Class", "Second Class", "First Class", "Same Day"]
SEGMENTS = ["Consumer", "Corporate", "Home Office"]
BRANDS = ["Acme", "Globex", "Initech", "Umbra", "Hooli", "Vandelay"]


def generate(path: Path, seed: int = 42) -> Path:
    rng = np.random.default_rng(seed)
    subs = list(CATALOG)
    products = {s: [f"{rng.choice(BRANDS)} {s[1]} Model {k}" for k in range(1, 7)] for s in subs}
    customers = [(f"CU-{i:05d}", f"Customer {i}", rng.choice(SEGMENTS, p=[.52, .30, .18])) for i in range(1, 701)]
    regions, rp = list(REGION_P), list(REGION_P.values())
    rows, row_id, order_no = [], 1, 100000
    for year in range(2015, 2019):
        for month in range(1, 13):
            n_orders = rng.poisson(85 * SEASON[month - 1] * GROWTH[year])
            for _ in range(n_orders):
                order_no += 1
                odate = pd.Timestamp(year=year, month=month, day=int(rng.integers(1, 29)))
                sdate = odate + pd.Timedelta(days=int(rng.integers(1, 8)))
                cid, cname, seg = customers[int(rng.integers(len(customers)))]
                region = str(rng.choice(regions, p=rp))
                state = str(rng.choice(STATES[region]))
                for _ in range(1 + rng.poisson(1.0)):
                    tilt = {"Technology": 1.35 if region == "West" else 1.0,
                            "Furniture": 1.3 if region == "Central" else 1.0, "Office Supplies": 1.0}
                    w = np.array([tilt[s[0]] * (3 if s[0] == "Office Supplies" else 1) for s in subs], float)
                    cat, sub = subs[int(rng.choice(len(subs), p=w / w.sum()))]
                    mu, sg = CATALOG[(cat, sub)]
                    sales = float(rng.lognormal(np.log(mu), sg) * rng.integers(1, 5))
                    rows.append([row_id, f"US-{year}-{order_no}", odate, sdate,
                                 str(rng.choice(SHIP, p=[.6, .2, .15, .05])), cid, cname, seg, "United States",
                                 f"{state} City", state, 10000 + int(rng.integers(89999)), region,
                                 f"{cat[:3].upper()}-{sub[:2].upper()}-{int(rng.integers(10**7))}", cat, sub,
                                 str(rng.choice(products[(cat, sub)])), round(sales, 4)])
                    row_id += 1
    # planted anomaly: one very large copier order
    rows.append([row_id, "US-2017-999999", pd.Timestamp("2017-03-18"), pd.Timestamp("2017-03-22"), "First Class",
                 "CU-00042", "Customer 42", "Corporate", "United States", "Seattle", "Washington", 98115, "West",
                 "TEC-CO-99999999", "Technology", "Copiers", "Acme Copiers Model 9 (bulk enterprise order)", 17499.95])
    cols = ["Row ID", "Order ID", "Order Date", "Ship Date", "Ship Mode", "Customer ID", "Customer Name", "Segment",
            "Country", "City", "State", "Postal Code", "Region", "Product ID", "Category", "Sub-Category",
            "Product Name", "Sales"]
    df = pd.DataFrame(rows, columns=cols)
    for c in ("Order Date", "Ship Date"):
        df[c] = df[c].dt.strftime("%d/%m/%Y")  # same dd/mm/yyyy format as the Kaggle file
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
