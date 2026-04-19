import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from data_preprocessing_baselines import build_base_raw  # chỉnh tên module nếu khác
from config.config import PROC_DIR
OUT_DIR = PROC_DIR / "raw_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def plot_sales_per_product(temporal_type: str = "unit") -> None:
    df = build_base_raw(temporal_type=temporal_type)
    df = df.sort_values(["node_id", "day"])

    out_dir = OUT_DIR / f"per_product_{temporal_type}"
    out_dir.mkdir(parents=True, exist_ok=True)

    node_ids = df["node_id"].unique()

    for nid in node_ids:
        sub = df[df["node_id"] == nid].sort_values("day")
        plt.figure(figsize=(8, 4))
        plt.plot(sub["day"], sub["sales_order"], marker="o", ms=2)
        plt.xlabel("Day")
        plt.ylabel("Sales order")
        plt.title(f"Sales order time series for product {nid} ({temporal_type})")
        plt.tight_layout()

        out_path = out_dir / f"product_{nid}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved product {nid} plot to {out_path}")

def main():
    # Một số product cụ thể
    plot_sales_per_product("unit")
    plot_sales_per_product("weight")

if __name__ == "__main__":
    main()
