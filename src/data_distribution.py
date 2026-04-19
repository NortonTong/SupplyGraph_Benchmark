# data_distribution.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from config.config import (
    PROC_DIR,
)
# Nếu bạn muốn dùng lại helper từ file preprocessing:
from data_preprocessing_baselines import (
    load_raw_data_unit,
    load_raw_data_weight,
    add_calendar_features,
    create_labels,
    assign_splits,
    HORIZON,
)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 0)


def build_base_raw_for_inspect(temporal_type: str = "unit") -> pd.DataFrame:
    """
    Tạo bảng time-series RAW để inspect:
    - load_raw_data_unit / load_raw_data_weight
    - add_calendar_features
    - create_labels(y_h7)
    - assign_splits
    Không có lag/rolling.
    """
    if temporal_type == "unit":
        df = load_raw_data_unit()
    elif temporal_type == "weight":
        df = load_raw_data_weight()
    else:
        raise ValueError(f"Unknown temporal_type={temporal_type}, must be 'unit' or 'weight'.")

    df = add_calendar_features(df)
    df = create_labels(df, horizon=HORIZON)
    df = assign_splits(df)

    # Lưu lại để sau này tiện dùng nếu cần
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROC_DIR / f"base_raw_{temporal_type}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"[INFO] Saved RAW base timeseries ({temporal_type}) to {out_path}")
    return df


def inspect_sales_order_hist(df: pd.DataFrame, tag: str = "unit"):
    """
    Vẽ histogram của sales_order theo 4 kiểu scale:
    - linear-linear
    - log-log
    - linear-log
    - log-linear
    """
    if "sales_order" not in df.columns:
        raise KeyError("Column 'sales_order' not found in df")

    y = df["sales_order"].astype(float)

    print(f"\n[INFO] Summary of sales_order ({tag}):")
    print(y.describe(percentiles=[0.5, 0.9, 0.99]))
    print("Num zeros:", int((y == 0).sum()))
    print("Num negatives:", int((y < 0).sum()))

    # Bins cho histogram linear
    q99 = y.quantile(0.99)
    upper = max(q99, y.max())
    if upper <= 0:
        upper = 1.0
    bins = np.linspace(0, upper, 100)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    # 1) linear-linear
    axes[0].hist(y, bins=bins, color="steelblue", alpha=0.8)
    axes[0].set_title(f"Histogram sales_order ({tag}) - linear-linear")
    axes[0].set_xlabel("sales_order")
    axes[0].set_ylabel("count")

    # 2) log-log (chỉ giá trị > 0)
    y_pos = y[y > 0]
    if len(y_pos) > 0:
        bins_log = np.logspace(
            np.log10(max(1e-3, y_pos.min())),
            np.log10(y_pos.max()),
            100,
        )

        axes[1].hist(y_pos, bins=bins_log, color="darkorange", alpha=0.8)
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        axes[1].set_title(f"Histogram sales_order ({tag}) - log-log")
        axes[1].set_xlabel("sales_order (log)")
        axes[1].set_ylabel("count (log)")
    else:
        axes[1].set_title("No positive sales_order values")

    # 3) linear-log (x linear, y log)
    axes[2].hist(y, bins=bins, color="seagreen", alpha=0.8)
    axes[2].set_yscale("log")
    axes[2].set_title(f"Histogram sales_order ({tag}) - linear-log")
    axes[2].set_xlabel("sales_order")
    axes[2].set_ylabel("count (log)")

    # 4) log-linear (x log, y linear)
    if len(y_pos) > 0:
        axes[3].hist(y_pos, bins=bins_log, color="indigo", alpha=0.8)
        axes[3].set_xscale("log")
        axes[3].set_title(f"Histogram sales_order ({tag}) - log-linear")
        axes[3].set_xlabel("sales_order (log)")
        axes[3].set_ylabel("count")
    else:
        axes[3].set_title("No positive sales_order values")

    plt.tight_layout()

    out_dir = PROC_DIR / "sales_distribution"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sales_order_hist_4scales_{tag}.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[INFO] Saved histograms to {out_path}")


def main():
    # unit
    df_unit_raw = build_base_raw_for_inspect(temporal_type="unit")
    inspect_sales_order_hist(df_unit_raw, tag="unit")

    # weight
    df_weight_raw = build_base_raw_for_inspect(temporal_type="weight")
    inspect_sales_order_hist(df_weight_raw, tag="weight")


if __name__ == "__main__":
    main()