import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config.config import (
    NODE_DIR,
    TEMPORAL_DIR,
    PROC_DIR,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    DEFAULT_EXPERIMENTS,
)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 0)

CAT_COLS = [
    "group", "sub_group", "plant", "storage_location",
    "day_of_week", "is_weekend",
]

def plot_sales_per_product_grid(
    temporal_type: str,
    horizon: int,
    max_products: int = 20,
) -> None:
    base_dir = PROC_DIR / "base"
    raw_path = base_dir / f"base_raw_h{horizon}_{temporal_type}.parquet"
    if not raw_path.exists():
        raise FileNotFoundError(f"Base RAW not found at {raw_path}")

    df = pd.read_parquet(raw_path)
    df = df.sort_values(["node_id", "day"])

    node_ids = df["node_id"].unique()
    node_ids = node_ids[:max_products]

    n = len(node_ids)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4 * ncols, 3 * nrows),
        sharex=False,
        sharey=False,
    )
    axes = axes.flatten()

    for ax, nid in zip(axes, node_ids):
        sub = df[df["node_id"] == nid].sort_values("day")
        ax.plot(sub["day"], sub["sales_order"], marker="o", ms=2)
        ax.set_title(str(nid))
        ax.set_xlabel("day")
        ax.set_ylabel("sales")

    # tắt các subplot dư
    for j in range(len(node_ids), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    out_dir = PROC_DIR / "plots" / "raw_timeseries"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ts_products_{temporal_type}_h{horizon}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved raw product time series grid to {out_path}")

def compute_time_splits(
    num_days: int,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,) -> tuple[int, int, int]:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Train/val/test ratios must sum to 1.0.")
    train_end = int(np.floor(num_days * train_ratio))
    val_end = int(np.floor(num_days * (train_ratio + val_ratio)))
    test_end = num_days
    return train_end, val_end, test_end


def load_node_metadata() -> pd.DataFrame:
    node_index_path = NODE_DIR / "NodesIndex.csv"
    node_group_path = NODE_DIR / "Node Types (Product Group and Subgroup).csv"
    node_plant_storage_path = NODE_DIR / "Nodes Type (Plant & Storage).csv"

    df_index = pd.read_csv(node_index_path)
    df_index = df_index.rename(columns={"Node": "node_id", "NodeIndex": "node_index"})
    df_index = df_index.drop_duplicates(keep="first")

    df_group = pd.read_csv(node_group_path)
    df_group = df_group.rename(
        columns={"Node": "node_id", "Group": "group", "Sub-Group": "sub_group"}
    )
    df_group = df_group.drop_duplicates(keep="first")

    df_plant = pd.read_csv(node_plant_storage_path)
    df_plant = df_plant.rename(
        columns={ "Node": "node_id", "Plant": "plant", "Storage Location": "storage_location",}
    )
    df_plant = df_plant.drop_duplicates(keep="first")

    df_meta = df_index.merge(df_group, on="node_id", how="left")
    df_meta = df_meta.merge(df_plant, on="node_id", how="left")
    return df_meta

def _load_temporal_wide_generic(subdir: str, filename: str, value_name: str) -> pd.DataFrame:
    path = TEMPORAL_DIR / subdir / filename
    df_wide = pd.read_csv(path)
    df_wide["Date"] = pd.to_datetime(df_wide["Date"])

    df_meta = load_node_metadata()
    valid_nodes = set(df_meta["node_id"].unique())

    cols_keep = ["Date"] + [c for c in df_wide.columns if c in valid_nodes]
    df_wide = df_wide[cols_keep]

    df_long = df_wide.melt(
        id_vars=["Date"],
        var_name="node_id",
        value_name=value_name,
    )
    df_long = df_long.rename(columns={"Date": "date"})
    df_long = (
        df_long
        .sort_values(["date", "node_id"])
        .drop_duplicates(subset=["date", "node_id"], keep="last")
    )
    return df_long

def load_temporal_unit_wide(filename: str, value_name: str) -> pd.DataFrame:
    return _load_temporal_wide_generic("Unit", filename, value_name)

def load_temporal_weight_wide(filename: str, value_name: str) -> pd.DataFrame:
    return _load_temporal_wide_generic("Weight", filename, value_name)

def _load_raw_data_generic(loader_func) -> pd.DataFrame:
    df_sales = loader_func("Sales Order.csv", "sales_order")
    df_prod = loader_func("Production.csv", "production")
    df_delv = loader_func("Delivery To distributor.csv", "delivery")
    df_issue = loader_func("Factory Issue.csv", "factory_issue")

    df = df_sales.merge(df_prod, on=["date", "node_id"], how="left")
    df = df.merge(df_delv, on=["date", "node_id"], how="left")
    df = df.merge(df_issue, on=["date", "node_id"], how="left")
    df = df.sort_values(["date", "node_id"]).reset_index(drop=True)

    unique_dates = df["date"].drop_duplicates().sort_values().reset_index(drop=True)
    date_to_day = {d: i + 1 for i, d in unique_dates.items()}
    df["day"] = df["date"].map(date_to_day)

    df_meta = load_node_metadata()
    df = df.merge(df_meta, on="node_id", how="left")

    cols_order = [
        "node_id", "node_index", "date", "day",
        "sales_order", "production", "delivery", "factory_issue",
        "group", "sub_group", "plant", "storage_location",
    ]
    df = df[cols_order]
    df = (
        df
        .sort_values(["date", "node_id"])
        .drop_duplicates(subset=["date", "node_id"], keep="last")
    )
    return df

def load_raw_data_unit() -> pd.DataFrame:
    return _load_raw_data_generic(load_temporal_unit_wide)

def load_raw_data_weight() -> pd.DataFrame:
    return _load_raw_data_generic(load_temporal_weight_wide)

def add_lag_features(df: pd.DataFrame, lag_cols: list[str], max_lag: int) -> pd.DataFrame:
    df = df.sort_values(["node_id", "day"]).copy()
    for col in lag_cols:
        for lag in range(1, max_lag + 1):
            df[f"{col}_lag{lag}"] = df.groupby("node_id")[col].shift(lag)
    return df

def add_rolling_stats(df: pd.DataFrame,
                      cols: list[str],
                      window: int = 7) -> pd.DataFrame:
    df = df.sort_values(["node_id", "day"]).copy()
    for col in cols:
        grp = df.groupby("node_id")[col]
        df[f"{col}_roll{window}_mean"] = grp.transform(lambda x: x.rolling(window).mean())
        df[f"{col}_roll{window}_std"]  = grp.transform(lambda x: x.rolling(window).std())
        df[f"{col}_roll{window}_max"]  = grp.transform(lambda x: x.rolling(window).max())
        df[f"{col}_roll{window}_min"]  = grp.transform(lambda x: x.rolling(window).min())
    return df

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day_of_week"] = df["date"].dt.weekday
    df["is_weekend"] = df["day_of_week"] >= 5
    df["month"] = df["date"].dt.month
    df["day_of_month"] = df["date"].dt.day
    return df

def create_labels(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    df = df.sort_values(["node_id", "day"]).copy()
    df[f"y_h{horizon}"] = df.groupby("node_id")["sales_order"].shift(-horizon)
    return df

def assign_splits(df: pd.DataFrame,
                  train_ratio: float = TRAIN_RATIO,
                  val_ratio: float = VAL_RATIO,
                  test_ratio: float = TEST_RATIO) -> pd.DataFrame:
    num_days = df["day"].max()
    train_end, val_end, test_end = compute_time_splits(
        num_days, train_ratio, val_ratio, test_ratio
    )

    df = df.copy()
    conditions = [
        df["day"] <= train_end,
        (df["day"] > train_end) & (df["day"] <= val_end),
        df["day"] > val_end,
    ]
    choices = ["train", "val", "test"]

    df["split"] = np.select(conditions, choices, default="test")
    return df

def filter_valid_samples(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    lag_cols = [c for c in df.columns if "lag" in c]
    label_cols = [f"y_h{horizon}"]
    df = df.dropna(subset=lag_cols + label_cols)
    return df

def one_hot_encode_splits(df: pd.DataFrame, cat_cols: list[str]):
    df_train = df[df["split"] == "train"].copy()
    df_val = df[df["split"] == "val"].copy()
    df_test = df[df["split"] == "test"].copy()

    cat_cols_present = [c for c in cat_cols if c in df.columns]

    df_train_enc = pd.get_dummies(df_train, columns=cat_cols_present, drop_first=False)
    df_val_enc = pd.get_dummies(df_val, columns=cat_cols_present, drop_first=False)
    df_test_enc = pd.get_dummies(df_test, columns=cat_cols_present, drop_first=False)

    df_val_enc = df_val_enc.reindex(columns=df_train_enc.columns, fill_value=0)
    df_test_enc = df_test_enc.reindex(columns=df_train_enc.columns, fill_value=0)

    df_all_enc = pd.concat([df_train_enc, df_val_enc, df_test_enc], axis=0)
    return df_all_enc

def build_base_raw(temporal_type: str, horizon: int) -> pd.DataFrame:
    """
    Base RAW (no lag/rolling), dùng chung cho:
    - GRU sequence
    - Base_full cho XGBoost / GNN / naive.
    Lưu vào: data/processed/base/base_raw_h{H}_{temporal_type}.parquet
    """
    if temporal_type == "unit":
        df = load_raw_data_unit()
    elif temporal_type == "weight":
        df = load_raw_data_weight()
    else:
        raise ValueError(
            f"Unknown temporal_type={temporal_type}, must be 'unit' or 'weight'."
        )

    df = add_calendar_features(df)
    df = create_labels(df, horizon=horizon)
    df = assign_splits(df)

    base_dir = PROC_DIR / "base"
    base_dir.mkdir(parents=True, exist_ok=True)
    out_path = base_dir / f"base_raw_h{horizon}_{temporal_type}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved RAW base timeseries (H={horizon}, {temporal_type}) to {out_path}")
    return df

def build_base_full(temporal_type: str, horizon: int, lag_window: int) -> pd.DataFrame:
    """
    Base FULL (lag + rolling), dùng chung cho:
    - XGBoost tabular
    - Naive baseline (full CSV)
    - Có thể dùng cho GNN nếu cần lag.
    Lưu vào: data/processed/base/base_full_h{H}_lag{L}_{temporal_type}.parquet
    """
    base_dir = PROC_DIR / "base"
    base_dir.mkdir(parents=True, exist_ok=True)
    raw_path = base_dir / f"base_raw_h{horizon}_{temporal_type}.parquet"
    if raw_path.exists():
        df = pd.read_parquet(raw_path)
        print(f"Loaded RAW base from {raw_path}")
    else:
        df = build_base_raw(temporal_type=temporal_type, horizon=horizon)

    lag_cols = ["sales_order", "production", "delivery", "factory_issue"]
    df = add_lag_features(df, lag_cols, max_lag=lag_window)
    df = add_rolling_stats(df, lag_cols, window=lag_window)
    df = filter_valid_samples(df, horizon=horizon)

    out_path = base_dir / f"base_full_h{horizon}_lag{lag_window}_{temporal_type}.parquet"
    df.to_parquet(out_path, index=False)
    print(
        f"Saved FULL base timeseries (H={horizon}, lag={lag_window}, "
        f"{temporal_type}) to {out_path}"
    )
    return df

def build_baseline_full_csv_for_naive(
    temporal_type: str,
    horizon: int,
    lag_window: int,
) -> pd.DataFrame:
    """
    Xuất base_full thành CSV cho naive baseline (hoặc các baseline khác cần CSV).
    Input: base/base_full_h{H}_lag{L}_{temporal_type}.parquet
    Output: baseline/xgboost/xgboost_h{H}_lag{L}_{temporal_type}_full.csv
    """
    print(
        f"\n=== Building baseline FULL CSV for naive, "
        f"H={horizon}, lag={lag_window}, temporal_type={temporal_type} ==="
    )

    base_dir = PROC_DIR / "base"
    full_path = base_dir / f"base_full_h{horizon}_lag{lag_window}_{temporal_type}.parquet"
    if full_path.exists():
        df = pd.read_parquet(full_path)
        print(f"Loaded FULL base from {full_path}")
    else:
        df = build_base_full(
            temporal_type=temporal_type,
            horizon=horizon,
            lag_window=lag_window,
        )

    out_dir = PROC_DIR / "baseline" / "xgboost"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"xgboost_h{horizon}_lag{lag_window}_{temporal_type}_full.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved FULL baseline CSV for naive to {out_path}")
    return df

def build_xgboost_tabular(
    temporal_type: str,
    horizon: int,
    lag_window: int,
):
    """
    Baseline 1: XGBoost tabular (no graph).
    - input: base/base_full_h{H}_lag{L}_{temporal_type}.parquet
    - one-hot CAT_COLS
    - lưu: baseline/xgboost/xgboost_tabular_h{H}_lag{L}_{temporal_type}.parquet
    """
    print(
        f"\n=== Building XGBoost tabular OHE baseline (no graph), "
        f"H={horizon}, lag={lag_window}, temporal_type={temporal_type} ==="
    )

    base_dir = PROC_DIR / "base"
    full_path = base_dir / f"base_full_h{horizon}_lag{lag_window}_{temporal_type}.parquet"
    if full_path.exists():
        df = pd.read_parquet(full_path)
        print(f"Loaded FULL base from {full_path}")
    else:
        df = build_base_full(
            temporal_type=temporal_type,
            horizon=horizon,
            lag_window=lag_window,
        )

    feature_cols = [
        c
        for c in df.columns
        if any(
            kw in c
            for kw in [
                "lag",                        
                f"roll{lag_window}_",          
                "group", "sub_group",
                "plant", "storage_location",
                "day_of_week", "is_weekend", "month", "day_of_month",
            ]
        )
    ]
    base_cols = ["node_id", "node_index", "date", "day", "split"]
    label_col = f"y_h{horizon}"

    df_h = df[base_cols + feature_cols + [label_col]].rename(
        columns={label_col: "target"}
    )

    df_ohe = one_hot_encode_splits(df_h, CAT_COLS)

    out_dir = PROC_DIR / "baseline" / "xgboost"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"xgboost_tabular_h{horizon}_lag{lag_window}_{temporal_type}.parquet"
    df_ohe.to_parquet(out_path, index=False)
    print(f"Saved XGBoost baseline to {out_path}")
    return df_ohe

def build_gru_sequence(
    temporal_type: str,
    horizon: int,
    seq_len: int,
):
    print(
        f"\n=== Building GRU sequence base (no graph), "
        f"H={horizon}, seq_len={seq_len}, temporal_type={temporal_type} ==="
    )

    base_dir = PROC_DIR / "base"
    raw_path = base_dir / f"base_raw_h{horizon}_{temporal_type}.parquet"
    if raw_path.exists():
        df = pd.read_parquet(raw_path)
        print(f"Loaded RAW base from {raw_path}")
    else:
        df = build_base_raw(temporal_type=temporal_type, horizon=horizon)

    label_col = f"y_h{horizon}"
    keep_cols = [
        "node_id", "node_index", "date", "day", "split",
        "sales_order", "production", "delivery", "factory_issue",
        "day_of_week", "is_weekend", "month", "day_of_month",
        label_col,
        "group", "sub_group", "plant", "storage_location",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df_gru = df[keep_cols].rename(columns={label_col: "target"})

    df_ohe = one_hot_encode_splits(df_gru, CAT_COLS)

    out_dir = PROC_DIR / "baseline" / "gru"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"gru_sequence_h{horizon}_L{seq_len}_{temporal_type}.parquet"
    df_ohe.to_parquet(out_path, index=False)
    print(f"Saved GRU base to {out_path}")
    return df_ohe

def inspect_sales_order_hist(df: pd.DataFrame, tag: str = "unit"):
    """
    Vẽ histogram của sales_order theo 4 kiểu scale và lưu PNG.
    """
    if "sales_order" not in df.columns:
        raise KeyError("Column 'sales_order' not found in df")

    y = df["sales_order"].astype(float)

    print(f"\n[INFO] Summary of sales_order ({tag}):")
    print(y.describe(percentiles=[0.5, 0.9, 0.99]))
    print("Num zeros:", int((y == 0).sum()))
    print("Num negatives:", int((y < 0).sum()))

    q99 = y.quantile(0.99)
    upper = max(q99, y.max())
    if upper <= 0:
        upper = 1.0
    bins = np.linspace(0, upper, 100)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    axes[0].hist(y, bins=bins, color="steelblue", alpha=0.8)
    axes[0].set_title(f"Histogram sales_order ({tag}) - linear-linear")
    axes[0].set_xlabel("sales_order")
    axes[0].set_ylabel("count")

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

    axes[2].hist(y, bins=bins, color="seagreen", alpha=0.8)
    axes[2].set_yscale("log")
    axes[2].set_title(f"Histogram sales_order ({tag}) - linear-log")
    axes[2].set_xlabel("sales_order")
    axes[2].set_ylabel("count (log)")

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

def main() -> None:
    for exp in DEFAULT_EXPERIMENTS:
        t_type = exp.temporal_type
        for H in exp.horizons:
            df_raw  =build_base_raw(temporal_type=t_type, horizon=H)
            inspect_sales_order_hist(df_raw, tag="unit")
            plot_sales_per_product_grid(
                temporal_type=t_type,
                horizon=H,
                max_products=40,  
            )

            for L_seq in exp.gru_seq_lengths:
                build_gru_sequence(
                    temporal_type=t_type,
                    horizon=H,
                    seq_len=L_seq,
                )
                
            for L in exp.lag_windows:
                build_base_full(temporal_type=t_type, horizon=H, lag_window=L)
                build_baseline_full_csv_for_naive(
                    temporal_type=t_type,
                    horizon=H,
                    lag_window=L,
                )
                build_xgboost_tabular(
                    temporal_type=t_type,
                    horizon=H,
                    lag_window=L,
                )
if __name__ == "__main__":
    main()