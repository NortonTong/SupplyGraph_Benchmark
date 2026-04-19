import pandas as pd
import numpy as np
from config.config import (
    NODE_DIR,
    TEMPORAL_DIR,
    PROC_DIR,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 0)

HORIZON = 7
LAG_WINDOWS = [7,14]

CAT_COLS = ["group", "sub_group", "plant", "storage_location",
            "day_of_week", "is_weekend"]


def compute_time_splits(
    num_days: int,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
) -> tuple[int, int, int]:
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
        columns={
            "Node": "node_id",
            "Plant": "plant",
            "Storage Location": "storage_location",
        }
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
        for lag in range(1, max_lag + 1):  # từ 1 đến max_lag
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


def create_labels(df: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    df = df.sort_values(["node_id", "day"]).copy()
    df["y_h7"] = df.groupby("node_id")["sales_order"].shift(-horizon)
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


def filter_valid_samples(df: pd.DataFrame) -> pd.DataFrame:
    lag_cols = [c for c in df.columns if "lag" in c]
    label_cols = ["y_h7"]
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

def build_base_raw(temporal_type: str = "unit") -> pd.DataFrame:
    if temporal_type == "unit":
        df = load_raw_data_unit()
    elif temporal_type == "weight":
        df = load_raw_data_weight()
    else:
        raise ValueError(f"Unknown temporal_type={temporal_type}, must be 'unit' or 'weight'.")

    df = add_calendar_features(df)
    df = create_labels(df, horizon=HORIZON)
    df = assign_splits(df)

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROC_DIR / f"base_raw_{temporal_type}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved RAW base timeseries ({temporal_type}) to {out_path}")
    return df

def build_base_full(temporal_type: str = "unit",
                    lag_window: int = 7) -> pd.DataFrame:
    raw_path = PROC_DIR / f"base_raw_{temporal_type}.parquet"
    if raw_path.exists():
        df = pd.read_parquet(raw_path)
        print(f"Loaded RAW base from {raw_path}")
    else:
        df = build_base_raw(temporal_type=temporal_type)

    lag_cols = ["sales_order", "production", "delivery", "factory_issue"]
    # lag 1..lag_window
    df = add_lag_features(df, lag_cols, max_lag=lag_window)
    # rolling với cùng window
    df = add_rolling_stats(df, lag_cols, window=lag_window)
    df = filter_valid_samples(df)

    out_path = PROC_DIR / f"base_full_h{HORIZON}_lag{lag_window}_{temporal_type}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved FULL base timeseries (H={HORIZON}, lag={lag_window}, {temporal_type}) to {out_path}")
    return df

def build_xgboost_tabular(temporal_type: str = "unit",
                          lag_window: int = 7):
    """
    Baseline 1: XGBoost tabular (no graph), cho H=7, lag_window ∈ {7,14}.
    - input: base_full_h7_lag{lag_window}_{temporal_type}.parquet
    - one-hot CAT_COLS
    - lưu: baseline/xgboost_tabular_h7_lag{lag_window}_{temporal_type}.parquet
    """
    print(f"\n=== Building XGBoost tabular OHE baseline (no graph), "
          f"H={HORIZON}, lag={lag_window}, temporal_type={temporal_type} ===")

    full_path = PROC_DIR / f"base_full_h{HORIZON}_lag{lag_window}_{temporal_type}.parquet"
    if full_path.exists():
        df = pd.read_parquet(full_path)
        print(f"Loaded FULL base from {full_path}")
    else:
        df = build_base_full(temporal_type=temporal_type, lag_window=lag_window)

    # chọn feature có lag/roll đúng window đó + meta/calendar
    feature_cols = [
        c for c in df.columns
        if any(
            kw in c
            for kw in [
                f"lag",           # chứa tất cả lag1..lag_window
                f"roll{lag_window}_",  # chỉ rolling cho window hiện tại
                "group", "sub_group",
                "plant", "storage_location",
                "day_of_week", "is_weekend", "month", "day_of_month",
            ]
        )
    ]
    base_cols = ["node_id", "node_index", "date", "day", "split"]
    df_h7 = df[base_cols + feature_cols + ["y_h7"]].rename(columns={"y_h7": "target"})

    df_ohe = one_hot_encode_splits(df_h7, CAT_COLS)

    out_dir = PROC_DIR / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"xgboost_tabular_h{HORIZON}_lag{lag_window}_{temporal_type}.parquet"
    df_ohe.to_parquet(out_path, index=False)
    print(f"Saved XGBoost baseline to {out_path}")
    return df_ohe

def build_gru_sequence(temporal_type: str = "unit"):
    print(f"\n=== Building GRU sequence base (no graph), "
          f"H={HORIZON}, temporal_type={temporal_type} ===")

    raw_path = PROC_DIR / f"base_raw_{temporal_type}.parquet"
    if raw_path.exists():
        df = pd.read_parquet(raw_path)
        print(f"Loaded RAW base from {raw_path}")
    else:
        df = build_base_raw(temporal_type=temporal_type)

    keep_cols = [
        "node_id", "node_index", "date", "day", "split",
        "sales_order", "production", "delivery", "factory_issue",
        "day_of_week", "is_weekend", "month", "day_of_month",
        "y_h7",
        "group", "sub_group", "plant", "storage_location",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df_gru = df[keep_cols].rename(columns={"y_h7": "target"})

    df_ohe = one_hot_encode_splits(df_gru, CAT_COLS)

    out_dir = PROC_DIR / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"gru_sequence_{temporal_type}.parquet"
    df_ohe.to_parquet(out_path, index=False)
    print(f"Saved GRU base to {out_path}")
    return df_ohe

def build_baseline_full_csv_for_naive(temporal_type: str = "unit",
                                      lag_window: int = 7) -> pd.DataFrame:
    """
    Tạo file baseline FULL dạng CSV cho naive baseline (chưa one-hot).
    Input: base_full_h7_lag{lag_window}_{temporal_type}.parquet
    Output: baseline/xgboost_h7_lag{lag_window}_{temporal_type}_full.csv
    """
    print(f"\n=== Building baseline FULL CSV for naive, "
          f"H={HORIZON}, lag={lag_window}, temporal_type={temporal_type} ===")

    full_path = PROC_DIR / f"base_full_h{HORIZON}_lag{lag_window}_{temporal_type}.parquet"
    if full_path.exists():
        df = pd.read_parquet(full_path)
        print(f"Loaded FULL base from {full_path}")
    else:
        df = build_base_full(temporal_type=temporal_type, lag_window=lag_window)

    out_dir = PROC_DIR / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"xgboost_h{HORIZON}_lag{lag_window}_{temporal_type}_full.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved FULL baseline CSV for naive to {out_path}")
    return df

def main() -> None:
    temporal_types = ["unit", "weight"]

    for temporal_type in temporal_types:
        # 1. RAW một lần – dùng cho GRU + base_full
        build_base_raw(temporal_type=temporal_type)

        # 2. Baseline GRU từ RAW
        build_gru_sequence(temporal_type=temporal_type)

        # 3. FULL (lag+rolling) + baseline FULL CSV + XGBoost baseline
        for lag_window in LAG_WINDOWS:
            build_base_full(temporal_type=temporal_type, lag_window=lag_window)
            build_baseline_full_csv_for_naive(temporal_type=temporal_type, lag_window=lag_window)
            build_xgboost_tabular(temporal_type=temporal_type, lag_window=lag_window)

if __name__ == "__main__":
    main()

