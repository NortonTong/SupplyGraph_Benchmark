# prepare_gnn_encoded_data.py
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from config.config import (PROJECT_ROOT, DATA_DIR, PROC_DIR, LAG_WINDOWS)


OUT_DIR = PROC_DIR / "gnn"
OUT_DIR.mkdir(parents=True, exist_ok=True)

from train_xgboost import (
    load_dataset,
    one_hot_encode_train_val_test,
    prepare_features,
)


def build_encoded_table(horizon: int, lag_window: int | None = None) -> pd.DataFrame:
    if lag_window is None:
        lag_window = LAG_WINDOWS[0]

    df = load_dataset(horizon=horizon, lag_window=lag_window)

    df_train_enc, df_val_enc, df_test_enc = one_hot_encode_train_val_test(df)
    df_enc = pd.concat([df_train_enc, df_val_enc, df_test_enc], ignore_index=True)
    df_enc = df_enc.sort_values(["date", "node_id"]).reset_index(drop=True)
    return df_enc

def load_base_with_labels() -> pd.DataFrame:
    """
    Load xgboost_base_filtered.parquet to get y_h1, y_h7.
    """
    path = PROC_DIR / "xgboost_base_filtered.parquet"
    df_base = pd.read_parquet(path)
    # Keep only keys + labels to merge later
    df_base = df_base[["node_id", "node_index", "date", "day", "y_h1", "y_h7"]]
    return df_base

def build_day_and_nodeindex(df: pd.DataFrame):
    days = np.sort(df["day"].dropna().unique())
    node_indices = np.sort(df["node_index"].dropna().unique())
    day2idx = {int(d): i for i, d in enumerate(days)}
    nodeindex2pos = {int(ni): i for i, ni in enumerate(node_indices)}
    return days, node_indices, day2idx, nodeindex2pos

def build_tensors_from_encoded(df_enc: pd.DataFrame, df_base: pd.DataFrame):
    # merge label từ base vào bảng encoded (theo node_id + date)
    df_merged = df_enc.merge(
        df_base,
        on=["node_id", "node_index", "date", "day"],
        how="left",
        validate="1:1",
    )

    # BỎ hết dòng không có node_index hoặc day
    df_merged = df_merged[ df_merged["node_index"].notna() & df_merged["day"].notna() ].copy()
    # ép node_index về int (baseline cũng dùng int)
    df_merged["node_index"] = df_merged["node_index"].astype(int)
    df_merged["day"] = df_merged["day"].astype(int)

    days, node_indices, day2idx, nodeindex2pos = build_day_and_nodeindex(df_merged)
    # features: giống XGBoost – dùng prepare_features để drop target/split/id/date/day
    X_tmp, _ = prepare_features(df_merged)
    feature_cols = X_tmp.columns.tolist()

    T = len(days)
    N = len(node_indices)
    Fdim = len(feature_cols)

    X = np.zeros((T, N, Fdim), dtype=np.float32)
    Y_h1 = np.full((T, N), np.nan, dtype=np.float32)
    Y_h7 = np.full((T, N), np.nan, dtype=np.float32)

    for _, row in df_merged.iterrows():
        di = day2idx[row["day"]]
        pos = nodeindex2pos[row["node_index"]]
        X[di, pos, :] = row[feature_cols].values.astype(np.float32)
        Y_h1[di, pos] = row["y_h1"]
        Y_h7[di, pos] = row["y_h7"]

    return (
        torch.from_numpy(X),
        torch.from_numpy(Y_h1),
        torch.from_numpy(Y_h7),
        torch.tensor(days, dtype=torch.long),
        torch.tensor(node_indices, dtype=torch.long),
        feature_cols,
    )

def get_day_split_from_encoded(df_enc: pd.DataFrame):
    day_split = (
        df_enc.groupby("day")["split"]
        .first()
        .sort_index()
    )
    return day_split

def main():
    df_enc = build_encoded_table(horizon=1)
    df_base = load_base_with_labels()

    X, Y_h1, Y_h7, days, node_index_arr, feature_cols = build_tensors_from_encoded(df_enc, df_base)
    day_split = get_day_split_from_encoded(df_enc)

    split_arr = np.array([day_split.loc[int(d)] for d in days.numpy()])

    pkg = {
        "X": X,                 # [T, N, F]
        "Y_h1": Y_h1,           # [T, N]
        "Y_h7": Y_h7,           # [T, N]
        "days": days,           # [T]
        "node_index": node_index_arr,  # [N]
        "feature_cols": feature_cols,
        "split": split_arr,     # [T], 'train'/'val'/'test'
    }

    out_path = OUT_DIR / "gnn_data_encoded.pt"
    torch.save(pkg, out_path)
    print(f"Saved encoded GNN data to {out_path}")

if __name__ == "__main__":
    main()