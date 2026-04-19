# prepare_gnn_encoded_data.py

import numpy as np
import pandas as pd
import torch
from pathlib import Path

from config.config import (
    PROC_DIR,
    LAG_WINDOWS,     # [7, 14]
    TEMPORAL_TYPE,  # ["unit", "weight"]
)

OUT_DIR = PROC_DIR / "gnn"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_xgboost_tabular(
    temporal_type: str,
    lag_window: int,
    horizon: int = 7,
) -> pd.DataFrame:
    """
    Load file XGBoost tabular baseline đã one-hot:
      baseline/xgboost_tabular_h{horizon}_lag{lag_window}_{temporal_type}.parquet

    File này (theo build_xgboost_tabular) chứa:
      - node_id, node_index, date, day, split
      - features: lag*, roll{lag_window}_*, meta/calendar OHE
      - target  (== y_h7)
    """
    base_dir = PROC_DIR / "baseline"
    path = base_dir / f"xgboost_tabular_h{horizon}_lag{lag_window}_{temporal_type}.parquet"
    print(f"[ENCODE] Loading XGBoost tabular from {path}")
    df = pd.read_parquet(path)

    required_cols = ["node_id", "node_index", "date", "day", "split", "target"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"[ENCODE] XGBoost tabular file {path} missing columns: {missing}")

    # chuẩn hóa type
    df["date"] = pd.to_datetime(df["date"])
    df["day"] = df["day"].astype(int)
    df["node_index"] = df["node_index"].astype(int)

    # sort cho chắc
    df = df.sort_values(["date", "node_id"]).drop_duplicates(
        subset=["date", "node_id"], keep="last"
    )
    return df


def build_day_and_nodeindex(df: pd.DataFrame):
    """
    Từ df (đã có node_index + day), suy ra:
      - days: sorted unique day
      - node_indices: sorted unique node_index
      - day2idx: map day -> 0..T-1
      - nodeindex2pos: map node_index -> 0..N-1
    """
    days = np.sort(df["day"].dropna().unique())
    node_indices = np.sort(df["node_index"].dropna().unique())

    day2idx = {int(d): i for i, d in enumerate(days)}
    nodeindex2pos = {int(ni): i for i, ni in enumerate(node_indices)}

    print(f"[ENCODE] #days={len(days)}, #nodes={len(node_indices)}")
    return days, node_indices, day2idx, nodeindex2pos


def prepare_features_for_gnn(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Lấy đúng feature set mà XGBoost dùng:
      - bỏ các cột id/split/label: node_id, node_index, date, day, split, target
      - giữ lại toàn bộ lag/rolling + OHE meta/calendar.

    Trả về:
      - df_feat: DataFrame chỉ chứa feature
      - feature_cols: list tên feature
    """
    drop_cols = ["node_id", "node_index", "date", "day", "split", "target"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    df_feat = df[feature_cols].copy()
    return df_feat, feature_cols


def build_tensors_from_tabular(df: pd.DataFrame):
    """
    Từ df (XGB tabular đã OHE), build:
      X: [T, N, F]
      Y: [T, N]
      days: [T]
      node_index_arr: [N]
      feature_cols: list[str]
    """
    # mapping day/node_index
    days, node_indices, day2idx, nodeindex2pos = build_day_and_nodeindex(df)

    df_feat, feature_cols = prepare_features_for_gnn(df)

    T = len(days)
    N = len(node_indices)
    Fdim = len(feature_cols)

    print(
        f"[ENCODE] Building tensors with shape X[{T}, {N}, {Fdim}], "
        f"Y[{T}, {N}]"
    )

    X = np.zeros((T, N, Fdim), dtype=np.float32)
    Y = np.full((T, N), np.nan, dtype=np.float32)

    # ensure order consistent
    df_iter = df[["day", "node_index", "target"]].join(df_feat)

    for _, row in df_iter.iterrows():
        di = day2idx[int(row["day"])]
        pos = nodeindex2pos[int(row["node_index"])]
        X[di, pos, :] = row[feature_cols].values.astype(np.float32)
        Y[di, pos] = row["target"]

    X_t = torch.from_numpy(X)
    Y_t = torch.from_numpy(Y)
    days_t = torch.tensor(days, dtype=torch.long)
    node_index_arr_t = torch.tensor(node_indices, dtype=torch.long)

    return X_t, Y_t, days_t, node_index_arr_t, feature_cols


def get_day_split_from_tabular(df: pd.DataFrame) -> np.ndarray:
    """
    Từ df (có cột day, split), suy ra split per day.
    Giả định: mỗi day có một split duy nhất ('train'/'val'/'test').
    """
    day_split = (
        df.groupby("day")["split"]
        .first()
        .sort_index()
    )
    print("[ENCODE] day_split value_counts:", day_split.value_counts(dropna=False))

    # day trong tensor days là sorted, nên ta sẽ map lại theo đó
    # việc lấy về np.ndarray [T] làm ở hàm main.
    return day_split


def main():
    """
    Sinh ra:
      PROC_DIR / "gnn" / f"gnn_data_encoded_{temporal_type}_lag{lag_window}.pt"

    Nội dung mỗi file:
      - "X": [T, N, F]
      - "Y": [T, N] (target = y_h7)
      - "days": [T]
      - "node_index": [N]
      - "feature_cols": list[str]
      - "split": [T] (chuỗi 'train'/'val'/'test')
      - "lag_window": int
      - "horizon": int (7)
    """

    horizon = 7

    for temporal_type in TEMPORAL_TYPE:
        for lag_window in LAG_WINDOWS:
            print(
                f"\n=== Building encoded GNN data: temporal_type={temporal_type}, "
                f"horizon={horizon}, lag_window={lag_window} ==="
            )

            # 1) Load XGB tabular baseline (đã OHE)
            df_tab = load_xgboost_tabular(
                temporal_type=temporal_type,
                lag_window=lag_window,
                horizon=horizon,
            )

            # 2) Build tensor X, Y
            X, Y, days, node_index_arr, feature_cols = build_tensors_from_tabular(df_tab)

            # 3) split per day
            day_split = get_day_split_from_tabular(df_tab)
            split_arr = np.array([day_split.loc[int(d)] for d in days.numpy()])

            # 4) đóng gói và lưu
            pkg = {
                "X": X,                   # [T, N, F]
                "Y": Y,                   # [T, N]
                "days": days,             # [T]
                "node_index": node_index_arr,  # [N]
                "feature_cols": feature_cols,
                "split": split_arr,       # [T], 'train'/'val'/'test'
                "lag_window": lag_window,
                "horizon": horizon,
            }

            out_path = OUT_DIR / f"gnn_data_encoded_{temporal_type}_lag{lag_window}.pt"
            torch.save(pkg, out_path)
            print(f"[ENCODE] Saved encoded GNN data to {out_path}")


if __name__ == "__main__":
    main()