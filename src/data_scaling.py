from __future__ import annotations

from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


ScalerInfo = Dict[str, Any]  # keys: "scaler", "cols", "target_idx"


def get_scaling_columns(
    df: pd.DataFrame,
    target_col: str,
    extra_exclude_cols: List[str] | None = None,
) -> List[str]:
    """
    Xác định danh sách numerical columns sẽ được scale (ngoại trừ meta + target).

    - df: dataframe đã OHE (ví dụ output của one_hot_encode_splits).
    - target_col: tên cột target (ví dụ: "target").
    - extra_exclude_cols: các cột numeric nhưng muốn exclude (nếu có).

    Trả về: list tên cột numeric dùng để scale (sẽ được scale cùng với target).
    """
    meta_cols = ["node_id", "node_index", "date", "day", "split"]
    if extra_exclude_cols is None:
        extra_exclude_cols = []

    non_scale_cols = set(meta_cols + [target_col] + list(extra_exclude_cols))

    num_cols: List[str] = []
    for c in df.columns:
        if c in non_scale_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
    return num_cols


def fit_scalers_per_node(
    df: pd.DataFrame,
    num_cols: List[str],
    target_col: str = "target",
    split_col: str = "split",
    train_split_value: str = "train",
) -> Dict[Any, ScalerInfo]:
    """
    Fit MinMaxScaler riêng cho từng node_id, chỉ dùng các mẫu thuộc train split.

    - df: dataframe đã có split, target, features (có thể đã OHE).
    - num_cols: danh sách numeric feature columns để scale.
    - target_col: tên cột target (sẽ scale chung với num_cols).
    - split_col: tên cột split (mặc định "split").
    - train_split_value: giá trị cho train split (mặc định "train").

    Trả về:
        dict[node_id] -> {
            "scaler": MinMaxScaler đã fit,
            "cols": list(num_cols + [target_col]),
            "target_idx": index của target trong "cols"
        }
    """
    df_train = df[df[split_col] == train_split_value].copy()

    if df_train.empty:
        raise ValueError("No training samples found for fitting scalers.")

    scalers: Dict[Any, ScalerInfo] = {}

    cols = list(num_cols) + [target_col]
    target_idx = len(cols) - 1

    for nid, g in df_train.groupby("node_id"):
        data = g[cols].values
        scaler = MinMaxScaler()
        scaler.fit(data)
        scalers[nid] = {
            "scaler": scaler,
            "cols": cols,
            "target_idx": target_idx,
        }

    return scalers


def apply_scalers_per_node(
    df: pd.DataFrame,
    scalers: Dict[Any, ScalerInfo],
    split_col: str = "split",
    strict: bool = False,
) -> pd.DataFrame:
    """
    Áp dụng scaler per-node cho toàn bộ df (train/val/test).

    - df: dataframe chứa node_id, split, các numeric cols cần scale.
    - scalers: dict từ fit_scalers_per_node.
    - split_col: tên cột split (để lọc/kiểm tra nếu cần), thực tế không bắt buộc.
    - strict: nếu True, raise error nếu có node_id không có scaler.

    Trả về:
        df_scaled: bản copy đã được scale các cột trong mỗi scaler["cols"].
    """
    df = df.copy()
    all_scaled: List[pd.DataFrame] = []

    for nid, g in df.groupby("node_id"):
        info = scalers.get(nid, None)
        if info is None:
            if strict:
                raise KeyError(f"No scaler found for node_id={nid}")
            # nếu không strict: giữ nguyên node này, không scale
            all_scaled.append(g)
            continue

        scaler: MinMaxScaler = info["scaler"]
        cols: List[str] = info["cols"]

        # Không phải node nào cũng có đủ cột (trường hợp filter trước đó), nên intersect
        cols_present = [c for c in cols if c in g.columns]
        # Đảm bảo target nằm trong cols_present
        if len(cols_present) != len(cols):
            # Có thể do drop cột ở ngoài, xử lý tuỳ ý
            # Ở đây raise để tránh silent bug
            missing = set(cols) - set(cols_present)
            raise KeyError(
                f"Some columns for scaling are missing in group node_id={nid}: {missing}"
            )

        data = g[cols].values
        scaled = scaler.transform(data)

        scaled_df = g.copy()
        scaled_df[cols] = scaled
        all_scaled.append(scaled_df)

    df_scaled = pd.concat(all_scaled, axis=0).sort_index()
    return df_scaled


def inverse_target_per_node(
    df_scaled: pd.DataFrame,
    y_scaled_pred: np.ndarray,
    scalers: Dict[Any, ScalerInfo],
    target_col: str = "target",
) -> np.ndarray:
    """
    Inverse transform prediction target từ scaled space về original scale, per node_id.

    - df_scaled: dataframe (cùng thứ tự với y_scaled_pred) đã scaled (chứa node_id, các cols dùng để fit scaler).
    - y_scaled_pred: numpy array predictions trong scaled space (1D).
    - scalers: dict từ fit_scalers_per_node.
    - target_col: tên cột target.

    Trả về:
        y_pred_original: numpy array cùng shape với y_scaled_pred, đã ở original scale.
    """
    if len(df_scaled) != len(y_scaled_pred):
        raise ValueError(
            f"Length mismatch: df_scaled={len(df_scaled)}, y_scaled_pred={len(y_scaled_pred)}"
        )

    y_pred_original = np.zeros_like(y_scaled_pred, dtype=float)

    # Đảm bảo index align
    df_scaled_iter = df_scaled.reset_index(drop=True)

    for i, row in df_scaled_iter.iterrows():
        nid = row["node_id"]
        info = scalers.get(nid, None)
        if info is None:
            raise KeyError(f"No scaler found for node_id={nid} during inverse transform.")

        scaler: MinMaxScaler = info["scaler"]
        cols: List[str] = info["cols"]
        target_idx: int = info["target_idx"]

        # Lấy full vector các feature + target (scaled hiện tại)
        vec = row[cols].values.astype(float)
        # Thay target bằng y_scaled_pred
        vec[target_idx] = y_scaled_pred[i]

        # Inverse transform
        inv = scaler.inverse_transform(vec.reshape(1, -1))[0]
        y_pred_original[i] = inv[target_idx]

    return y_pred_original


def scale_dataframe_per_node(
    df: pd.DataFrame,
    target_col: str = "target",
    split_col: str = "split",
    train_split_value: str = "train",
    extra_exclude_cols: List[str] | None = None,
    strict: bool = False,
) -> Tuple[pd.DataFrame, Dict[Any, ScalerInfo], List[str]]:
    """
    Convenience function:
    - Tự động detect num_cols,
    - Fit scaler per node trên train only,
    - Apply scaler lên toàn df,
    - Trả về df_scaled + scalers + num_cols (để log/debug).

    Phù hợp dùng ngay trong build_xgboost_tabular / build_gru_sequence.

    - extra_exclude_cols: nếu có cột numeric nhưng muốn exclude khỏi scaling (vd. 1 vài index feature).
    - strict: nếu True, yêu cầu mọi node_id phải có scaler (tức phải có train samples).
    """
    num_cols = get_scaling_columns(
        df=df,
        target_col=target_col,
        extra_exclude_cols=extra_exclude_cols,
    )

    scalers = fit_scalers_per_node(
        df=df,
        num_cols=num_cols,
        target_col=target_col,
        split_col=split_col,
        train_split_value=train_split_value,
    )

    df_scaled = apply_scalers_per_node(
        df=df,
        scalers=scalers,
        split_col=split_col,
        strict=strict,
    )

    return df_scaled, scalers, num_cols