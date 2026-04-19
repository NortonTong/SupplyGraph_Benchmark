from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from config.config import PROC_DIR

NAIVE_DIR = PROC_DIR / "predictions_naive"
NAIVE_DIR.mkdir(parents=True, exist_ok=True)

RUN_SUMMARY: list[dict] = []


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = np.abs(y_true) > eps
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.abs(y_true) + np.abs(y_pred) + eps
    return 100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / denom)


def load_baseline_dataset(horizon: int, lag_window: int, temporal_type: str = "unit") -> pd.DataFrame:
    path = PROC_DIR / "baseline" / f"xgboost_h{horizon}_lag{lag_window}_{temporal_type}_full.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["node_id", "day"]).reset_index(drop=True)
    return df


def save_and_eval_naive_last_t0(
    df_test: pd.DataFrame,
    horizon: int,
    lag_window: int,
    temporal_type: str,
) -> Tuple[float, float, float, float, Path]:
    """
    Tính MAE, RMSE, MAPE, sMAPE cho naive_last_t0 và lưu CSV:
      columns: node_id, date, day, y_true, y_pred
    """
    y_true = df_test["y_h7"].values   # label trong base_full
    y_pred = df_test["y_pred_last_t0"].values

    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mape_val = mape(y_true, y_pred)
    smape_val = smape(y_true, y_pred)

    print(
        f"[naive_last_t0][H{horizon}][lag{lag_window}][{temporal_type}] "
        f"MAE={mae:.4f} RMSE={rmse:.4f} MAPE={mape_val:.4f} sMAPE={smape_val:.4f}"
    )

    out_dir = NAIVE_DIR / f"naive_last_t0_{temporal_type}" / f"lag{lag_window}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"naive_last_t0_h{horizon}_test_predictions.csv"

    out_df = df_test[["node_id", "date", "day", "y_h7", "y_pred_last_t0"]].copy()
    out_df = out_df.rename(columns={"y_h7": "y_true", "y_pred_last_t0": "y_pred"})
    out_df.to_csv(out_path, index=False)

    # append vào RUN_SUMMARY
    RUN_SUMMARY.append(
        {
            "temporal_type": temporal_type,
            "lag_window": lag_window,
            "horizon": horizon,
            "model_family": "naive_last_t0",
            "graph_type": "no_graph",
            "variant": "naive_last_t0",
            "tag": f"naive_last_t0_h{horizon}_lag{lag_window}_{temporal_type}",
            "edge_view": None,
            "target_transform": "raw",
            "clip_pred": False,
            "use_softplus_output": False,
            "MAE_test": mae,
            "RMSE_test": rmse,
            "MAPE_test": mape_val,
            "sMAPE_test": smape_val,
        }
    )

    return mae, rmse, mape_val, smape_val, out_path


def eval_naive_last_at_t0(horizon: int, lag_window: int, temporal_type: str = "unit"):
    """
    Naive: y_hat(t0+H) = last value tại t0.
    Với pipeline hiện tại:
    - base_full có cột sales_order là demand tại t0.
    - Label y_h7 = sales_order tại t0+7.
    => dùng 'sales_order' làm y_pred_last_t0.
    """
    df = load_baseline_dataset(horizon, lag_window, temporal_type)
    df = df.sort_values(["node_id", "day"]).reset_index(drop=True)

    if "sales_order" not in df.columns:
        raise ValueError("Expected 'sales_order' column in FULL baseline dataset.")

    df["last_value_t0"] = df["sales_order"]

    df_test = df[df["split"] == "test"].copy()
    df_train = df[df["split"] == "train"].copy()

    # fallback nếu thiếu sales_order
    mean_per_node = df_train.groupby("node_id")["y_h7"].mean()
    global_mean = df_train["y_h7"].mean()

    df_test = df_test.merge(
        mean_per_node.rename("train_mean_node"),
        on="node_id",
        how="left",
    )

    df_test["y_pred_last_t0"] = df_test["last_value_t0"]
    df_test["y_pred_last_t0"] = df_test["y_pred_last_t0"].fillna(df_test["train_mean_node"])
    df_test["y_pred_last_t0"] = df_test["y_pred_last_t0"].fillna(global_mean)

    return save_and_eval_naive_last_t0(df_test, horizon, lag_window, temporal_type)


def run_naive_last_t0():
    global RUN_SUMMARY
    RUN_SUMMARY = []

    horizons = [7]
    lag_windows = [7, 14]
    temporal_types = ["unit", "weight"]

    for h in horizons:
        for lag_w in lag_windows:
            for tt in temporal_types:
                print("\n==============================")
                print(f"Naive LAST at t0 for H{h}, lag{lag_w}, {tt}")
                print("==============================")
                eval_naive_last_at_t0(h, lag_w, tt)

    # lưu summary
    if RUN_SUMMARY:
        df_sum = pd.DataFrame(RUN_SUMMARY)
        df_sum = df_sum.sort_values(
            ["temporal_type", "lag_window", "horizon"]
        )
        out_path = NAIVE_DIR / "summary_naive_last_t0.csv"
        df_sum.to_csv(out_path, index=False)
        print(f"\nSaved naive_last_t0 summary to {out_path}")
        print(df_sum)


if __name__ == "__main__":
    run_naive_last_t0()