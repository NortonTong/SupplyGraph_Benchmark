# train_arma.py (rolling forecast)
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROC_DIR = DATA_DIR / "processed"

ARIMA_ORDER_H1 = (2, 1, 1)
ARIMA_ORDER_H7 = (2, 1, 1)


def load_base() -> pd.DataFrame:
    path = PROC_DIR / "xgboost_base_filtered.parquet"
    df = pd.read_parquet(path)
    df = df.sort_values(["node_id", "day"]).reset_index(drop=True)
    return df


def arma_walk_forward(
    series: pd.Series,
    train_days: np.ndarray,
    test_days: np.ndarray,
    arima_order=(2, 1, 1),
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Walk-forward ARIMA:
      - history khởi tạo = dữ liệu tới max(train_days)
      - với mỗi test day t:
          * dùng toàn bộ history có day < t để fit ARIMA
          * forecast horizon bước
          * so sánh với giá trị thật tại day t
          * thêm y_true(t) vào history (giả lập online).
    series: Series indexed by day (int) với sales_order.
    """
    y_true_all, y_pred_all = [], []

    all_days = series.index.values

    # history khởi tạo: tất cả ngày <= max(train_days)
    max_train_day = int(train_days.max())
    history_days = all_days[all_days <= max_train_day].tolist()
    history_values = [series.loc[d] for d in history_days]

    # chỉ giữ test_days >= max_train_day + horizon (để có “horizon-step ahead” đúng nghĩa)
    test_days = np.array(sorted(test_days))
    valid_test_days = [d for d in test_days if d >= max_train_day + horizon]
    if len(valid_test_days) == 0:
        return np.array([]), np.array([])

    for t in valid_test_days:
        # fit ARIMA trên history
        hist_arr = np.array(history_values, dtype=float)
        if np.isnan(hist_arr).sum() > 0 or len(hist_arr) < 10:
            continue

        try:
            model = ARIMA(hist_arr, order=arima_order)
            model_fit = model.fit()
        except Exception:
            continue

        # forecast horizon-step ahead từ cuối history
        try:
            forecast = model_fit.forecast(steps=horizon)
            y_hat = forecast[-1]
        except Exception:
            continue

        # true value tại day t
        y_true = series.loc[t]
        if np.isnan(y_true):
            continue

        y_true_all.append(y_true)
        y_pred_all.append(y_hat)

        # walk-forward: thêm observation thật vào history
        history_days.append(t)
        history_values.append(y_true)

    return np.array(y_true_all), np.array(y_pred_all)


def run_arma_rolling(horizon: int = 1) -> None:
    df = load_base()

    if horizon == 1:
        arima_order = ARIMA_ORDER_H1
    elif horizon == 7:
        arima_order = ARIMA_ORDER_H7
    else:
        raise ValueError("Only support horizon 1 or 7.")

    all_y_true, all_y_pred = [], []

    node_ids = df["node_id"].unique()
    print(f"Running ARIMA rolling baseline for H={horizon} on {len(node_ids)} nodes...")

    for nid in node_ids:
        df_node = df[df["node_id"] == nid].copy()
        df_node = df_node.sort_values("day")
        days = df_node["day"].values
        series = pd.Series(df_node["sales_order"].values, index=days)

        train_days = df_node.loc[df_node["split"].isin(["train", "val"]), "day"].values
        test_days = df_node.loc[df_node["split"] == "test", "day"].values

        if len(train_days) == 0 or len(test_days) == 0:
            continue

        y_true, y_pred = arma_walk_forward(
            series=series,
            train_days=train_days,
            test_days=test_days,
            arima_order=arima_order,
            horizon=horizon,
        )

        if y_true.size == 0:
            continue

        all_y_true.append(y_true)
        all_y_pred.append(y_pred)

    if not all_y_true:
        print("No valid series for ARIMA rolling baseline.")
        return

    y_true_concat = np.concatenate(all_y_true)
    y_pred_concat = np.concatenate(all_y_pred)

    mae = mean_absolute_error(y_true_concat, y_pred_concat)
    rmse = root_mean_squared_error(y_true_concat, y_pred_concat)

    print(f"[ARIMA-rolling][H={horizon}] Test MAE={mae:.4f} RMSE={rmse:.4f}")

    out_dir = PROC_DIR / "predictions" / "arma_rolling"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"y_true": y_true_concat, "y_pred": y_pred_concat}
    ).to_csv(out_dir / f"arma_rolling_h{horizon}_test_predictions.csv", index=False)
    print(f"Saved ARIMA rolling predictions to {out_dir / f'arma_rolling_h{horizon}_test_predictions.csv'}")


def main():
    run_arma_rolling(horizon=1)
    run_arma_rolling(horizon=7)


if __name__ == "__main__":
    main()