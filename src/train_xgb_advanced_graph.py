import pandas as pd
import numpy as np
from pathlib import Path

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import matplotlib.pyplot as plt

from config.config import PROC_DIR, DEFAULT_EXPERIMENTS


RUN_SUMMARY: list[dict] = []


# =========================
# Metrics
# =========================

def mape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > eps
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def smape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) + eps
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0)


# =========================
# Load tabular with graph features
# =========================

def load_tabular_graphfeat(
    horizon: int,
    temporal_type: str,
    lag_window: int,
    graph_type: str,
) -> pd.DataFrame:
    """
    graph_type in {"projected", "homo5", "hetero5"}.

    Tương ứng file:
    xgboost_tabular_graphfeat_{graph_type}_h{H}_lag{L}_{temporal_type}.parquet
    """
    fname = f"xgboost_tabular_graphfeat_{graph_type}_h{horizon}_lag{lag_window}_{temporal_type}.parquet"
    path = PROC_DIR / "baseline" / "xgboost" / fname
    print(f"Loading tabular + graph features from {path}")
    return pd.read_parquet(path)


def split_train_val_test(df: pd.DataFrame):
    df_train = df[df["split"] == "train"].copy()
    df_val   = df[df["split"] == "val"].copy()
    df_test  = df[df["split"] == "test"].copy()
    return df_train, df_val, df_test


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = df["target"].astype(float)
    drop_cols = [
        "target",
        "split",
        "node_id",
        "node_index",
        "date",
        "day",
    ]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return X, y


# =========================
# Plot predictions per product
# =========================

def plot_predictions_per_product(
    df_test: pd.DataFrame,
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
    out_dir: Path,
    lag_window: int,
    temporal_type: str,
    horizon: int,
    graph_type: str,
    max_plots: int | None = None,
) -> None:
    """
    Vẽ y_true vs y_pred theo ngày cho từng sản phẩm (node_id) trên test split.
    Lưu 1 file .png / sản phẩm vào out_dir, prefix xgb_tabular_graphfeat_*.
    """
    df_plot = df_test[["node_id", "date"]].copy()
    df_plot["y_true"] = np.asarray(y_test, dtype=float)
    df_plot["y_pred"] = np.asarray(y_test_pred, dtype=float)

    unique_nodes = df_plot["node_id"].unique()
    if max_plots is not None:
        unique_nodes = unique_nodes[:max_plots]

    for node in unique_nodes:
        sub = df_plot[df_plot["node_id"] == node].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("date")

        plt.figure(figsize=(10, 4))
        plt.plot(sub["date"], sub["y_true"], label="True", marker="o", linewidth=1)
        plt.plot(sub["date"], sub["y_pred"], label="Pred", marker="x", linewidth=1)
        plt.title(
            f"XGB Tabular + {graph_type} (H{horizon}, lag={lag_window}, {temporal_type}) - node_id={node}"
        )
        plt.xlabel("Date")
        plt.ylabel("Sales order")
        plt.legend()
        plt.tight_layout()

        fname = out_dir / f"xgb_tabular_graphfeat_{graph_type}_h{horizon}_lag{lag_window}_{temporal_type}_node_{node}.png"
        plt.savefig(fname, dpi=150)
        plt.close()


# =========================
# Train 1 cấu hình (tabular + 1 graph_type)
# =========================

def train_xgb_tabular_graphfeat(
    horizon: int,
    temporal_type: str,
    lag_window: int,
    graph_type: str,
) -> None:
    """
    graph_type: 'projected' | 'homo5' | 'hetero5'
    """
    target_type = "raw"
    tag = f"xgb_tabular_graphfeat_{graph_type}_h{horizon}_lag{lag_window}_{temporal_type}"

    print(
        f"\n=== Training XGBoost TABULAR + {graph_type} graph features "
        f"(H{horizon}, lag={lag_window}, temporal_type={temporal_type}, target_type={target_type}) ==="
    )

    # 1) Load tabular + graph features
    df_base = load_tabular_graphfeat(
        horizon=horizon,
        temporal_type=temporal_type,
        lag_window=lag_window,
        graph_type=graph_type,
    )

    print(
        f"H{horizon} lag{lag_window} {temporal_type} [{graph_type}]: "
        f"rows={len(df_base)}, unique(node,date)={df_base[['node_id','date']].drop_duplicates().shape[0]}"
    )

    # 2) Split train/val/test
    df_train, df_val, df_test = split_train_val_test(df_base)
    print("Splits rows:", len(df_train), len(df_val), len(df_test))

    # 3) Features & targets
    X_train, y_train_raw = prepare_features(df_train)
    X_val,   y_val_raw   = prepare_features(df_val)
    X_test,  y_test_raw  = prepare_features(df_test)

    y_train = y_train_raw.values
    y_val   = y_val_raw.values

    feature_names = list(X_train.columns)
    print(
        f"\n[H{horizon}][lag{lag_window}][{graph_type}] Using {len(feature_names)} features:"
    )
    print(feature_names[:50])  # in thử 50 feature đầu cho gọn
    print(f"Train samples: {X_train.shape[0]}")
    print(f"Val   samples: {X_val.shape[0]}")
    print(f"Test  samples: {X_test.shape[0]}")

    # 4) XGBoost model
    model = XGBRegressor(
        n_estimators=5000,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        eval_metric="rmse",
        early_stopping_rounds=100,
    )

    eval_set = [(X_train, y_train), (X_val, y_val)]

    model.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        verbose=False,
    )

    evals_result = model.evals_result()
    train_rmse_hist = evals_result["validation_0"]["rmse"]
    val_rmse_hist   = evals_result["validation_1"]["rmse"]

    # 5) Learning curve
    plt.figure(figsize=(8, 5))
    plt.plot(train_rmse_hist, label="Train RMSE")
    plt.plot(val_rmse_hist, label="Val RMSE")
    plt.axvline(model.best_iteration, color="red", linestyle="--", label="Best iter")
    plt.xlabel("Boosting round")
    plt.ylabel("RMSE")
    plt.title(
        f"Learning curve H{horizon} lag{lag_window} - {graph_type} - {temporal_type}"
    )
    plt.legend()
    plt.tight_layout()

    out_curve = (
        PROC_DIR
        / "predictions"
        / "baseline_7"
        / "plots_learning_curves"
        / f"learning_curve_{graph_type}_h{horizon}_lag{lag_window}_{temporal_type}.png"
    )
    out_curve.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_curve, dpi=150)
    plt.close()
    print(f"Saved learning curve to {out_curve}")

    # 6) Metrics (raw scale, clip output >= 0)
    # Train
    y_train_pred = model.predict(X_train)
    y_train_pred = np.clip(y_train_pred, 0.0, None)  # không cho âm
    y_train_true = y_train_raw.values

    mae_train   = mean_absolute_error(y_train_true, y_train_pred)
    rmse_train  = root_mean_squared_error(y_train_true, y_train_pred)
    mape_train  = mape(y_train_true, y_train_pred)
    smape_train = smape(y_train_true, y_train_pred)

    # Validation
    y_val_pred = model.predict(X_val)
    y_val_pred = np.clip(y_val_pred, 0.0, None)
    y_val_true = y_val_raw.values

    mae_val   = mean_absolute_error(y_val_true, y_val_pred)
    rmse_val  = root_mean_squared_error(y_val_true, y_val_pred)
    mape_val  = mape(y_val_true, y_val_pred)
    smape_val = smape(y_val_true, y_val_pred)

    # Test
    y_test_pred = model.predict(X_test)
    y_test_pred = np.clip(y_test_pred, 0.0, None)
    y_test_true = y_test_raw.values

    mae_test   = mean_absolute_error(y_test_true, y_test_pred)
    rmse_test  = root_mean_squared_error(y_test_true, y_test_pred)
    mape_test  = mape(y_test_true, y_test_pred)
    smape_test = smape(y_test_true, y_test_pred)

    print(f"\n[H{horizon}][lag{lag_window}][{graph_type}] Train:")
    print(f"  MAE   : {mae_train:.4f}")
    print(f"  RMSE  : {rmse_train:.4f}")
    print(f"  MAPE  : {mape_train:.4f}")
    print(f"  sMAPE : {smape_train:.4f}")

    print(f"\n[H{horizon}][lag{lag_window}][{graph_type}] Val:")
    print(f"  MAE   : {mae_val:.4f}")
    print(f"  RMSE  : {rmse_val:.4f}")
    print(f"  MAPE  : {mape_val:.4f}")
    print(f"  sMAPE : {smape_val:.4f}")

    print(f"\n[H{horizon}][lag{lag_window}][{graph_type}] Test:")
    print(f"  MAE   : {mae_test:.4f}")
    print(f"  RMSE  : {rmse_test:.4f}")
    print(f"  MAPE  : {mape_test:.4f}")
    print(f"  sMAPE : {smape_test:.4f}")

    # 7) Lưu test predictions + plots
    base_pred_dir = PROC_DIR / "predictions" / "baseline_7"
    out_dir_csv = base_pred_dir / "csv" / f"{temporal_type}"
    plot_folder = f"{graph_type}_h{horizon}_lag{lag_window}"
    out_dir_plot = base_pred_dir / "plots_xgb_tabular" / plot_folder / f"{temporal_type}"
    out_dir_csv.mkdir(parents=True, exist_ok=True)
    out_dir_plot.mkdir(parents=True, exist_ok=True)

    df_test_pred = pd.DataFrame(
        {
            "node_id": df_test["node_id"].values,
            "date": df_test["date"].values,
            "y_true": y_test_true,
            "y_pred": y_test_pred,
        }
    )
    out_pred_file = out_dir_csv / f"xgb_tabular_graphfeat_{graph_type}_h{horizon}_lag{lag_window}_{temporal_type}_test_predictions.csv"
    df_test_pred.to_csv(out_pred_file, index=False)
    print(f"Saved test predictions to {out_pred_file}")

    plot_predictions_per_product(
        df_test=df_test,
        y_test=y_test_true,
        y_test_pred=y_test_pred,
        out_dir=out_dir_plot,
        lag_window=lag_window,
        temporal_type=temporal_type,
        horizon=horizon,
        graph_type=graph_type,
        max_plots=None,
    )
    print(f"Saved per-product prediction plots to {out_dir_plot}")

    RUN_SUMMARY.append(
        {
            "temporal_type": temporal_type,
            "lag_window": lag_window,
            "horizon": horizon,
            "graph_type": graph_type,
            "variant": f"xgb_tabular_graphfeat_{graph_type}_{target_type}",
            "tag": tag,
            "target_type": target_type,
            "n_features": X_train.shape[1],
            "MAE_train": mae_train,
            "RMSE_train": rmse_train,
            "MAPE_train": mape_train,
            "sMAPE_train": smape_train,
            "MAE_val": mae_val,
            "RMSE_val": rmse_val,
            "MAPE_val": mape_val,
            "sMAPE_val": smape_val,
            "MAE_test": mae_test,
            "RMSE_test": rmse_test,
            "MAPE_test": mape_test,
            "sMAPE_test": smape_test,
        }
    )


# =========================
# Main: chạy hết 3 graph types × tất cả lag
# =========================

def main():
    global RUN_SUMMARY
    RUN_SUMMARY = []

    graph_types = ["projected", "homo5", "hetero5"]

    for exp in DEFAULT_EXPERIMENTS:
        temporal_type = exp.temporal_type
        # theo code baseline của bạn: HORIZON = list(exp.horizons)[0]
        horizons = list(exp.horizons)
        if not horizons:
            continue
        horizon = horizons[0]

        lag_windows = list(exp.gru_seq_lengths)  # cùng cách lấy LAG_WINDOWS như bạn

        print(f"\n====== EXP: temporal_type={temporal_type}, H={horizon}, lags={lag_windows} ======")

        for lag_window in lag_windows:
            for gtype in graph_types:
                try:
                    train_xgb_tabular_graphfeat(
                        horizon=horizon,
                        temporal_type=temporal_type,
                        lag_window=lag_window,
                        graph_type=gtype,
                    )
                except FileNotFoundError:
                    print(
                        f"[WARN] File for {gtype}, H{horizon}, lag{lag_window}, {temporal_type} không tồn tại, skip."
                    )

    if RUN_SUMMARY:
        df_sum = pd.DataFrame(RUN_SUMMARY)
        print("\n=== XGB Tabular + Graph Features summary ===")
        df_sum = df_sum.sort_values(
            ["temporal_type", "graph_type", "lag_window", "horizon"]
        )
        print(
            df_sum[
                [
                    "temporal_type",
                    "graph_type",
                    "lag_window",
                    "horizon",
                    "n_features",
                    "MAE_train",
                    "RMSE_train",
                    "MAE_val",
                    "RMSE_val",
                    "MAE_test",
                    "RMSE_test",
                ]
            ]
        )

        out_path = (
            PROC_DIR
            / "predictions"
            / "baseline_7"
            / "summary_xgb_tabular_graphfeat_raw_targets.csv"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_sum.to_csv(out_path, index=False)
        print(f"\nSaved XGB + graph features summary to {out_path}")


if __name__ == "__main__":
    main()