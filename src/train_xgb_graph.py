import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Literal

from config.config import PROC_DIR, DEFAULT_EXPERIMENTS

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 0)


RUN_SUMMARY: list[dict] = []


def get_experiment_params():
    temporal_types = sorted({exp.temporal_type for exp in DEFAULT_EXPERIMENTS})
    horizons = sorted({h for exp in DEFAULT_EXPERIMENTS for h in exp.horizons})
    lag_windows = sorted({L for exp in DEFAULT_EXPERIMENTS for L in exp.lag_windows})
    if len(horizons) != 1:
        raise ValueError(f"[train_xgb_graph] Expected a single horizon, got {horizons}")
    return temporal_types, horizons[0], lag_windows


TEMPORAL_TYPES, HORIZON, LAG_WINDOWS = get_experiment_params()
GRAPH_MODES: list[Literal["proj", "homo", "hetero"]] = ["proj", "homo", "hetero"]


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




def load_tabular_graph_baseline(
    temporal_type: str = "unit",
    lag_window: int = 7,
    graph_mode: Literal["proj", "homo", "hetero"] = "proj",
) -> pd.DataFrame:
    base_dir = PROC_DIR / "baseline" / "xgb_graph"

    if graph_mode == "proj":
        fname = f"xgboost_tabular_graph_projected_h{HORIZON}_lag{lag_window}_{temporal_type}.parquet"
    elif graph_mode == "homo":
        fname = f"xgboost_tabular_graph_homo5_h{HORIZON}_lag{lag_window}_{temporal_type}.parquet"
    elif graph_mode == "hetero":
        fname = f"xgboost_tabular_graph_hetero5_h{HORIZON}_lag{lag_window}_{temporal_type}.parquet"
    else:
        raise ValueError(f"Unknown graph_mode={graph_mode}")

    path = base_dir / fname
    print(f"[Baseline3] Loading XGB+graph tabular from {path}")
    return pd.read_parquet(path)


# =========================
# Features
# =========================


def split_train_val_test(df: pd.DataFrame):
    df_train = df[df["split"] == "train"].copy()
    df_val = df[df["split"] == "val"].copy()
    df_test = df[df["split"] == "test"].copy()
    return df_train, df_val, df_test


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    target = df["target"].astype(float)
    drop_cols = [
        "target",
        "split",
        "node_id",
        "node_index",
        "date",
        "day",
    ]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].copy()
    return X, target


# =========================
# Plot predictions (per product)
# =========================


def plot_predictions_per_product(
    df_test: pd.DataFrame,
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
    out_dir: Path,
    lag_window: int,
    graph_mode: str,
    temporal_type: str,
    max_plots: int | None = None,
) -> None:
    """
    Vẽ y_true vs y_pred theo ngày cho từng sản phẩm (node_id) trên test split.
    Lưu 1 file .png / sản phẩm vào out_dir, prefix xgb_graph_*.
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
            f"Baseline 3 - XGB + Graph ({graph_mode}, H={HORIZON}, lag={lag_window}, {temporal_type}) - node_id={node}"
        )
        plt.xlabel("Date")
        plt.ylabel("Sales order")
        plt.legend()
        plt.tight_layout()

        fname = (
            out_dir
            / f"xgb_graph_{graph_mode}_h{HORIZON}_lag{lag_window}_node_{node}_{temporal_type}.png"
        )
        plt.savefig(fname, dpi=150)
        plt.close()


# =========================
# Train 1 cấu hình
# =========================


def train_xgb_graph_baseline(
    temporal_type: str = "unit",
    lag_window: int = 7,
    graph_mode: Literal["proj", "homo", "hetero"] = "proj",
    tag: str | None = None,
) -> None:
    target_type = "raw"  # baseline 3: dùng target raw, align với baseline 1 & GRU

    if tag is None:
        tag = (
            f"baseline3_xgb_graph_{graph_mode}_{target_type}"
            f"_lag{lag_window}_{temporal_type}"
        )

    print(
        f"\n=== Training XGBoost GRAPH baseline (Baseline 3) "
        f"H{HORIZON}, lag={lag_window}, temporal_type={temporal_type}, "
        f"graph_mode={graph_mode}, target_type={target_type}, tag={tag} ==="
    )

    # 1) Load tabular + graph baseline (precomputed)
    df_base = load_tabular_graph_baseline(
        temporal_type=temporal_type,
        lag_window=lag_window,
        graph_mode=graph_mode,
    )
    df_base["node_id"] = df_base["node_id"].astype(str)

    print(
        f"H{HORIZON} tabular+graph lag{lag_window} {temporal_type}: "
        f"df rows={len(df_base)}, "
        f"unique(node,date)={df_base[['node_id','date']].drop_duplicates().shape[0]}"
    )

    # 2) Split train/val/test
    df_train, df_val, df_test = split_train_val_test(df_base)
    print("Splits rows:", len(df_train), len(df_val), len(df_test))

    # 3) Features & targets (raw y)
    X_train, y_train = prepare_features(df_train)
    X_val, y_val = prepare_features(df_val)
    X_test, y_test = prepare_features(df_test)

    feature_names = list(X_train.columns)
    print(
        f"\n[H{HORIZON}][lag{lag_window}][{tag}] Using {len(feature_names)} features (tabular + graph):"
    )
    print(feature_names)
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
    val_rmse_hist = evals_result["validation_1"]["rmse"]

    # 5) Learning curve
    plt.figure(figsize=(8, 5))
    plt.plot(train_rmse_hist, label="Train RMSE (target scale)")
    plt.plot(val_rmse_hist, label="Val RMSE (target scale)")
    plt.axvline(
        model.best_iteration, color="red", linestyle="--", label="Best iter"
    )
    plt.xlabel("Boosting round")
    plt.ylabel("RMSE")
    plt.title(
        f"Learning curve H{HORIZON} lag{lag_window} - {tag} - {temporal_type}"
    )
    plt.legend()
    plt.tight_layout()

    out_curve = (
        PROC_DIR
        / "predictions"
        / "baseline_3"
        / "plots_learning_curves"
        / graph_mode
        / f"learning_curve_h{HORIZON}_lag{lag_window}_{target_type}_{tag}_{temporal_type}.png"
    )
    out_curve.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_curve, dpi=150)
    plt.close()

    print(f"Saved learning curve to {out_curve}")

    # 6) Metrics trên scale gốc (raw)
    # Train
    y_train_pred = model.predict(X_train)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    rmse_train = root_mean_squared_error(y_train, y_train_pred)
    mape_train = mape(y_train, y_train_pred)
    smape_train = smape(y_train, y_train_pred)

    # Validation
    y_val_pred = model.predict(X_val)
    mae_val = mean_absolute_error(y_val, y_val_pred)
    rmse_val = root_mean_squared_error(y_val, y_val_pred)
    mape_val = mape(y_val, y_val_pred)
    smape_val = smape(y_val, y_val_pred)

    # Test
    y_test_pred = model.predict(X_test)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    rmse_test = root_mean_squared_error(y_test, y_test_pred)
    mape_test = mape(y_test, y_test_pred)
    smape_test = smape(y_test, y_test_pred)

    print(f"\n[H{HORIZON}][lag{lag_window}][{graph_mode}][{target_type}][{tag}] Train:")
    print(f"  MAE   : {mae_train:.4f}")
    print(f"  RMSE  : {rmse_train:.4f}")
    print(f"  MAPE  : {mape_train:.4f}")
    print(f"  sMAPE : {smape_train:.4f}")

    print(
        f"\n[H{HORIZON}][lag{lag_window}][{graph_mode}][{target_type}][{tag}] Validation:"
    )
    print(f"  MAE   : {mae_val:.4f}")
    print(f"  RMSE  : {rmse_val:.4f}")
    print(f"  MAPE  : {mape_val:.4f}")
    print(f"  sMAPE : {smape_val:.4f}")

    print(f"\n[H{HORIZON}][lag{lag_window}][{graph_mode}][{target_type}][{tag}] Test:")
    print(f"  MAE   : {mae_test:.4f}")
    print(f"  RMSE  : {rmse_test:.4f}")
    print(f"  MAPE  : {mape_test:.4f}")
    print(f"  sMAPE : {smape_test:.4f}")

    RUN_SUMMARY.append(
        {
            "temporal_type": temporal_type,
            "graph_mode": graph_mode,
            "lag_window": lag_window,
            "horizon": HORIZON,
            "variant": f"baseline_3_xgb_graph_{graph_mode}_{target_type}",
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

    # 7) Save test predictions & plots per product
    base_pred_dir = PROC_DIR / "predictions" / "baseline_3"
    out_dir_csv = base_pred_dir / "csv" / temporal_type / graph_mode
    plot_folder = f"{target_type}_lag{lag_window}"
    out_dir_plot = (
        base_pred_dir / "plots_xgb_graph" / graph_mode / plot_folder / temporal_type
    )
    out_dir_csv.mkdir(parents=True, exist_ok=True)
    out_dir_plot.mkdir(parents=True, exist_ok=True)

    df_test_pred = pd.DataFrame(
        {
            "node_id": df_test["node_id"].values,
            "date": df_test["date"].values,
            "y_true": y_test,
            "y_pred": y_test_pred,
        }
    )
    out_pred_file = (
        out_dir_csv
        / f"xgb_graph_{graph_mode}_h{HORIZON}_lag{lag_window}_{target_type}_test_predictions_{temporal_type}.csv"
    )
    df_test_pred.to_csv(out_pred_file, index=False)
    print(f"\nSaved test predictions to {out_pred_file}")

    plot_predictions_per_product(
        df_test=df_test,
        y_test=y_test,
        y_test_pred=y_test_pred,
        out_dir=out_dir_plot,
        lag_window=lag_window,
        graph_mode=graph_mode,
        temporal_type=temporal_type,
        max_plots=None,
    )
    print(f"Saved per-product prediction plots to {out_dir_plot}")


# =========================
# Main: run all configs
# =========================


def main():
    global RUN_SUMMARY
    RUN_SUMMARY = []

    for temporal_type in TEMPORAL_TYPES:
        for lag_window in LAG_WINDOWS:
            for graph_mode in GRAPH_MODES:
                train_xgb_graph_baseline(
                    temporal_type=temporal_type,
                    lag_window=lag_window,
                    graph_mode=graph_mode,
                    tag=(
                        f"baseline3_xgb_graph_{graph_mode}_raw_"
                        f"lag{lag_window}_{temporal_type}"
                    ),
                )

    if RUN_SUMMARY:
        df_sum = pd.DataFrame(RUN_SUMMARY)
        print("\n=== Baseline 3 (XGB + Graph) summary ===")
        df_sum = df_sum.sort_values(
            [
                "temporal_type",
                "graph_mode",
                "lag_window",
                "horizon",
                "target_type",
                "tag",
            ]
        )
        print(
            df_sum[
                [
                    "temporal_type",
                    "graph_mode",
                    "lag_window",
                    "horizon",
                    "target_type",
                    "tag",
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
            / "baseline_3"
            / "summary_xgb_graph_baseline3_raw_lags_graphmodes.csv"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_sum.to_csv(out_path, index=False)
        print(f"\nSaved baseline 3 summary to {out_path}")


if __name__ == "__main__":
    main()