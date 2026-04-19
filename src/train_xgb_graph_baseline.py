import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Literal

from config.config import PROC_DIR, TEMPORAL_TYPE

RUN_SUMMARY = []

HORIZON = 7
LAG_WINDOWS = [7, 14]          # 2 lookback windows
TARGET_TYPES = ["raw", "log1p"]  # 2 loại target
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


# =========================
# Target transforms
# =========================

def transform_target(y: np.ndarray, target_type: str) -> np.ndarray:
    if target_type == "raw":
        return y
    elif target_type == "log1p":
        return np.log1p(np.clip(y, a_min=0.0, a_max=None))
    else:
        raise ValueError(f"Unknown target_type={target_type}")


def inverse_transform_target(y_hat: np.ndarray, target_type: str) -> np.ndarray:
    if target_type == "raw":
        y = y_hat
    elif target_type == "log1p":
        y = np.expm1(y_hat)
    else:
        raise ValueError(f"Unknown target_type={target_type}")
    return np.maximum(y, 0.0)


# =========================
# Load baseline (no graph)
# =========================

def load_tabular_baseline(
    temporal_type: str = "unit",
    lag_window: int = 7,
) -> pd.DataFrame:
    """
    Load file baseline XGBoost đã được preprocessing (không graph),
    ví dụ: data/processed/baseline/xgboost_tabular_h7_lag7_unit.parquet
    """
    path = (
        PROC_DIR
        / "baseline"
        / f"xgboost_tabular_h{HORIZON}_lag{lag_window}_{temporal_type}.parquet"
    )
    print(f"Loading baseline1 dataset from {path}")
    return pd.read_parquet(path)


# =========================
# Load & merge graph features
# =========================

def load_graph_features(mode: Literal["proj", "homo", "hetero"]) -> pd.DataFrame:
    """
    Load precomputed graph features theo node_id/node_index.
    - 'proj'   -> graph_features_projected.parquet (theo node_id).
    - 'homo'   -> graph_features_homogeneous.parquet (node_id,node_index).
    - 'hetero' -> graph_features_heterogeneous.parquet (node_id,node_index).
    """
    graph_dir = PROC_DIR / "graphs"
    if mode == "proj":
        path = graph_dir / "graph_features_projected.parquet"
    elif mode == "homo":
        path = graph_dir / "graph_features_homogeneous.parquet"
    elif mode == "hetero":
        path = graph_dir / "graph_features_heterogeneous.parquet"
    else:
        raise ValueError(f"Unknown graph mode: {mode}")

    print(f"Loading graph features ({mode}) from {path}")
    return pd.read_parquet(path)


def merge_tabular_with_graph(
    df_tab: pd.DataFrame,
    df_graph: pd.DataFrame,
    mode: Literal["proj", "homo", "hetero"],
) -> pd.DataFrame:
    """
    Merge graph features vào tabular theo node_id/node_index:
    - Projected features: chỉ có node_id, merge on node_id.
    - Homo/hetero: có node_id, node_index; merge on node_id (node_index từ tabular).
    """
    if mode == "proj":
        merged = df_tab.merge(df_graph, on="node_id", how="left")
    else:
        merged = df_tab.merge(
            df_graph.drop_duplicates(subset=["node_id"]),
            on="node_id",
            how="left",
            suffixes=("", "_g"),
        )

    graph_cols = [
        c
        for c in merged.columns
        if any(
            c.startswith(prefix)
            for prefix in [
                "proj_group_",
                "proj_subgrp_",
                "proj_plant_",
                "proj_storage_",
                "homo_",
                "hetero_",
            ]
        )
    ]
    merged[graph_cols] = merged[graph_cols].fillna(0.0)

    return merged


# =========================
# Encoding & features
# =========================

def one_hot_encode_train_val_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_train = df[df["split"] == "train"].copy()
    df_val   = df[df["split"] == "val"].copy()
    df_test  = df[df["split"] == "test"].copy()

    cat_cols = [
        "group", "sub_group",
        "plant", "storage_location",
        "day_of_week", "is_weekend",
    ]
    cat_cols = [c for c in cat_cols if c in df.columns]

    df_train_enc = pd.get_dummies(df_train, columns=cat_cols, drop_first=False)
    df_val_enc   = pd.get_dummies(df_val,   columns=cat_cols, drop_first=False)
    df_test_enc  = pd.get_dummies(df_test,  columns=cat_cols, drop_first=False)

    df_val_enc  = df_val_enc.reindex(columns=df_train_enc.columns, fill_value=0)
    df_test_enc = df_test_enc.reindex(columns=df_train_enc.columns, fill_value=0)

    return df_train_enc, df_val_enc, df_test_enc


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


def plot_predictions_per_product(
    df_test_enc: pd.DataFrame,
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
    out_dir: Path,
    target_type: str,
    lag_window: int,
    graph_mode: str,
    max_plots: int | None = None,
) -> None:
    """
    Vẽ y_true vs y_pred theo ngày cho từng sản phẩm (node_id) trên test split.
    Lưu 1 file .png / sản phẩm vào out_dir, prefix xgb_graph_*.
    """
    df_plot = df_test_enc[["node_id", "date"]].copy()
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
            f"Baseline 3 - XGB + Graph ({graph_mode}, {target_type}, lag={lag_window}) - node_id={node}"
        )
        plt.xlabel("Date")
        plt.ylabel("Sales order")
        plt.legend()
        plt.tight_layout()

        fname = out_dir / f"xgb_graph_{graph_mode}_{target_type}_lag{lag_window}_node_{node}.png"
        plt.savefig(fname, dpi=150)
        plt.close()


# =========================
# Train 1 cấu hình
# =========================

def train_xgb_graph_baseline(
    temporal_type: str = "unit",
    lag_window: int = 7,
    target_type: str = "raw",
    graph_mode: Literal["proj", "homo", "hetero"] = "proj",
    tag: str | None = None,
) -> None:
    if tag is None:
        tag = f"baseline3_xgb_graph_{graph_mode}_{target_type}_lag{lag_window}_{temporal_type}"

    print(
        f"\n=== Training XGBoost GRAPH baseline (Baseline 3) "
        f"H{HORIZON}, lag={lag_window}, temporal_type={temporal_type}, "
        f"graph_mode={graph_mode}, target_type={target_type}, tag={tag} ==="
    )

    # 1) Load tabular baseline 1
    df_base = load_tabular_baseline(temporal_type=temporal_type, lag_window=lag_window)

    print(
        f"H{HORIZON} tabular+graph lag{lag_window} {temporal_type}: df rows={len(df_base)}, "
        f"unique(node,date)={df_base[['node_id','date']].drop_duplicates().shape[0]}"
    )

    # 2) Load graph features & merge
    df_graph = load_graph_features(graph_mode)
    df_merged = merge_tabular_with_graph(df_base, df_graph, graph_mode)

    # 3) One-hot encode splits
    df_train_enc, df_val_enc, df_test_enc = one_hot_encode_train_val_test(df_merged)
    print("Splits rows:", len(df_train_enc), len(df_val_enc), len(df_test_enc))

    # 4) Features & targets (raw y)
    X_train, y_train_raw = prepare_features(df_train_enc)
    X_val, y_val_raw     = prepare_features(df_val_enc)
    X_test, y_test_raw   = prepare_features(df_test_enc)

    # 5) Transform targets theo target_type
    y_train = transform_target(y_train_raw, target_type)
    y_val   = transform_target(y_val_raw,   target_type)

    feature_names = list(X_train.columns)
    print(
        f"\n[H{HORIZON}][lag{lag_window}][{tag}] Using {len(feature_names)} features (tabular + graph):"
    )
    print(feature_names)
    print(f"Train samples: {X_train.shape[0]}")
    print(f"Val   samples: {X_val.shape[0]}")
    print(f"Test  samples: {X_test.shape[0]}")

    # 6) XGBoost model
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

    # 7) Learning curve
    plt.figure(figsize=(8, 5))
    plt.plot(train_rmse_hist, label="Train RMSE (target scale)")
    plt.plot(val_rmse_hist, label="Val RMSE (target scale)")
    plt.axvline(model.best_iteration, color="red", linestyle="--", label="Best iter")
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

    # 8) Metrics trên scale gốc
    # Train
    y_train_pred_t = model.predict(X_train)
    y_train_pred   = inverse_transform_target(y_train_pred_t, target_type)
    y_train_true   = y_train_raw

    mae_train  = mean_absolute_error(y_train_true, y_train_pred)
    rmse_train = root_mean_squared_error(y_train_true, y_train_pred)
    mape_train = mape(y_train_true, y_train_pred)
    smape_train = smape(y_train_true, y_train_pred)

    # Validation
    y_val_pred_t = model.predict(X_val)
    y_val_pred   = inverse_transform_target(y_val_pred_t, target_type)
    y_val_true   = y_val_raw

    mae_val  = mean_absolute_error(y_val_true, y_val_pred)
    rmse_val = root_mean_squared_error(y_val_true, y_val_pred)
    mape_val = mape(y_val_true, y_val_pred)
    smape_val = smape(y_val_true, y_val_pred)

    # Test
    y_test_pred_t = model.predict(X_test)
    y_test_pred   = inverse_transform_target(y_test_pred_t, target_type)
    y_test_true   = y_test_raw

    mae_test  = mean_absolute_error(y_test_true, y_test_pred)
    rmse_test = root_mean_squared_error(y_test_true, y_test_pred)
    mape_test = mape(y_test_true, y_test_pred)
    smape_test = smape(y_test_true, y_test_pred)

    print(f"\n[H{HORIZON}][lag{lag_window}][{graph_mode}][{target_type}][{tag}] Train:")
    print(f"  MAE   : {mae_train:.4f}")
    print(f"  RMSE  : {rmse_train:.4f}")
    print(f"  MAPE  : {mape_train:.4f}")
    print(f"  sMAPE : {smape_train:.4f}")

    print(f"\n[H{HORIZON}][lag{lag_window}][{graph_mode}][{target_type}][{tag}] Validation:")
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
            "lag_window": lag_window,
            "horizon": HORIZON,
            "variant": f"baseline_3_xgb_graph_{graph_mode}_{target_type}",
            "tag": tag,
            "target_type": target_type,
            "graph_mode": graph_mode,
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

    # 9) Save test predictions & plots per product
    base_pred_dir = PROC_DIR / "predictions" / "baseline_3"
    out_dir_csv = base_pred_dir / "csv" / f"{temporal_type}" / graph_mode
    plot_folder = f"{target_type}_lag{lag_window}"
    out_dir_plot = base_pred_dir / "plots_xgb_graph" / graph_mode / plot_folder / f"{temporal_type}"
    out_dir_csv.mkdir(parents=True, exist_ok=True)
    out_dir_plot.mkdir(parents=True, exist_ok=True)

    df_test_pred = pd.DataFrame(
        {
            "node_id": df_test_enc["node_id"],
            "date": df_test_enc["date"],
            "y_true": y_test_true,
            "y_pred": y_test_pred,
        }
    )
    out_pred_file = out_dir_csv / f"xgb_graph_{graph_mode}_h{HORIZON}_lag{lag_window}_{target_type}_test_predictions_{temporal_type}.csv"
    df_test_pred.to_csv(out_pred_file, index=False)
    print(f"\nSaved test predictions to {out_pred_file}")

    plot_predictions_per_product(
        df_test_enc=df_test_enc,
        y_test=y_test_true,
        y_test_pred=y_test_pred,
        out_dir=out_dir_plot,
        target_type=target_type,
        lag_window=lag_window,
        graph_mode=graph_mode,
        max_plots=None,
    )
    print(f"Saved per-product prediction plots to {out_dir_plot}")


# =========================
# Main: run all configs
# =========================

def main():
    global RUN_SUMMARY
    RUN_SUMMARY = []

    temporal_types = TEMPORAL_TYPE  # ví dụ ["unit", "weight"]

    for temporal_type in temporal_types:
        for lag_window in LAG_WINDOWS:
            for graph_mode in GRAPH_MODES:
                for target_type in TARGET_TYPES:
                    train_xgb_graph_baseline(
                        temporal_type=temporal_type,
                        lag_window=lag_window,
                        target_type=target_type,
                        graph_mode=graph_mode,
                        tag=f"baseline3_xgb_graph_{graph_mode}_{target_type}_lag{lag_window}_{temporal_type}",
                    )

    if RUN_SUMMARY:
        df_sum = pd.DataFrame(RUN_SUMMARY)
        print("\n=== Baseline 3 (XGB + Graph) summary ===")
        df_sum = df_sum.sort_values(
            ["temporal_type", "graph_mode", "lag_window", "horizon", "target_type", "tag"]
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
                    "MAE_train", "RMSE_train",
                    "MAE_val", "RMSE_val",
                    "MAE_test", "RMSE_test",
                ]
            ]
        )

        out_path = PROC_DIR / "predictions" / "baseline_3" / "summary_xgb_graph_baseline3_targets_lags_graphmodes.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_sum.to_csv(out_path, index=False)
        print(f"\nSaved baseline 3 summary to {out_path}")


if __name__ == "__main__":
    main()