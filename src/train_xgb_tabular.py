import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import random

from config.config import PROC_DIR, DEFAULT_EXPERIMENTS

RUN_SUMMARY: list[dict] = []

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

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

def load_tabular_baseline(temporal_type: str = "unit", lag_window: int = 7, horizon: int = 7) -> pd.DataFrame:
    path = (
        PROC_DIR
        / "baseline"
        / "xgboost"
        / f"xgboost_tabular_h{horizon}_lag{lag_window}_{temporal_type}.parquet"
    )
    print(f"Loading tabular baseline (no graph) from {path}")
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_parquet(path)
    if "target" not in df.columns:
        raise KeyError(f"'target' not found in {path}")
    return df

def split_train_val_test(df: pd.DataFrame):
    return (
        df[df["split"] == "train"].copy(),
        df[df["split"] == "val"].copy(),
        df[df["split"] == "test"].copy(),
    )

def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if "target" not in df.columns:
        raise KeyError("target not found in dataframe passed to prepare_features()")
    y = df["target"].astype(float)
    drop_cols = ["target", "split", "node_id", "node_index", "date", "day"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return X, y

def plot_predictions_per_product(df_test, y_test, y_test_pred, out_dir, lag_window, temporal_type, max_plots=None):
    df_plot = df_test[["node_id", "date"]].copy()
    df_plot["y_true"] = np.asarray(y_test, dtype=float)
    df_plot["y_pred"] = np.asarray(y_test_pred, dtype=float)

    unique_nodes = df_plot["node_id"].unique()
    if max_plots is not None:
        unique_nodes = unique_nodes[:max_plots]

    out_dir.mkdir(parents=True, exist_ok=True)

    for node in unique_nodes:
        sub = df_plot[df_plot["node_id"] == node].sort_values("date")
        if sub.empty:
            continue
        plt.figure(figsize=(10, 4))
        plt.plot(sub["date"], sub["y_true"], label="True", marker="o", linewidth=1)
        plt.plot(sub["date"], sub["y_pred"], label="Pred", marker="x", linewidth=1)
        plt.title(f"Baseline 1 - XGB Tabular (lag={lag_window}, {temporal_type}) - node_id={node}")
        plt.xlabel("Date")
        plt.ylabel("Sales order")
        plt.legend()
        plt.tight_layout()
        fname = out_dir / f"xgb_tabular_raw_lag{lag_window}_{temporal_type}_node_{node}.png"
        plt.savefig(fname, dpi=150)
        plt.close()

def train_xgb_tabular_baseline(temporal_type="unit", lag_window=7, horizon=7, tag=None, seed=42):
    target_type = "raw"
    if tag is None:
        tag = f"baseline1_xgb_tabular_{target_type}_h{horizon}_lag{lag_window}_{temporal_type}_seed{seed}"

    set_global_seed(seed)
    print(
        f"\n=== Training XGBoost TABULAR baseline (Baseline 1) "
        f"H{horizon}, lag={lag_window}, temporal_type={temporal_type}, "
        f"target_type={target_type}, tag={tag}, seed={seed} ==="
    )

    df_base = load_tabular_baseline(temporal_type=temporal_type, lag_window=lag_window, horizon=horizon)

    print(f"Loaded shape: {df_base.shape}")
    print(f"Columns: {df_base.columns.tolist()[:15]} ...")
    print("Target checksum:", float(df_base["target"].sum()))
    print("Split counts:\n", df_base["split"].value_counts())

    df_train, df_val, df_test = split_train_val_test(df_base)
    print("Splits rows:", len(df_train), len(df_val), len(df_test))
    print("Train target head:", df_train["target"].head().tolist())
    print("Val target head:", df_val["target"].head().tolist())
    print("Test target head:", df_test["target"].head().tolist())

    X_train, y_train_raw = prepare_features(df_train)
    X_val, y_val_raw = prepare_features(df_val)
    X_test, y_test_raw = prepare_features(df_test)

    y_train = y_train_raw.values
    y_val = y_val_raw.values

    print(f"\n[H{horizon}][lag{lag_window}][{tag}] Using {X_train.shape[1]} features")
    print(f"Train samples: {X_train.shape[0]}")
    print(f"Val samples:   {X_val.shape[0]}")
    print(f"Test samples:  {X_test.shape[0]}")

    model = XGBRegressor(
        n_estimators=5000,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
        eval_metric="rmse",
        early_stopping_rounds=100,
    )

    eval_set = [(X_train, y_train), (X_val, y_val)]
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    evals_result = model.evals_result()
    train_rmse_hist = evals_result["validation_0"]["rmse"]
    val_rmse_hist = evals_result["validation_1"]["rmse"]

    plt.figure(figsize=(8, 5))
    plt.plot(train_rmse_hist, label="Train RMSE")
    plt.plot(val_rmse_hist, label="Val RMSE")
    plt.axvline(model.best_iteration, color="red", linestyle="--", label="Best iter")
    plt.xlabel("Boosting round")
    plt.ylabel("RMSE")
    plt.title(f"Learning curve H{horizon} lag{lag_window} - {tag} - {temporal_type}")
    plt.legend()
    plt.tight_layout()

    out_curve = (
        PROC_DIR
        / "predictions"
        / "baseline_1"
        / "plots_learning_curves"
        / f"learning_curve_h{horizon}_lag{lag_window}_{target_type}_{tag}_{temporal_type}.png"
    )
    out_curve.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_curve, dpi=150)
    plt.close()
    print(f"Saved learning curve to {out_curve}")

    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    y_train_true = y_train_raw.values
    y_val_true = y_val_raw.values
    y_test_true = y_test_raw.values

    mae_train = mean_absolute_error(y_train_true, y_train_pred)
    rmse_train = root_mean_squared_error(y_train_true, y_train_pred)
    mape_train = mape(y_train_true, y_train_pred)
    smape_train = smape(y_train_true, y_train_pred)

    mae_val = mean_absolute_error(y_val_true, y_val_pred)
    rmse_val = root_mean_squared_error(y_val_true, y_val_pred)
    mape_val = mape(y_val_true, y_val_pred)
    smape_val = smape(y_val_true, y_val_pred)

    mae_test = mean_absolute_error(y_test_true, y_test_pred)
    rmse_test = root_mean_squared_error(y_test_true, y_test_pred)
    mape_test = mape(y_test_true, y_test_pred)
    smape_test = smape(y_test_true, y_test_pred)

    print(f"\n[H{horizon}][lag{lag_window}][tabular][{target_type}][{tag}] Train:")
    print(f"  MAE  : {mae_train:.4f}")
    print(f"  RMSE : {rmse_train:.4f}")
    print(f"  MAPE : {mape_train:.4f}")
    print(f"  sMAPE: {smape_train:.4f}")

    print(f"\n[H{horizon}][lag{lag_window}][tabular][{target_type}][{tag}] Val:")
    print(f"  MAE  : {mae_val:.4f}")
    print(f"  RMSE : {rmse_val:.4f}")
    print(f"  MAPE : {mape_val:.4f}")
    print(f"  sMAPE: {smape_val:.4f}")

    print(f"\n[H{horizon}][lag{lag_window}][tabular][{target_type}][{tag}] Test:")
    print(f"  MAE  : {mae_test:.4f}")
    print(f"  RMSE : {rmse_test:.4f}")
    print(f"  MAPE : {mape_test:.4f}")
    print(f"  sMAPE: {smape_test:.4f}")

    df_train_pred = df_train[["node_id", "node_index", "date", "day"]].copy()
    df_train_pred["split"] = "train"
    df_train_pred["y_xgb"] = y_train_pred

    df_val_pred = df_val[["node_id", "node_index", "date", "day"]].copy()
    df_val_pred["split"] = "val"
    df_val_pred["y_xgb"] = y_val_pred

    df_test_pred_full = df_test[["node_id", "node_index", "date", "day"]].copy()
    df_test_pred_full["split"] = "test"
    df_test_pred_full["y_xgb"] = y_test_pred

    df_all_pred = pd.concat([df_train_pred, df_val_pred, df_test_pred_full], ignore_index=True)

    pred_path = (
        PROC_DIR
        / "baseline"
        / "xgboost"
        / f"xgboost_predictions_h{horizon}_lag{lag_window}_{temporal_type}_seed{seed}.parquet"
    )
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    df_all_pred.to_parquet(pred_path, index=False)
    print(f"[XGB] Saved full predictions for residual GNN to {pred_path}")

    base_pred_dir = PROC_DIR / "predictions" / "baseline_1"
    out_dir_csv = base_pred_dir / "csv" / f"{temporal_type}"
    plot_folder = f"{target_type}_lag{lag_window}_h{horizon}"
    out_dir_plot = base_pred_dir / "plots_xgb_tabular" / plot_folder / f"{temporal_type}"
    out_dir_csv.mkdir(parents=True, exist_ok=True)
    out_dir_plot.mkdir(parents=True, exist_ok=True)

    df_test_pred = pd.DataFrame({
        "node_id": df_test["node_id"].values,
        "date": df_test["date"].values,
        "y_true": y_test_true,
        "y_pred": y_test_pred,
    })
    out_pred_file = (
        out_dir_csv
        / f"xgb_tabular_h{horizon}_lag{lag_window}_{target_type}_test_predictions_{temporal_type}_seed{seed}.csv"
    )
    df_test_pred.to_csv(out_pred_file, index=False)
    print(f"\nSaved test predictions to {out_pred_file}")

    plot_predictions_per_product(
        df_test=df_test,
        y_test=y_test_true,
        y_test_pred=y_test_pred,
        out_dir=out_dir_plot,
        lag_window=lag_window,
        temporal_type=temporal_type,
        max_plots=None,
    )
    print(f"Saved per-product prediction plots to {out_dir_plot}")

    RUN_SUMMARY.append({
        "temporal_type": temporal_type,
        "lag_window": lag_window,
        "horizon": horizon,
        "seed": seed,
        "variant": f"baseline_1_xgb_tabular_{target_type}",
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
    })

def main():
    global RUN_SUMMARY
    RUN_SUMMARY = []

    seeds = [0, 1, 2]

    for exp in DEFAULT_EXPERIMENTS:
        temporal_type = exp.temporal_type
        horizons = list(exp.horizons)
        lag_windows = list(exp.lag_windows)

        for lag_window in lag_windows:
            for horizon in horizons:
                for seed in seeds:
                    train_xgb_tabular_baseline(
                        temporal_type=temporal_type,
                        lag_window=lag_window,
                        horizon=horizon,
                        tag=(
                            f"baseline1_xgb_tabular_raw_h{horizon}_lag{lag_window}_"
                            f"{temporal_type}_seed{seed}"
                        ),
                        seed=seed,
                    )

    if RUN_SUMMARY:
        df_sum = pd.DataFrame(RUN_SUMMARY)
        print("\n=== Baseline 1 (XGB Tabular) summary with seeds ===")
        df_sum = df_sum.sort_values(["temporal_type", "lag_window", "horizon", "target_type", "seed", "tag"])
        print(df_sum[[
            "temporal_type", "lag_window", "horizon", "target_type", "seed", "tag",
            "n_features", "MAE_train", "RMSE_train", "MAE_val", "RMSE_val", "MAE_test", "RMSE_test"
        ]])

        out_path = (
            PROC_DIR
            / "predictions"
            / "baseline_1"
            / "summary_xgb_tabular_baseline1_raw_targets_lags_with_seeds.csv"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_sum.to_csv(out_path, index=False)
        print(f"\nSaved baseline 1 summary (with seeds) to {out_path}")

if __name__ == "__main__":
    main()