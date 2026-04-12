# summarize_experiments.py

import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from config.config import PROC_DIR


def summarize_xgb_predictions():
    base_dir = PROC_DIR / "predictions"
    rows = []

    # duyệt tất cả file csv con
    for csv_path in base_dir.rglob("*.csv"):
        name = csv_path.name

        # bỏ file summary cũ nếu có
        if "summary" in name:
            continue

        # cố gắng parse horizon, lag, variant, tag từ path
        # ví dụ: predictions/baseline_3/lag21/B3_gcn_only/xgb_h7_lag21_test_predictions.csv
        parts = csv_path.relative_to(base_dir).parts
        if len(parts) < 4:
            # không đúng pattern, vẫn tính nhưng meta để None
            variant = parts[0] if len(parts) > 0 else None
            lag_window = None
            tag = None
        else:
            variant = parts[0]               # baseline_1 / baseline_2 / baseline_3 / ...
            lag_dir = parts[1]               # lag21
            tag = parts[2]                   # B3_gcn_only ...
            lag_match = re.search(r"lag(\d+)", lag_dir)
            lag_window = int(lag_match.group(1)) if lag_match else None

        # parse horizon từ tên file
        h_match = re.search(r"h(\d+)", name)
        horizon = int(h_match.group(1)) if h_match else None

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Skip {csv_path} due to read error: {e}")
            continue

        if not {"y_true", "y_pred"}.issubset(df.columns):
            print(f"Skip {csv_path}, missing y_true/y_pred")
            continue

        mae = mean_absolute_error(df["y_true"], df["y_pred"])
        rmse = root_mean_squared_error(df["y_true"], df["y_pred"])

        rows.append(
            {
                "model_type": "xgb",
                "variant": variant,
                "tag": tag,
                "lag_window": lag_window,
                "horizon": horizon,
                "n_samples": len(df),
                "MAE": mae,
                "RMSE": rmse,
                "file": str(csv_path.relative_to(PROC_DIR)),
            }
        )

    if not rows:
        print("No XGB prediction files found.")
        return None

    df_sum = pd.DataFrame(rows)
    out_path = PROC_DIR / "predictions" / "summary_xgb_predictions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_sum = df_sum.sort_values(["variant", "tag", "lag_window", "horizon"])
    df_sum.to_csv(out_path, index=False)
    print(f"Saved XGB summary to {out_path}")
    return df_sum


def summarize_gnn_predictions():
    base_dir = PROC_DIR / "predictions_gnn_ablation"
    rows = []

    for csv_path in base_dir.rglob("*.csv"):
        name = csv_path.name
        # ví dụ: gnn_h7_test_predictions.csv dưới folder tag: GNN_plant_GCN, GNN_all4_GIN, ...
        parts = csv_path.relative_to(base_dir).parts
        if len(parts) < 2:
            tag = None
        else:
            tag = parts[0]  # folder chính là tag

        h_match = re.search(r"h(\d+)", name)
        horizon = int(h_match.group(1)) if h_match else None

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Skip {csv_path} due to read error: {e}")
            continue

        if not {"y_true", "y_pred"}.issubset(df.columns):
            print(f"Skip {csv_path}, missing y_true/y_pred")
            continue

        mae = mean_absolute_error(df["y_true"], df["y_pred"])
        rmse = root_mean_squared_error(df["y_true"], df["y_pred"])

        rows.append(
            {
                "model_type": "gnn",
                "variant": "ablation",
                "tag": tag,
                "lag_window": None,  # GNN không có lag_window
                "horizon": horizon,
                "n_samples": len(df),
                "MAE": mae,
                "RMSE": rmse,
                "file": str(csv_path.relative_to(PROC_DIR)),
            }
        )

    if not rows:
        print("No GNN prediction files found.")
        return None

    df_sum = pd.DataFrame(rows)
    out_path = PROC_DIR / "predictions_gnn_ablation" / "summary_gnn_predictions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_sum = df_sum.sort_values(["tag", "horizon"])
    df_sum.to_csv(out_path, index=False)
    print(f"Saved GNN summary to {out_path}")
    return df_sum


def main():
    df_xgb = summarize_xgb_predictions()
    df_gnn = summarize_gnn_predictions()

    if df_xgb is not None or df_gnn is not None:
        all_rows = []
        if df_xgb is not None:
            all_rows.append(df_xgb)
        if df_gnn is not None:
            all_rows.append(df_gnn)
        df_all = pd.concat(all_rows, axis=0, ignore_index=True)
        out_path = PROC_DIR / "predictions" / "summary_all_models.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_all.to_csv(out_path, index=False)
        print(f"Saved combined summary to {out_path}")


if __name__ == "__main__":
    main()