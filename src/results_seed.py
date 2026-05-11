from pathlib import Path
import pandas as pd
from config.config import PROC_DIR


def agg_mean_std(df, group_cols, mae_col, rmse_col, prefix=""):
    g = (
        df.groupby(group_cols, dropna=False)
        .agg(
            **{
                f"{prefix}mean_mae": (mae_col, "mean"),
                f"{prefix}std_mae": (mae_col, "std"),
                f"{prefix}mean_rmse": (rmse_col, "mean"),
                f"{prefix}std_rmse": (rmse_col, "std"),
            }
        )
        .reset_index()
    )
    return g


def load_if_exists(path: Path):
    print(f"Checking {path}")
    if path.exists():
        print("  -> FOUND")
        return pd.read_csv(path)
    else:
        print("  -> NOT FOUND")
        return None


def main():
    print("PROC_DIR =", PROC_DIR)

    dfs = []

    # ---------- Baseline 1: XGB tabular ----------
    path_b1 = (
        PROC_DIR
        / "predictions"
        / "baseline_1"
        / "summary_xgb_tabular_baseline1_raw_targets_lags_with_seeds.csv"
    )
    df_b1 = load_if_exists(path_b1)
    if df_b1 is not None:
        gcols = ["temporal_type", "lag_window", "horizon", "variant", "target_type"]
        df_b1_agg = agg_mean_std(
            df_b1,
            group_cols=gcols,
            mae_col="MAE_test",
            rmse_col="RMSE_test",
        )
        df_b1_agg["baseline"] = "baseline_1"
        dfs.append(df_b1_agg)

    # ---------- Baseline 2: GRU ----------
    dir_b2 = PROC_DIR / "predictions" / "baseline_2" / "gru"
    print(f"Checking dir {dir_b2}")
    if dir_b2.exists():
        # hiện tại anh có 1 file: summary_gru_baseline2_h7_windows_7_14_30_outputs_raw.csv
        for path_b2 in dir_b2.glob("summary_gru_baseline2_*.csv"):
            print(f"  Found baseline_2 file: {path_b2.name}")
            df_b2 = pd.read_csv(path_b2)
            gcols = [
                "temporal_type",
                "window",             # cột trong code anh
                "horizon",
                "output_transform",   # transform_tag
                "seq_len",
            ]
            df_b2_agg = agg_mean_std(
                df_b2,
                group_cols=gcols,
                mae_col="test_mae",
                rmse_col="test_rmse",
            )
            df_b2_agg["baseline"] = "baseline_2"
            dfs.append(df_b2_agg)
    else:
        print("  -> baseline_2 dir NOT FOUND")

    # ---------- Baseline 3: XGB graph ----------
    path_b3 = (
        PROC_DIR
        / "predictions"
        / "baseline_3"
        / "summary_xgb_graph_baseline3_raw_lags_graphmodes_horizons_with_seeds.csv"
    )
    df_b3 = load_if_exists(path_b3)
    if df_b3 is not None:
        gcols = [
            "temporal_type",
            "lag_window",
            "horizon",
            "graph_mode",
            "variant",
            "target_type",
        ]
        df_b3_agg = agg_mean_std(
            df_b3,
            group_cols=gcols,
            mae_col="MAE_test",
            rmse_col="RMSE_test",
        )
        df_b3_agg["baseline"] = "baseline_3"
        dfs.append(df_b3_agg)

    # ---------- Baseline 4: GNN projected / homo5 / hetero5 ----------
    # theo listing: data/processed/predictions/baseline_4/unit_raw/summary_baseline_4_unit_raw.csv
    dir_b4 = PROC_DIR / "predictions" / "baseline_4"
    if dir_b4.exists():
        for path_b4 in dir_b4.rglob("summary_baseline_4_*.csv"):
            print(f"  Found baseline_4 file: {path_b4}")
            df_b4 = pd.read_csv(path_b4)
            gcols = [
                "temporal_type",
                "lag_window",
                "horizon",
                "variant",           # gnn_projected / gnn_homo5 / gnn_hetero5
                "edge_view",
                "target_transform",
            ]
            df_b4_agg = agg_mean_std(
                df_b4,
                group_cols=gcols,
                mae_col="MAE_test",
                rmse_col="RMSE_test",
            )
            df_b4_agg["baseline"] = "baseline_4"
            dfs.append(df_b4_agg)

    # ---------- Baseline 5: XGB + GNN embedding ----------
    path_b5 = (
        PROC_DIR
        / "predictions"
        / "baseline_5"
        / "summary_xgb_gnn_embed_baseline5_raw_lags_graphmodes_horizons.csv"
    )
    df_b5 = load_if_exists(path_b5)
    if df_b5 is not None:
        gcols = [
            "temporal_type",
            "lag_window",
            "horizon",
            "graph_mode",
            "variant",
            "target_type",
        ]
        df_b5_agg = agg_mean_std(
            df_b5,
            group_cols=gcols,
            mae_col="MAE_test",
            rmse_col="RMSE_test",
        )
        df_b5_agg["baseline"] = "baseline_5"
        dfs.append(df_b5_agg)

    # ---------- Baseline 6: residual XGB + GNN ----------
    path_b6 = (
        PROC_DIR
        / "predictions"
        / "baseline_6"
        / "summary_baseline_6_residual_xgb_gnn.csv"
    )
    df_b6 = load_if_exists(path_b6)
    if df_b6 is not None:
        gcols = [
            "temporal_type",
            "lag_window",
            "horizon",
            "graph_type",    # projected / homo5 / hetero5
            "edge_view",
            "mode",          # mode_name
            "variant",
        ]
        df_b6_agg = agg_mean_std(
            df_b6,
            group_cols=gcols,
            mae_col="MAE_test",
            rmse_col="RMSE_test",
        )
        df_b6_agg["baseline"] = "baseline_6"
        dfs.append(df_b6_agg)

    # ---------- Baseline 7: XGB + advanced graph features ----------
    path_b7 = (
        PROC_DIR
        / "predictions"
        / "baseline_7"
        / "summary_xgb_tabular_graphfeat_raw_targets_with_seeds.csv"
    )
    df_b7 = load_if_exists(path_b7)
    if df_b7 is not None:
        gcols = [
            "temporal_type",
            "lag_window",
            "horizon",
            "graph_type",
            "variant",
            "target_type",
        ]
        df_b7_agg = agg_mean_std(
            df_b7,
            group_cols=gcols,
            mae_col="MAE_test",
            rmse_col="RMSE_test",
        )
        df_b7_agg["baseline"] = "baseline_7"
        dfs.append(df_b7_agg)

    # ---------- Naive baseline ----------
    path_naive = (
        PROC_DIR
        / "predictions"
        / "naive"
        / "summary_naive_last_t0.csv"
    )
    df_naive = load_if_exists(path_naive)
    if df_naive is not None:
        df_n = df_naive.copy()
        df_n["mean_mae"] = df_n["MAE_test"]
        df_n["std_mae"] = 0.0
        df_n["mean_rmse"] = df_n["RMSE_test"]
        df_n["std_rmse"] = 0.0
        df_n["baseline"] = "naive"
        dfs.append(df_n)

    # ---------- Gộp & lưu ----------
    if not dfs:
        print("No summary files found, nothing to aggregate.")
        return

    df_all = pd.concat(dfs, axis=0, ignore_index=True, sort=False)

    # chuẩn hóa tên cột: nếu có 'window' mà không có 'lag_window' → rename
    if "window" in df_all.columns and "lag_window" not in df_all.columns:
        df_all = df_all.rename(columns={"window": "lag_window"})

    out_dir = PROC_DIR / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary_all_baselines_agg.csv"
    df_all.to_csv(out_path, index=False)
    print(f"\nSaved aggregated summary to {out_path}")


if __name__ == "__main__":
    main()