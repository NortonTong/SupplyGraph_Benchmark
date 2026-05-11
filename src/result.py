import pandas as pd
from pathlib import Path
from config.config import PROC_DIR

def load_if_exists(path: Path, **read_kwargs) -> pd.DataFrame:
    if not path.exists():
        print(f"[WARN] Summary file not found, skip: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path, **read_kwargs)
    print(f"[LOAD] {path} shape={df.shape}")
    return df

def aggregate_all_baseline_tests():
    """
    Tổng hợp kết quả TEST của tất cả baseline:
    - Naive (last_t0)
    - Baseline 1: XGB tabular
    - Baseline 2: GRU
    - Baseline 3: XGB + graph (projected/homo/hetero)
    - Baseline 4: GNN (projected/homo5/hetero5)
    - Baseline 5: XGB + GNN embedding
    - Baseline 6: Residual XGB + GNN
    - Baseline 7: XGB + graph features

    Chỉ lấy test metrics, không lấy train/val.
    """

    base_pred = Path(PROC_DIR) / "predictions"

    dfs = []

    # ---------- Naive ----------
    naive_path = base_pred / "naive" / "summary_naive_last_t0.csv"
    df_naive = load_if_exists(naive_path)
    if not df_naive.empty:
        # giả sử có các cột: temporal_type, lag_window, horizon, MAE_test, RMSE_test, MAPE_test, sMAPE_test
        df = df_naive.copy()
        df["model_family"] = "naive"
        df["model_name"] = "last_t0"
        # nếu không có các cột MAPE/sMAPE thì bỏ, chỉ giữ những cột tồn tại
        keep_cols = [
            "temporal_type", "lag_window", "horizon",
            "model_family", "model_name",
        ]
        for c in ["MAE_test", "RMSE_test", "MAPE_test", "sMAPE_test"]:
            if c in df.columns:
                keep_cols.append(c)
        dfs.append(df[keep_cols])

    # ---------- Baseline 1: XGB Tabular ----------
    b1_path = base_pred / "baseline_1" / "summary_xgb_tabular_baseline1_raw_targets_lags.csv"
    df_b1 = load_if_exists(b1_path)
    if not df_b1.empty:
        df = df_b1.copy()
        df["model_family"] = "xgb_tabular"
        df["model_name"] = df["tag"]
        keep_cols = [
            "temporal_type", "lag_window", "horizon",
            "model_family", "model_name",
            "MAE_test", "RMSE_test",
        ]
        # nếu summary baseline1 có thêm MAPE/sMAPE thì pick thêm
        for c in ["MAPE_test", "sMAPE_test"]:
            if c in df.columns:
                keep_cols.append(c)
        dfs.append(df[keep_cols])

    # ---------- Baseline 2: GRU ----------
    # file tên dạng summary_gru_baseline2_h{...}_windows_{...}_outputs_{...}.csv
    b2_dir = base_pred / "baseline_2" / "gru"
    if b2_dir.exists():
        for path in b2_dir.glob("summary_gru_baseline2_*.csv"):
            df_b2 = load_if_exists(path)
            if df_b2.empty:
                continue
            df = df_b2.copy()
            df["model_family"] = "gru"
            df["model_name"] = df["tag"]
            # map column names -> *_test để align
            df = df.rename(
                columns={
                    "window": "lag_window",
                    "test_mae": "MAE_test",
                    "test_rmse": "RMSE_test",
                }
            )
            keep_cols = [
                "temporal_type", "lag_window", "horizon",
                "model_family", "model_name",
                "MAE_test", "RMSE_test",
            ]
            dfs.append(df[keep_cols])

    # ---------- Baseline 3: XGB + Graph ----------
    b3_path = base_pred / "baseline_3" / "summary_xgb_graph_baseline3_raw_lags_graphmodes_horizons.csv"
    df_b3 = load_if_exists(b3_path)
    if not df_b3.empty:
        df = df_b3.copy()
        df["model_family"] = "xgb_graph"
        df["model_name"] = df["tag"]  # trong đó encode graph_mode
        keep_cols = [
            "temporal_type", "lag_window", "horizon",
            "model_family", "model_name",
            "MAE_test", "RMSE_test",
        ]
        for c in ["MAPE_test", "sMAPE_test"]:
            if c in df.columns:
                keep_cols.append(c)
        dfs.append(df[keep_cols])

    # ---------- Baseline 4: pure GNN ----------
    # mỗi temporal_type + mode_name (raw/log1p/softplus) có 1 summary file
    b4_dir = base_pred / "baseline_4"
    if b4_dir.exists():
        for temporal_dir in b4_dir.iterdir():
            if not temporal_dir.is_dir():
                continue
            # trong mỗi temporal_type_mode folder có summary_baseline_4_*.csv
            for path in temporal_dir.glob("summary_baseline_4_*.csv"):
                df_b4 = load_if_exists(path)
                if df_b4.empty:
                    continue
                df = df_b4.copy()
                df["model_family"] = "gnn"
                # model_name = variant + edge_view (nếu có) + target_transform
                df["model_name"] = df.apply(
                    lambda r: f"{r['variant']}"
                              + (f"_{r['edge_view']}" if pd.notna(r["edge_view"]) else "")
                              + f"_{r['target_transform']}",
                    axis=1,
                )
                keep_cols = [
                    "temporal_type", "lag_window", "horizon",
                    "model_family", "model_name",
                    "MAE_test", "RMSE_test",
                    "MAPE_test", "sMAPE_test",
                ]
                # 4 baseline này luôn có full 4 metric, nếu không thì filter theo tồn tại
                keep_cols = [c for c in keep_cols if c in df.columns]
                dfs.append(df[keep_cols])

    # ---------- Baseline 5: XGB + GNN embedding ----------
    b5_path = base_pred / "baseline_5" / "summary_xgb_gnn_embed_baseline5_raw_lags_graphmodes_horizons.csv"
    df_b5 = load_if_exists(b5_path)
    if not df_b5.empty:
        df = df_b5.copy()
        df["model_family"] = "xgb_gnn_embed"
        df["model_name"] = df["tag"]
        keep_cols = [
            "temporal_type", "lag_window", "horizon",
            "model_family", "model_name",
            "MAE_test", "RMSE_test",
        ]
        for c in ["MAPE_test", "sMAPE_test"]:
            if c in df.columns:
                keep_cols.append(c)
        dfs.append(df[keep_cols])

    # ---------- Baseline 6: Residual XGB + GNN ----------
    b6_path = base_pred / "baseline_6" / "summary_baseline_6_residual_xgb_gnn.csv"
    df_b6 = load_if_exists(b6_path)
    if not df_b6.empty:
        df = df_b6.copy()
        df["model_family"] = "residual_xgb_gnn"
        # model_name = graph_type + edge_view + mode
        df["model_name"] = df.apply(
            lambda r: f"{r['graph_type']}"
                      + (f"_{r['edge_view']}" if pd.notna(r["edge_view"]) else "")
                      + f"_{r['mode']}",
            axis=1,
        )
        keep_cols = [
            "temporal_type", "lag_window", "horizon",
            "model_family", "model_name",
            "MAE_test", "RMSE_test",
            "MAPE_test", "sMAPE_test",
        ]
        keep_cols = [c for c in keep_cols if c in df.columns]
        dfs.append(df[keep_cols])

    # ---------- Baseline 7: XGB + graph features ----------
    b7_path = base_pred / "baseline_7" / "summary_xgb_tabular_graphfeat_raw_targets.csv"
    df_b7 = load_if_exists(b7_path)
    if not df_b7.empty:
        df = df_b7.copy()
        df["model_family"] = "xgb_tabular_graphfeat"
        df["model_name"] = df["graph_type"]  # product_graph_same_* / homo5 / hetero5
        keep_cols = [
            "temporal_type", "lag_window", "horizon",
            "model_family", "model_name",
            "MAE_test", "RMSE_test",
        ]
        for c in ["MAPE_test", "sMAPE_test"]:
            if c in df.columns:
                keep_cols.append(c)
        dfs.append(df[keep_cols])

    # ---------- Gộp và lưu ----------
    if not dfs:
        print("[AGG] No summary files found, nothing to aggregate.")
        return

    df_all = pd.concat(dfs, ignore_index=True, sort=False)

    # đảm bảo các cột metric đều tồn tại (nếu thiếu, fill NaN)
    for c in ["MAE_test", "RMSE_test", "MAPE_test", "sMAPE_test"]:
        if c not in df_all.columns:
            df_all[c] = pd.NA

    df_all = df_all.sort_values(
        ["temporal_type", "lag_window", "horizon", "model_family", "model_name"]
    )

    out_dir = base_pred / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "all_baselines_test_metrics.csv"
    df_all.to_csv(out_path, index=False)
    print(f"[AGG] Saved aggregated test metrics to {out_path}")
    print(df_all.head())

if __name__ == "__main__":
    aggregate_all_baseline_tests()