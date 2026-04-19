# aggregate_all_results.py

from pathlib import Path
import pandas as pd

from config.config import PROC_DIR


def load_naive_last() -> pd.DataFrame:
    path = PROC_DIR / "predictions_naive" / "summary_naive_last_t0.csv"
    if not path.exists():
        print(f"[NAIVE] Not found: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    # đảm bảo có các cột nhận diện chung
    df["model_family"] = "naive_last_t0"
    df["graph_type"] = "no_graph"
    df["variant"] = "naive_last_t0"
    df["edge_view"] = None
    return df


def load_baseline1_xgb() -> pd.DataFrame:
    path = PROC_DIR / "predictions" / "baseline_1" / "summary_tabular_baseline1_targets_lags.csv"
    if not path.exists():
        print(f"[BL1] Not found: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df["model_family"] = "xgb_tabular"
    df["graph_type"] = "no_graph"
    df["edge_view"] = None
    # chuẩn hóa tên cột target_transform, clip_pred, use_softplus_output nếu cần
    if "target_type" in df.columns and "target_transform" not in df.columns:
        df["target_transform"] = df["target_type"]
    if "clip_pred" not in df.columns:
        df["clip_pred"] = False
    if "use_softplus_output" not in df.columns:
        df["use_softplus_output"] = False
    if "variant" not in df.columns:
        df["variant"] = "baseline_1_xgb_tabular"
    return df


import pandas as pd

def load_baseline2_gru():
    path = r"D:\SupplyGraph_experiment\data\processed\predictions\baseline_2\gru\summary_gru_baseline2_h7_windows7_14_targets_raw_log1p.csv"
    df = pd.read_csv(path)

    # Map cột GRU → format chung của results
    df_out = pd.DataFrame({
        "temporal_type": df["temporal_type"],          # unit / weight
        "lag_window": df["window"],                    # 7 / 14
        "horizon": df["horizon"],                      # 7
        "model_family": "gru_sequence",
        "graph_type": "no_graph",
        "variant": "baseline_2_gru",
        "tag": df["tag"],                              # gru_baseline2_h7_w7_log1p_unit ...
        "edge_view": None,
        "target_transform": df["target_type"],         # log1p hoặc raw
        "clip_pred": False,
        "use_softplus_output": False,

        "MAE_train": df["train_mae"],
        "RMSE_train": df["train_rmse"],
        "MAPE_train": df["train_mape"],
        "sMAPE_train": df["train_smape"],

        "MAE_val": df["val_mae"],
        "RMSE_val": df["val_rmse"],
        "MAPE_val": df["val_mape"],
        "sMAPE_val": df["val_smape"],

        "MAE_test": df["test_mae"],
        "RMSE_test": df["test_rmse"],
        "MAPE_test": df["test_mape"],
        "sMAPE_test": df["test_smape"],
    })

    return df_out
def load_baseline3_xgb_graph() -> pd.DataFrame:
    path = (
        PROC_DIR
        / "predictions"
        / "baseline_3"
        / "summary_xgb_graph_baseline3_targets_lags_graphmodes.csv"
    )
    if not path.exists():
        print(f"[BL3] Not found: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df["model_family"] = "xgb_graph"
    # giả sử file này đã có cột graph_mode/graph_type/edge_view; nếu không thì patch:
    if "graph_type" not in df.columns:
        # ví dụ: nếu có cột graph_mode thì copy qua
        if "graph_mode" in df.columns:
            df["graph_type"] = df["graph_mode"]
        else:
            df["graph_type"] = "graph_xgb"
    if "edge_view" not in df.columns:
        df["edge_view"] = None
    if "target_transform" not in df.columns and "target_type" in df.columns:
        df["target_transform"] = df["target_type"]
    if "clip_pred" not in df.columns:
        df["clip_pred"] = False
    if "use_softplus_output" not in df.columns:
        df["use_softplus_output"] = False
    if "variant" not in df.columns:
        df["variant"] = "baseline_3_xgb_graph"
    return df


def load_gnn_summary() -> pd.DataFrame:
    path = PROC_DIR / "predictions" / "gnn_baselines" / "summary_gnn_baselines_all_variants.csv"
    if not path.exists():
        print(f"[GNN] Not found: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df["model_family"] = "gnn"
    if "graph_type" not in df.columns:
        def _graph_type_from_variant(v: str) -> str:
            if "projected" in v:
                return "projected"
            if "homo5" in v:
                return "homo5"
            if "hetero5" in v:
                return "hetero5"
            return "gnn"
        df["graph_type"] = df["variant"].apply(_graph_type_from_variant)
    if "edge_view" not in df.columns:
        df["edge_view"] = None
    return df


def harmonize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Đưa về schema chung để concat."""
    required_cols = [
        "temporal_type",
        "lag_window",
        "horizon",

        "model_family",
        "graph_type",
        "variant",
        "tag",
        "edge_view",

        "target_transform",
        "clip_pred",
        "use_softplus_output",

        "MAE_train",
        "RMSE_train",
        "MAPE_train",
        "sMAPE_train",
        "MAE_val",
        "RMSE_val",
        "MAPE_val",
        "sMAPE_val",
        "MAE_test",
        "RMSE_test",
        "MAPE_test",
        "sMAPE_test",
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = pd.NA
    return df[required_cols]


def main():
    dfs = []

    df_naive = load_naive_last()
    if not df_naive.empty:
        dfs.append(harmonize_columns(df_naive))

    df_b1 = load_baseline1_xgb()
    if not df_b1.empty:
        dfs.append(harmonize_columns(df_b1))

    df_b2 = load_baseline2_gru()
    if not df_b2.empty:
        dfs.append(harmonize_columns(df_b2))

    df_b3 = load_baseline3_xgb_graph()
    if not df_b3.empty:
        dfs.append(harmonize_columns(df_b3))

    df_gnn = load_gnn_summary()
    if not df_gnn.empty:
        dfs.append(harmonize_columns(df_gnn))

    if not dfs:
        print("No summary files found, nothing to aggregate.")
        return

    df_all = pd.concat(dfs, ignore_index=True)

    df_all = df_all.sort_values(
        [
            "temporal_type",
            "lag_window",
            "horizon",
            "model_family",
            "graph_type",
            "edge_view",
            "target_transform",
            "use_softplus_output",
            "clip_pred",
            "tag",
        ]
    )

    out_dir = PROC_DIR / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary_all_baselines.csv"
    df_all.to_csv(out_path, index=False)

    print(f"Aggregated {len(df_all)} rows into {out_path}")
    print(df_all.head())


if __name__ == "__main__":
    main()