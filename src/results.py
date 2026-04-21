from pathlib import Path
import glob
import pandas as pd

from config.config import PROC_DIR


def load_naive_last() -> pd.DataFrame:
    path = PROC_DIR / "predictions_naive" / "summary_naive_last_t0.csv"
    if not path.exists():
        print(f"[NAIVE] Not found: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df["model_family"] = "naive_last_t0"
    df["graph_type"] = "no_graph"
    df["variant"] = "naive_last_t0"
    df["edge_view"] = None

    if "target_transform" not in df.columns:
        df["target_transform"] = "raw"

    return df


def load_baseline1_xgb() -> pd.DataFrame:
    path = (
        PROC_DIR
        / "predictions"
        / "baseline_1"
        / "summary_xgb_tabular_baseline1_raw_targets_lags.csv"
    )
    if not path.exists():
        print(f"[BL1] Not found: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df["model_family"] = "xgb_tabular"
    df["graph_type"] = "no_graph"
    df["edge_view"] = None

    if "target_transform" not in df.columns:
        if "target_type" in df.columns:
            df["target_transform"] = df["target_type"]
        else:
            df["target_transform"] = "raw"

    if "variant" not in df.columns:
        df["variant"] = "baseline_1_xgb_tabular"

    return df


def load_baseline2_gru() -> pd.DataFrame:
    base_dir = PROC_DIR / "predictions" / "baseline_2" / "gru"
    pattern = str(base_dir / "summary_gru_baseline2_*.csv")
    files = glob.glob(pattern)

    if not files:
        print(f"[BL2] No GRU summary files found under {pattern}")
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"[BL2] Failed to read {f}: {e}")
            continue

        if "window" not in df.columns and "lag_window" in df.columns:
            df = df.rename(columns={"lag_window": "window"})

        if "output_transform" not in df.columns:
            if "target_transform" in df.columns:
                df["output_transform"] = df["target_transform"]
            elif "target_type" in df.columns:
                df["output_transform"] = df["target_type"]
            else:
                df["output_transform"] = "raw"

        def _norm_tt(x: str) -> str:
            if isinstance(x, str):
                xl = x.lower()
                if "log1p" in xl:
                    return "log1p"
                if "soft" in xl:
                    return "softplus"
                if "raw" in xl:
                    return "raw"
            return "raw"

        df["output_transform"] = df["output_transform"].apply(_norm_tt)

        df_out = pd.DataFrame(
            {
                "temporal_type": df.get("temporal_type"),
                "lag_window": df.get("window"),
                "horizon": df.get("horizon"),

                "model_family": "gru_sequence",
                "graph_type": "no_graph",
                "variant": "baseline_2_gru",
                "tag": df.get("tag"),
                "edge_view": None,

                "target_transform": df.get("output_transform"),

                "MAE_train": df.get("train_mae"),
                "RMSE_train": df.get("train_rmse"),
                "MAPE_train": df.get("train_mape"),
                "sMAPE_train": df.get("train_smape"),

                "MAE_val": df.get("val_mae"),
                "RMSE_val": df.get("val_rmse"),
                "MAPE_val": df.get("val_mape"),
                "sMAPE_val": df.get("val_smape"),

                "MAE_test": df.get("test_mae"),
                "RMSE_test": df.get("test_rmse"),
                "MAPE_test": df.get("test_mape"),
                "sMAPE_test": df.get("test_smape"),
            }
        )

        dfs.append(df_out)

    if not dfs:
        print(f"[BL2] No valid GRU summary loaded from {base_dir}")
        return pd.DataFrame()

    df_all = pd.concat(dfs, ignore_index=True)
    print(f"[BL2] Loaded {len(df_all)} GRU summary rows from {len(files)} files")
    return df_all


def load_baseline3_xgb_graph() -> pd.DataFrame:
    path = (
        PROC_DIR
        / "predictions"
        / "baseline_3"
        / "summary_xgb_graph_baseline3_raw_lags_graphmodes.csv"
    )
    if not path.exists():
        print(f"[BL3] Not found: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df["model_family"] = "xgb_graph"

    if "graph_type" not in df.columns:
        if "graph_mode" in df.columns:
            df["graph_type"] = df["graph_mode"]
        else:
            df["graph_type"] = "graph_xgb"

    if "edge_view" not in df.columns:
        df["edge_view"] = None

    if "target_transform" not in df.columns:
        if "target_type" in df.columns:
            df["target_transform"] = df["target_type"]
        else:
            df["target_transform"] = "raw"

    if "variant" not in df.columns:
        df["variant"] = "baseline_3_xgb_graph"

    return df


def load_gnn_summary() -> pd.DataFrame:
    """
    Load tất cả summary baseline_4 GNN:
        PROC_DIR/predictions/baseline_4/*/summary_baseline_4_*.csv
    Mỗi subfolder tương ứng 1 (temporal_type, mode) như:
        baseline_4/unit_raw/summary_baseline_4_unit_raw.csv
    """
    base_dir = PROC_DIR / "predictions" / "baseline_4"
    pattern = str(base_dir / "*" / "summary_baseline_4_*.csv")
    files = glob.glob(pattern)

    if not files:
        print(f"[GNN] No GNN summary files found under {pattern}")
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"[GNN] Failed to read {f}: {e}")
            continue

        df["model_family"] = "gnn"

        if "graph_type" not in df.columns:
            def _graph_type_from_variant(v: str) -> str:
                if isinstance(v, str):
                    v = v.lower()
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

        if "target_transform" not in df.columns:
            df["target_transform"] = "raw"
        else:
            def _norm_tt(x: str) -> str:
                if isinstance(x, str):
                    xl = x.lower()
                    if "log1p" in xl:
                        return "log1p"
                    if "soft" in xl:
                        return "softplus"
                    if "raw" in xl:
                        return "raw"
                return "raw"
            df["target_transform"] = df["target_transform"].apply(_norm_tt)

        dfs.append(df)

    if not dfs:
        print(f"[GNN] No valid GNN summary loaded from {base_dir}")
        return pd.DataFrame()

    df_all = pd.concat(dfs, ignore_index=True)
    print(f"[GNN] Loaded {len(df_all)} GNN summary rows from {len(files)} files")
    return df_all

def load_baseline5_xgb_gnn_embed() -> pd.DataFrame:
    """
    Load summary baseline 5: XGB + GNN embeddings.
    File: PROC_DIR/predictions/baseline_5/summary_xgb_gnn_embed_baseline5_raw_lags_graphmodes.csv
    """
    path = (
        PROC_DIR
        / "predictions"
        / "baseline_5"
        / "summary_xgb_gnn_embed_baseline5_raw_lags_graphmodes.csv"
    )
    if not path.exists():
        print(f"[BL5] Not found: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)

    # Chuẩn hóa metadata
    df["model_family"] = "xgb_gnn_embed"

    if "graph_type" not in df.columns:
        if "graph_mode" in df.columns:
            df["graph_type"] = df["graph_mode"]
        else:
            df["graph_type"] = "graph_xgb_gnn_embed"

    if "edge_view" not in df.columns:
        df["edge_view"] = None

    if "target_transform" not in df.columns:
        if "target_type" in df.columns:
            df["target_transform"] = df["target_type"]
        else:
            df["target_transform"] = "raw"

    if "variant" not in df.columns:
        df["variant"] = "baseline_5_xgb_gnn_embed"

    return df

def harmonize_columns(df: pd.DataFrame) -> pd.DataFrame:
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
    df_b5 = load_baseline5_xgb_gnn_embed()
    if not df_b5.empty:
        dfs.append(harmonize_columns(df_b5))
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
            "variant",
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