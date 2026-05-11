import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error

from config.config import DEFAULT_EXPERIMENTS

BASE_DIR = Path(r"D:\SupplyGraph_experiment - S1\data\processed\predictions")
BASE_RAW_DIR = Path(r"D:\SupplyGraph_experiment - S1\data\processed\base")
TEMPORAL_TYPE = "unit"
SEEDS = [0, 1, 2]  # chỉnh theo seeds anh có


# --------------------------------------------------
# 0. Mapping day index -> date thực từ base_raw
# --------------------------------------------------
def load_node_day_date_mapping(horizon: int, temporal_type: str = "unit") -> pd.DataFrame:
    """
    Đọc base_raw_h{H}_{temporal_type}.parquet và trả về mapping
    (node_id, day) -> date thực.
    """
    path = BASE_RAW_DIR / f"base_raw_h{horizon}_{temporal_type}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Base RAW not found at {path}")

    df_raw = pd.read_parquet(path)
    df_raw["date"] = pd.to_datetime(df_raw["date"])
    map_df = df_raw[["node_id", "day", "date"]].drop_duplicates()
    return map_df


# --------------------------------------------------
# 1. Chuẩn hóa cột date cho các baseline
# --------------------------------------------------
def ensure_datetime_date(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """
    Với các baseline có 'date' là datetime/string:
    - Convert sang datetime và giữ nguyên cột 'date'.
    - Nếu 'date' đã là datetime -> trả về luôn.
    - Nếu 'date' là numeric -> raise (trường hợp này là baseline 4, xử lý riêng).
    """
    df = df.copy()

    if "date" not in df.columns:
        raise ValueError(f"{source_name}: missing 'date' column")

    col = df["date"]

    # datetime: ok
    if np.issubdtype(col.dtype, np.datetime64):
        df["date"] = pd.to_datetime(df["date"])
        return df

    # string / object: parse datetime
    if col.dtype == "object" or col.dtype == "string":
        try:
            df["date"] = pd.to_datetime(df["date"])
            return df
        except Exception as e:
            raise ValueError(f"{source_name}: cannot parse 'date' to datetime: {e}")

    # numeric ở đây là bất thường (trừ baseline 4), raise để bắt bug
    if np.issubdtype(col.dtype, np.number):
        raise ValueError(
            f"{source_name}: 'date' is numeric; this source should use datetime. "
            f"Use mapping via base_raw instead."
        )

    raise ValueError(f"{source_name}: unsupported date dtype: {col.dtype}")


def map_gnn_date_from_day_index(
    df_gnn: pd.DataFrame,
    map_node_day: pd.DataFrame,
    source_name: str,
) -> pd.DataFrame:
    """
    Baseline 4: 'date' trong file prediction là index ngày (int) -> map về date thực.
    - Rename 'date' -> 'day' (int).
    - Join với map_node_day trên (node_id, day) để lấy 'date' datetime.
    """
    df_local = df_gnn.copy()
    if "date" not in df_local.columns:
        raise ValueError(f"{source_name}: missing 'date' column (expected day index)")

    # 'date' ở đây phải là numeric index (177, 178, ...)
    if not np.issubdtype(df_local["date"].dtype, np.number):
        raise ValueError(
            f"{source_name}: expected numeric 'date' as day index, "
            f"got dtype {df_local['date'].dtype}"
        )

    df_local = df_local.rename(columns={"date": "day"})
    df_local["day"] = df_local["day"].astype(int)

    df_local = df_local.merge(
        map_node_day,
        on=["node_id", "day"],
        how="left",
    )
    missing = df_local["date"].isna().sum()
    if missing > 0:
        print(
            f"[WARN] {source_name}: {missing} rows have no mapped date "
            f"(check base_raw mapping)."
        )
    return df_local


# -----------------------------
# 2. Đọc test predictions cho 1 (H, L, seed)
# -----------------------------
def load_test_predictions(horizon: int, lag: int, seed: int) -> pd.DataFrame:
    """
    Đọc test predictions cho 1 cấu hình (horizon, lag, seed, temporal_type=unit).
    Trả về DataFrame khung từ XGB baseline 1 + các cột prediction khác.
    - Nếu 'date' là datetime/string: giữ nguyên.
    - Nếu 'date' là int index (baseline 4): map về date thực bằng base_raw.
    """
    H = horizon
    L = lag
    s = seed

    # mapping (node_id, day) -> date thực, dùng cho baseline 4
    map_node_day = load_node_day_date_mapping(H, TEMPORAL_TYPE)

    # 1) Plain XGB (baseline 1) - khung test set
    path_xgb = (
        BASE_DIR
        / "baseline_1"
        / "csv"
        / TEMPORAL_TYPE
        / f"xgb_tabular_h{H}_lag{L}_raw_test_predictions_{TEMPORAL_TYPE}_seed{s}.csv"
    )
    df = pd.read_csv(path_xgb)
    df["y_true"] = df["y_true"].astype(float)
    df = df.rename(columns={"y_pred": "pred_xgb"})
    df = ensure_datetime_date(df, "baseline_1")
    print(f"[LOAD] XGB baseline 1 from {path_xgb}, rows={len(df)}")

    # 2) XGB + graph proj/homo/hetero (baseline 3)
    def load_xgb_graph(mode: str, col_name: str):
        path = (
            BASE_DIR
            / "baseline_3"
            / "csv"
            / TEMPORAL_TYPE
            / mode
            / f"xgb_graph_{mode}_h{H}_lag{L}_raw_test_predictions_{TEMPORAL_TYPE}_seed{s}.csv"
        )
        df_local = pd.read_csv(path)
        df_local = df_local.rename(columns={"y_pred": col_name})
        df_local = ensure_datetime_date(df_local, f"baseline_3_{mode}")
        print(f"[LOAD] XGB+Graph {mode} from {path}, rows={len(df_local)}")
        return df_local

    df_xgb_proj = load_xgb_graph("proj", "pred_xgb_proj")
    df_xgb_homo = load_xgb_graph("homo", "pred_xgb_homo")
    df_xgb_hetero = load_xgb_graph("hetero", "pred_xgb_hetero")

    # 3) GNN projected (baseline 4) - date là day index -> map từ base_raw
    def load_gnn_proj(name: str, col_name: str):
        path = (
            BASE_DIR
            / "baseline_4"
            / "unit_raw"
            / "csv"
            / "projected"
            / f"gnn_projected_{name}_h{H}_lag{L}_unit_raw_seed{s}_test_predictions.csv"
        )
        df_local = pd.read_csv(path)
        df_local = df_local.rename(columns={"y_pred": col_name})
        df_local = map_gnn_date_from_day_index(
            df_local,
            map_node_day,
            f"baseline_4_proj_{name}",
        )
        print(f"[LOAD] GNN projected {name} from {path}, rows={len(df_local)}")
        return df_local

    df_gnn_proj_group = load_gnn_proj("same_group", "pred_gnn_proj_group")
    df_gnn_proj_plant = load_gnn_proj("same_plant", "pred_gnn_proj_plant")
    df_gnn_proj_storage = load_gnn_proj("same_storage", "pred_gnn_proj_storage")
    df_gnn_proj_subgroup = load_gnn_proj("same_subgroup", "pred_gnn_proj_subgroup")

    # 4) GNN homo5 / hetero5 (baseline 4) - cũng map từ day index
    path_gnn_homo5 = (
        BASE_DIR
        / "baseline_4"
        / "unit_raw"
        / "csv"
        / "homo5"
        / f"gnn_homo5_h{H}_lag{L}_unit_raw_seed{s}_test_predictions.csv"
    )
    df_gnn_homo5 = pd.read_csv(path_gnn_homo5)
    df_gnn_homo5 = df_gnn_homo5.rename(columns={"y_pred": "pred_gnn_homo5"})
    df_gnn_homo5 = map_gnn_date_from_day_index(
        df_gnn_homo5,
        map_node_day,
        "baseline_4_homo5",
    )
    print(f"[LOAD] GNN homo5 from {path_gnn_homo5}, rows={len(df_gnn_homo5)}")

    path_gnn_hetero5 = (
        BASE_DIR
        / "baseline_4"
        / "unit_raw"
        / "csv"
        / "hetero5"
        / f"gnn_hetero5_h{H}_lag{L}_unit_raw_seed{s}_test_predictions.csv"
    )
    df_gnn_hetero5 = pd.read_csv(path_gnn_hetero5)
    df_gnn_hetero5 = df_gnn_hetero5.rename(columns={"y_pred": "pred_gnn_hetero5"})
    df_gnn_hetero5 = map_gnn_date_from_day_index(
        df_gnn_hetero5,
        map_node_day,
        "baseline_4_hetero5",
    )
    print(f"[LOAD] GNN hetero5 from {path_gnn_hetero5}, rows={len(df_gnn_hetero5)}")

    # 5) XGB + GNN embedding (baseline 5)
    def load_xgb_embed(mode: str, col_name: str):
        path = (
            BASE_DIR
            / "baseline_5"
            / "csv"
            / TEMPORAL_TYPE
            / mode
            / f"xgb_gnn_embed_{mode}_h{H}_lag{L}_raw_test_predictions_{TEMPORAL_TYPE}_seed{s}.csv"
        )
        df_local = pd.read_csv(path)
        df_local = df_local.rename(columns={"y_pred": col_name})
        df_local = ensure_datetime_date(df_local, f"baseline_5_{mode}")
        print(f"[LOAD] XGB+GNN embed {mode} from {path}, rows={len(df_local)}")
        return df_local

    df_xgb_embed_proj = load_xgb_embed("proj", "pred_xgb_embed_proj")
    df_xgb_embed_homo = load_xgb_embed("homo", "pred_xgb_embed_homo")
    df_xgb_embed_hetero = load_xgb_embed("hetero", "pred_xgb_embed_hetero")

    # 6) XGB tabular + graph features (baseline 7)
    def load_xgb_graphfeat(suffix: str, col_name: str):
        path = (
            BASE_DIR
            / "baseline_7"
            / "csv"
            / TEMPORAL_TYPE
            / f"xgb_tabular_graphfeat_{suffix}_h{H}_lag{L}_{TEMPORAL_TYPE}_test_predictions_seed{s}.csv"
        )
        df_local = pd.read_csv(path)
        df_local = df_local.rename(columns={"y_pred": col_name})
        df_local = ensure_datetime_date(df_local, f"baseline_7_{suffix}")
        print(f"[LOAD] XGB+graphfeat {suffix} from {path}, rows={len(df_local)}")
        return df_local

    df_xgb_graphfeat_proj = load_xgb_graphfeat("projected", "pred_xgb_graphfeat_proj")
    df_xgb_graphfeat_homo = load_xgb_graphfeat("homo5", "pred_xgb_graphfeat_homo")
    df_xgb_graphfeat_hetero = load_xgb_graphfeat("hetero5", "pred_xgb_graphfeat_hetero")
    # 7) Residual GNN (baseline 6) - giả sử cũng dùng day index như baseline 4
    # 7) Residual GNN (baseline 6)
    def load_residual_proj(name: str, col_name: str):
        """
        name: 'same_group', 'same_plant', ... (projected residual GNN).
        Ví dụ file:
          baseline6_residual_proj_same_group_h7_lag7_unit_raw_seed0_test_predictions.csv
        """
        path = (
            BASE_DIR
            / "baseline_6"
            / "unit_raw"
            / "projected"
            / f"baseline6_residual_proj_{name}_h{H}_lag{L}_unit_raw_seed{s}_test_predictions.csv"
        )
        df_local = pd.read_csv(path)
        df_local = df_local.rename(columns={"y_pred": col_name})
        # date ở đây là day index (int) -> map sang date thực giống baseline 4
        df_local = map_gnn_date_from_day_index(
            df_local,
            map_node_day,
            f"baseline_6_proj_{name}",
        )
        print(f"[LOAD] Residual GNN proj {name} from {path}, rows={len(df_local)}")
        return df_local

    df_residual_proj_group = load_residual_proj("same_group", "pred_residual_proj_group")
    df_residual_proj_plant = load_residual_proj("same_plant", "pred_residual_proj_plant")
    df_residual_proj_storage = load_residual_proj("same_storage", "pred_residual_proj_storage")
    df_residual_proj_subgroup = load_residual_proj("same_subgroup", "pred_residual_proj_subgroup")

    def load_residual_gnn(mode: str, col_name: str):
            """
            mode: 'homo5' hoặc 'hetero5'.
            Ví dụ file:
            baseline_6/unit_raw/hetero5/baseline6_residual_hetero5_h7_lag7_unit_raw_seed0_test_predictions.csv
            """
            path = (
                BASE_DIR
                / "baseline_6"
                / "unit_raw"
                / mode
                / f"baseline6_residual_{mode}_h{H}_lag{L}_unit_raw_seed{s}_test_predictions.csv"
            )
            df_local = pd.read_csv(path)
            df_local = df_local.rename(columns={"y_pred": col_name})
            df_local = map_gnn_date_from_day_index(
                df_local,
                map_node_day,
                f"baseline_6_{mode}",
            )
            print(f"[LOAD] Residual GNN {mode} from {path}, rows={len(df_local)}")
            return df_local
    
    df_residual_homo5 = load_residual_gnn("homo5", "pred_residual_homo5")
    df_residual_hetero5 = load_residual_gnn("hetero5", "pred_residual_hetero5")
    # Merge helper: luôn merge theo (node_id, date)
    def merge_pred(df_main, df_other, col_pred):
        return df_main.merge(
            df_other[["node_id", "date", col_pred]],
            on=["node_id", "date"],
            how="inner",
        )

    # Merge tất cả vào df (khung baseline 1)
    df = merge_pred(df, df_xgb_proj, "pred_xgb_proj")
    df = merge_pred(df, df_xgb_homo, "pred_xgb_homo")
    df = merge_pred(df, df_xgb_hetero, "pred_xgb_hetero")

    df = merge_pred(df, df_gnn_proj_group, "pred_gnn_proj_group")
    df = merge_pred(df, df_gnn_proj_plant, "pred_gnn_proj_plant")
    df = merge_pred(df, df_gnn_proj_storage, "pred_gnn_proj_storage")
    df = merge_pred(df, df_gnn_proj_subgroup, "pred_gnn_proj_subgroup")

    df = merge_pred(df, df_gnn_homo5, "pred_gnn_homo5")
    df = merge_pred(df, df_gnn_hetero5, "pred_gnn_hetero5")

    df = merge_pred(df, df_xgb_embed_proj, "pred_xgb_embed_proj")
    df = merge_pred(df, df_xgb_embed_homo, "pred_xgb_embed_homo")
    df = merge_pred(df, df_xgb_embed_hetero, "pred_xgb_embed_hetero")

    df = merge_pred(df, df_xgb_graphfeat_proj, "pred_xgb_graphfeat_proj")
    df = merge_pred(df, df_xgb_graphfeat_homo, "pred_xgb_graphfeat_homo")
    df = merge_pred(df, df_xgb_graphfeat_hetero, "pred_xgb_graphfeat_hetero")
    df = merge_pred(df, df_residual_proj_group, "pred_residual_proj_group")
    df = merge_pred(df, df_residual_proj_plant, "pred_residual_proj_plant")
    df = merge_pred(df, df_residual_proj_storage, "pred_residual_proj_storage")
    df = merge_pred(df, df_residual_proj_subgroup, "pred_residual_proj_subgroup")
    df = merge_pred(df, df_residual_homo5, "pred_residual_homo5")
    df = merge_pred(df, df_residual_hetero5, "pred_residual_hetero5")
    print(f"[INFO] Final merged test set shape for H={H}, L={L}, seed={s}: {df.shape}")
    df["horizon"] = H
    df["lag_window"] = L
    df["seed"] = s
    return df


# -----------------------------
# 3. Zero-ratio buckets theo node_id
# -----------------------------
def compute_zero_ratio_buckets(df_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Tính zero_ratio, zero_bucket per node_id:
      - high_zero: zero_ratio >= 0.75
      - mid_zero : 0.25 <= zero_ratio < 0.75
      - low_zero : zero_ratio < 0.25
    """
    stats = (
        df_test.groupby("node_id")["y_true"]
        .agg(
            n_days="count",
            n_zero=lambda s: (s == 0).sum(),
        )
        .reset_index()
    )
    stats["zero_ratio"] = stats["n_zero"] / stats["n_days"]

    def bucket(z: float) -> str:
        if z >= 0.75:
            return "high_zero"
        elif z >= 0.25:
            return "mid_zero"
        else:
            return "low_zero"

    stats["zero_bucket"] = stats["zero_ratio"].apply(bucket)
    df_merged = df_test.merge(stats, on="node_id", how="left")
    return df_merged, stats


# -----------------------------
# 4. Tính MAE theo zero_bucket
# -----------------------------
def compute_mae_by_zero_bucket(df: pd.DataFrame, horizon: int, lag: int, seed: int) -> pd.DataFrame:
    model_cols = [
        "pred_xgb",
        "pred_xgb_proj",
        "pred_xgb_homo",
        "pred_xgb_hetero",
        "pred_gnn_proj_group",
        "pred_gnn_proj_plant",
        "pred_gnn_proj_storage",
        "pred_gnn_proj_subgroup",
        "pred_gnn_homo5",
        "pred_gnn_hetero5",
        "pred_xgb_embed_proj",
        "pred_xgb_embed_homo",
        "pred_xgb_embed_hetero",
        "pred_xgb_graphfeat_proj",
        "pred_xgb_graphfeat_homo",
        "pred_xgb_graphfeat_hetero",
        "pred_residual_proj_group",
        "pred_residual_proj_plant",
        "pred_residual_proj_storage",
        "pred_residual_proj_subgroup",
        "pred_residual_homo5",
        "pred_residual_hetero5",
    ]

    name_map = {
        "pred_xgb": "XGB",
        "pred_xgb_proj": "XGB+Graph proj",
        "pred_xgb_homo": "XGB+Graph homo",
        "pred_xgb_hetero": "XGB+Graph hetero",
        "pred_gnn_proj_group": "GNN proj same_group",
        "pred_gnn_proj_plant": "GNN proj same_plant",
        "pred_gnn_proj_storage": "GNN proj same_storage",
        "pred_gnn_proj_subgroup": "GNN proj same_subgroup",
        "pred_gnn_homo5": "GNN homo5",
        "pred_gnn_hetero5": "GNN hetero5",
        "pred_xgb_embed_proj": "XGB + GNN embed proj",
        "pred_xgb_embed_homo": "XGB + GNN embed homo",
        "pred_xgb_embed_hetero": "XGB + GNN embed hetero",
        "pred_xgb_graphfeat_proj": "XGB + graphfeat proj",
        "pred_xgb_graphfeat_homo": "XGB + graphfeat homo",
        "pred_xgb_graphfeat_hetero": "XGB + graphfeat hetero",
        "pred_residual_proj_group": "Residual GNN proj same_group",
        "pred_residual_proj_plant": "Residual GNN proj same_plant",
        "pred_residual_proj_storage": "Residual GNN proj same_storage",
        "pred_residual_proj_subgroup": "Residual GNN proj same_subgroup",
        "pred_residual_homo5": "Residual GNN homo5",
        "pred_residual_hetero5": "Residual GNN hetero5",
    }

    rows = []
    for zb in ["high_zero", "mid_zero", "low_zero"]:
        sub = df[df["zero_bucket"] == zb]
        if sub.empty:
            continue
        y = sub["y_true"].values
        for col in model_cols:
            if col not in sub.columns:
                continue
            mae = mean_absolute_error(y, sub[col].values)
            rows.append(
                {
                    "horizon": horizon,
                    "lag_window": lag,
                    "seed": seed,
                    "zero_bucket": zb,
                    "model": name_map.get(col, col),
                    "MAE": mae,
                    "n_samples": len(sub),
                }
            )

    return pd.DataFrame(rows)


# -----------------------------
# 5. Main
# -----------------------------
def main():
    out_diag_dir = BASE_DIR / "diagnostics"
    out_diag_dir.mkdir(parents=True, exist_ok=True)

    all_mae_rows = []
    all_sku_stats = []

    for exp in DEFAULT_EXPERIMENTS:
        if exp.temporal_type != TEMPORAL_TYPE:
            continue
        for H in exp.horizons:
            for L in exp.lag_windows:
                print(f"\n===== H={H}, L={L}, temporal_type={TEMPORAL_TYPE} =====")
                for s in SEEDS:
                    print(f"  -- Seed {s} --")
                    df_test = load_test_predictions(H, L, s)
                    if df_test is None or df_test.empty:
                        continue  # không có dữ liệu, bỏ qua seed này

                    df_test_zb, sku_stats = compute_zero_ratio_buckets(df_test)

                    df_test_zb.to_csv(
                        out_diag_dir
                        / f"test_with_zero_buckets_h{H}_lag{L}_{TEMPORAL_TYPE}_seed{s}.csv",
                        index=False,
                    )
                    sku_stats.to_csv(
                        out_diag_dir
                        / f"sku_zero_ratio_h{H}_lag{L}_{TEMPORAL_TYPE}_seed{s}.csv",
                        index=False,
                    )

                    df_mae = compute_mae_by_zero_bucket(df_test_zb, H, L, s)
                    if df_mae.empty:
                        print(f"    [WARN] No MAE rows for H={H}, L={L}, seed={s}, skipping.")
                        continue

                    df_mae.to_csv(
                        out_diag_dir
                        / f"mae_by_zero_bucket_h{H}_lag{L}_{TEMPORAL_TYPE}_seed{s}.csv",
                        index=False,
                    )

                    all_mae_rows.append(df_mae)
                    sku_stats["horizon"] = H
                    sku_stats["lag_window"] = L
                    sku_stats["seed"] = s
                    all_sku_stats.append(sku_stats)

    if not all_mae_rows:
        print("No MAE rows computed; check predictions.")
        return

    df_mae_all = pd.concat(all_mae_rows, ignore_index=True)
    df_sku_all = pd.concat(all_sku_stats, ignore_index=True)

    # Lưu per-seed
    df_mae_all.to_csv(
        out_diag_dir
        / f"mae_by_zero_bucket_all_horizons_lags_{TEMPORAL_TYPE}_per_seed.csv",
        index=False,
    )
    df_sku_all.to_csv(
        out_diag_dir
        / f"sku_zero_ratio_all_horizons_lags_{TEMPORAL_TYPE}_per_seed.csv",
        index=False,
    )

    # Aggregate mean/std MAE over seeds
    df_agg = (
        df_mae_all
        .groupby(["horizon", "lag_window", "zero_bucket", "model"], as_index=False)
        .agg(
            mean_MAE=("MAE", "mean"),
            std_MAE=("MAE", "std"),
            n_seeds=("seed", "nunique"),
        )
    )
    df_agg.to_csv(
        out_diag_dir
        / f"mae_by_zero_bucket_all_horizons_lags_{TEMPORAL_TYPE}_agg_over_seeds.csv",
        index=False,
    )

    print("\n[SAVE] Per-seed MAE ->",
          out_diag_dir / f"mae_by_zero_bucket_all_horizons_lags_{TEMPORAL_TYPE}_per_seed.csv")
    print("[SAVE] Aggregated MAE over seeds ->",
          out_diag_dir / f"mae_by_zero_bucket_all_horizons_lags_{TEMPORAL_TYPE}_agg_over_seeds.csv")
    print("[DONE] Zero-heavy demand diagnostic (zero_ratio buckets) for all H, L, seeds.")


if __name__ == "__main__":
    main()