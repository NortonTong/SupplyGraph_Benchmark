import pandas as pd
import numpy as np
from pathlib import Path

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROC_DIR = DATA_DIR / "processed"
RUN_SUMMARY = []



def load_dataset(horizon, variant, graph_variant=None):
    if variant == "baseline_1":
        path = PROC_DIR / "baseline_1" / f"xgboost_h{horizon}.parquet"
    elif variant == "baseline_2":
        path = PROC_DIR / "baseline_2" / f"xgboost_h{horizon}_graph_stats_{graph_variant}.parquet"
    elif variant == "baseline_3":
        # dùng cùng file với baseline_2
        path = PROC_DIR / "baseline_2" / f"xgboost_h{horizon}_graph_stats_{graph_variant}.parquet"
    else:
        raise ValueError("Unknown variant")
    return pd.read_parquet(path)


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


def one_hot_encode_train_val_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    df_train = df[df["split"] == "train"].copy()
    df_val = df[df["split"] == "val"].copy()
    df_test = df[df["split"] == "test"].copy()

    cat_cols = [
        "group", "sub_group",
        "plant", "storage_location",
        "day_of_week", "month",
        "is_weekend",
    ]
    cat_cols = [c for c in cat_cols if c in df.columns]

    df_train_enc = pd.get_dummies(df_train, columns=cat_cols, drop_first=False)
    df_val_enc = pd.get_dummies(df_val, columns=cat_cols, drop_first=False)
    df_test_enc = pd.get_dummies(df_test, columns=cat_cols, drop_first=False)
    df_val_enc = df_val_enc.reindex(columns=df_train_enc.columns, fill_value=0)
    df_test_enc = df_test_enc.reindex(columns=df_train_enc.columns, fill_value=0)

    return df_train_enc, df_val_enc, df_test_enc

def rolling_origin_evaluation(horizon: int,
                              origins: list[int],
                              variant: str,
                              graph_variant: str | None = None) -> None:
    df = load_dataset(horizon=horizon, variant=variant, graph_variant=graph_variant)

    # We will ignore existing 'split' and redefine train/test per origin
    errors = []

    for T in origins:
        print(f"\n=== Rolling origin at day {T}, horizon {horizon} ===")

        # Train set: all data with day <= T
        df_train = df[df["day"] <= T].copy()
        # Test set: exact day = T + horizon
        df_test = df[df["day"] == T + horizon].copy()

        if df_test.empty:
            print(f"  No test samples for origin {T} (day={T+horizon}). Skipping.")
            continue

        # One-hot encode categorical using train as reference
        df_train_enc, df_val_dummy, df_test_enc = one_hot_encode_train_val_test_for_rolling(
            df_train, df_test
        )

        X_train, y_train = prepare_features(df_train_enc)
        X_test, y_test = prepare_features(df_test_enc)

        model = XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mae_val = mean_absolute_error(y_test, y_pred)
        rmse_val = root_mean_squared_error(y_test, y_pred)

        print(f"  MAE  : {mae_val:.4f}")
        print(f"  RMSE : {rmse_val:.4f}")

        errors.append(
            {"origin_day": T, "MAE": mae_val, "RMSE": rmse_val}
        )

    # Summarise across origins
    if errors:
        df_err = pd.DataFrame(errors)
        print("\n=== Rolling-origin summary ===")
        print(df_err)
        print("\nAverage over origins:")
        print(df_err.mean(numeric_only=True))

def one_hot_encode_train_val_test_for_rolling(df_train: pd.DataFrame,
                                              df_test: pd.DataFrame):
    cat_cols = [
        "group", "sub_group",
        "plant", "storage_location",
        "day_of_week", "month",
        "is_weekend",
    ]
    cat_cols = [c for c in cat_cols if c in df_train.columns]

    df_train_enc = pd.get_dummies(df_train, columns=cat_cols, drop_first=False)
    df_test_enc = pd.get_dummies(df_test, columns=cat_cols, drop_first=False)

    df_test_enc = df_test_enc.reindex(columns=df_train_enc.columns, fill_value=0)

    # Return dummy val as placeholder (not used)
    return df_train_enc, None, df_test_enc

def attach_gnn_embeddings_to_split(
    df_split: pd.DataFrame,
    graph_edge_types: list[str],
) -> pd.DataFrame:
    emb_path = PROC_DIR / "gnn_embeddings" / "gnn_node_embeddings_all_views.parquet"
    df_emb = pd.read_parquet(emb_path)

    # chỉ giữ 1 dòng / node_id (trong file này vốn đã vậy, nhưng cho chắc)
    df_emb = df_emb.drop_duplicates(subset=["node_id"], keep="first")

    # tất cả cột embedding
    all_emb_cols = [c for c in df_emb.columns if "_gcn_emb_" in c or "_gin_emb_" in c]

    # chỉ chọn prefix cần cho run này
    keep_prefixes = [et + "_" for et in graph_edge_types]
    emb_cols = [c for c in all_emb_cols if any(c.startswith(p) for p in keep_prefixes)]

    # ép float32
    df_emb[emb_cols] = df_emb[emb_cols].astype("float32")

    # tạo block gọn: node_id + embedding
    emb_block = df_emb[["node_id"] + emb_cols].copy()

    merged = df_split.merge(
        emb_block,
        on="node_id",
        how="left",
        copy=False,
    )
    return merged


def train_and_evaluate_xgboost(
    horizon: int,
    variant: str,
    graph_variant: str | None = None,
    graph_edge_types: list[str] | None = None,
    tag: str | None = None,
) -> None:
    """
    variant:
      - 'baseline_1': no graph stats
      - 'baseline_2': with graph stats (all edges in file), ablated by graph_edge_types
    graph_variant:
      - chỉ dùng nếu bạn vẫn lưu 4 file khác nhau; nếu đã gộp all_graph_stats vào 1 file,
        có thể bỏ qua tham số này.
    graph_edge_types:
      - list prefix graph muốn giữ, ví dụ ['plant', 'sub_group']
      - nếu None: giữ nguyên tất cả feature.
    tag:
      - dùng để log tên cấu hình (no_graph, plant_only, ...)
    """
    if tag is None:
        tag = "default"

    print(f"\n=== Training XGBoost H{horizon}, variant={variant}, tag={tag} ===")
    df = load_dataset(horizon=horizon, variant=variant, graph_variant=graph_variant)

    print(f"[DEBUG] H{horizon} {variant}: df rows={len(df)}, unique(node,date)={df[['node_id','date']].drop_duplicates().shape[0]}")

    df_train_enc, df_val_enc, df_test_enc = one_hot_encode_train_val_test(df)
    print("[DEBUG] splits rows:",
        len(df_train_enc), len(df_val_enc), len(df_test_enc))

    if variant == "baseline_3":
        df_train_enc = attach_gnn_embeddings_to_split(df_train_enc, graph_edge_types)
        df_val_enc   = attach_gnn_embeddings_to_split(df_val_enc,   graph_edge_types)
        df_test_enc  = attach_gnn_embeddings_to_split(df_test_enc,  graph_edge_types)
        print("[DEBUG] after GNN merge:",
            len(df_train_enc), len(df_val_enc), len(df_test_enc))
    if variant == "baseline_3":
        drop_static = True
    else:
        drop_static = False
    # Chuẩn bị X, y với ablation graph
    X_train, y_train = prepare_features_with_ablation(df_train_enc, graph_edge_types, drop_static)
    X_val, y_val = prepare_features_with_ablation(df_val_enc, graph_edge_types, drop_static)
    X_test, y_test = prepare_features_with_ablation(df_test_enc, graph_edge_types, drop_static)

    print(f"Train samples: {X_train.shape[0]}, features: {X_train.shape[1]}")
    print(f"Val   samples: {X_val.shape[0]}")
    print(f"Test  samples: {X_test.shape[0]}")

    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True,
    )

    # Validation
    y_val_pred = model.predict(X_val)
    mae_val = mean_absolute_error(y_val, y_val_pred)
    rmse_val = root_mean_squared_error(y_val, y_val_pred)


    print(f"\n[H{horizon}][{variant}][{tag}] Validation:")
    print(f"  MAE  : {mae_val:.4f}")
    print(f"  RMSE : {rmse_val:.4f}")


    # Test
    y_test_pred = model.predict(X_test)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    rmse_test = root_mean_squared_error(y_test, y_test_pred)


    print(f"\n[H{horizon}][{variant}][{tag}] Test:")
    print(f"  MAE  : {mae_test:.4f}")
    print(f"  RMSE : {rmse_test:.4f}")
    RUN_SUMMARY.append({
    "variant": variant,
    "tag": tag,
    "horizon": horizon,
    "n_features": X_train.shape[1],
    "MAE_val": mae_val,
    "RMSE_val": rmse_val,
    "MAE_test": mae_test,
    "RMSE_test": rmse_test,
    })
    out_dir = PROC_DIR / "predictions" / variant / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "node_id": df_test_enc["node_id"],
        "date": df_test_enc["date"],
        "y_true": y_test,
        "y_pred": y_test_pred,
    }).to_csv(out_dir / f"xgb_h{horizon}_test_predictions.csv", index=False)
    print(f"\nSaved test predictions to {out_dir / f'xgb_h{horizon}_test_predictions.csv'}")

STATIC_GRAPH_COLS = [
    "plant_deg", "plant_clustering", "plant_closeness", "plant_betweenness",
    "group_deg", "group_clustering", "group_closeness", "group_betweenness",
    "sub_group_deg", "sub_group_clustering", "sub_group_closeness", "sub_group_betweenness",
    "storage_location_deg", "storage_location_clustering",
    "storage_location_closeness", "storage_location_betweenness",
]

def prepare_features_with_ablation(
    df: pd.DataFrame,
    graph_edge_types: list[str] | None,
    drop_static_graph: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    target = df["target"].astype(float)

    drop_cols = [
        "target", "split", "node_id", "node_index", "date", "day", "month"
    ]
    if drop_static_graph:
        drop_cols += STATIC_GRAPH_COLS

    X = df.drop(columns=drop_cols, errors="ignore").copy()

    if graph_edge_types is None:
        return X, target

    all_graph_prefixes = [
        "plant_", "product_group_", "sub_group_", "storage_location_",
        "plant_gcn_emb_", "product_group_gcn_emb_",
        "sub_group_gcn_emb_", "storage_location_gcn_emb_",
        "plant_gin_emb_", "product_group_gin_emb_",
        "sub_group_gin_emb_", "storage_location_gin_emb_",
    ]
    keep_prefixes = [et + "_" for et in graph_edge_types]

    def is_graph_col(col: str) -> bool:
        return any(col.startswith(p) for p in all_graph_prefixes)

    for col in list(X.columns):
        if is_graph_col(col) and not any(col.startswith(p) for p in keep_prefixes):
            X = X.drop(columns=[col])

    return X, target
def main():
    global RUN_SUMMARY
    RUN_SUMMARY = []

    # ---------------- Baseline 1 (no graph) ----------------
    for h in (1, 7):
        train_and_evaluate_xgboost(
            horizon=h,
            variant="baseline_1",
            graph_variant=None,
            graph_edge_types=None,   # không có graph feature
            tag="B1_no_graph",
        )

    rolling_origin_evaluation(
        horizon=7,
        origins=[100, 150, 190],
        variant="baseline_1",
        graph_variant=None,
    )

    # ---------------- Baseline 2 (graph stats) ----------------
    # 4 single edges
    graph_settings = {
        "B2_plant": ["plant"],
        "B2_group": ["product_group"],
        "B2_subgroup": ["sub_group"],
        "B2_storage": ["storage_location"],

        # 3 đôi quan trọng
        "B2_plant_group": ["plant", "product_group"],
        "B2_plant_subgroup": ["plant", "sub_group"],
        "B2_plant_storage": ["plant", "storage_location"],
    }
    

    for tag, edges in graph_settings.items():
        for h in (1, 7):
            train_and_evaluate_xgboost(
                horizon=h,
                variant="baseline_2",
                graph_variant="all",       
                graph_edge_types=edges,
                tag=tag,
            )

        # rolling-origin chỉ cần cho H=7 (nếu muốn)
        rolling_origin_evaluation(
            horizon=7,
            origins=[100, 150, 190],
            variant="baseline_2",
            graph_variant="all",
        )
    
    graph_settings_gnn = {
        "B3_gcn_only": [
            "plant_gcn_emb", "product_group_gcn_emb",
            "sub_group_gcn_emb", "storage_location_gcn_emb"
        ],
        "B3_gcn_plus_graph": [
            "plant", "product_group", "sub_group", "storage_location",
            "plant_gcn_emb", "product_group_gcn_emb",
            "sub_group_gcn_emb", "storage_location_gcn_emb"
        ],
        "B3_gin_only": [
            "plant_gin_emb", "product_group_gin_emb",
            "sub_group_gin_emb", "storage_location_gin_emb"
        ],
        "B3_gin_plus_graph": [
            "plant", "product_group", "sub_group", "storage_location",
            "plant_gin_emb", "product_group_gin_emb",
            "sub_group_gin_emb", "storage_location_gin_emb"
        ],
    }

    for tag, edges in graph_settings_gnn.items():
        for h in (1, 7):
            train_and_evaluate_xgboost(
                horizon=h,
                variant="baseline_3",
                graph_variant="all",
                graph_edge_types=edges,
                tag=tag,
            )
    # ---------------- Summary table ----------------
    if RUN_SUMMARY:
        df_sum = pd.DataFrame(RUN_SUMMARY)
        print("\n=== Overall summary (all baselines & variants) ===")
        df_sum = df_sum.sort_values(["horizon", "variant", "tag"])
        print(df_sum[[
            "horizon", "variant", "tag",
            "n_features",
            "MAE_test", "RMSE_test",
        ]])

        out_path = PROC_DIR / "predictions" / "summary_all_baselines.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_sum.to_csv(out_path, index=False)
        print(f"\nSaved summary to {out_path}")

if __name__ == "__main__":
    main()