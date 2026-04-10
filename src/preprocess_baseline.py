import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from config.config import (
    NODE_DIR, TEMPORAL_DIR, PROC_DIR, EDGE_DIR,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, LAG_WINDOW
)
pd.set_option("display.max_columns", None) 
pd.set_option("display.width", 0)  
def compute_time_splits(num_days: int, train_ratio: float = TRAIN_RATIO, val_ratio: float = VAL_RATIO, test_ratio: float = TEST_RATIO) -> tuple[int, int, int]:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Train/val/test ratios must sum to 1.0.")

    train_end = int(np.floor(num_days * train_ratio))
    val_end = int(np.floor(num_days * (train_ratio + val_ratio)))
    test_end = num_days

    return train_end, val_end, test_end

def load_node_metadata() -> pd.DataFrame:
    node_index_path = NODE_DIR / "NodesIndex.csv"
    node_group_path = NODE_DIR / "Node Types (Product Group and Subgroup).csv"
    node_plant_storage_path = NODE_DIR / "Nodes Type (Plant & Storage).csv"

    df_index = pd.read_csv(node_index_path)
    df_index = df_index.rename(columns={"Node": "node_id", "NodeIndex": "node_index"})

    df_group = pd.read_csv(node_group_path)
    df_group = df_group.rename(
        columns={"Node": "node_id", "Group": "group", "Sub-Group": "sub_group"}
    )

    df_plant = pd.read_csv(node_plant_storage_path)
    df_plant = df_plant.rename(
        columns={
            "Node": "node_id",
            "Plant": "plant",
            "Storage Location": "storage_location",
        }
    )

    df_meta = df_index.merge(df_group, on="node_id", how="left")
    df_meta = df_meta.merge(df_plant, on="node_id", how="left")

    return df_meta

def load_temporal_wide(filename: str, value_name: str) -> pd.DataFrame:
    path = TEMPORAL_DIR / filename
    df_wide = pd.read_csv(path)

    df_wide["Date"] = pd.to_datetime(df_wide["Date"])

    df_long = df_wide.melt(
        id_vars=["Date"],
        var_name="node_id",
        value_name=value_name,
    )

    df_long = df_long.rename(columns={"Date": "date"})
    df_long = (
        df_long
        .sort_values(["date", "node_id"])
        .drop_duplicates(subset=["date", "node_id"], keep="last")
    )
    return df_long

def load_raw_data() -> pd.DataFrame:
    df_sales = load_temporal_wide("Sales Order.csv", "sales_order")
    df_prod = load_temporal_wide("Production.csv", "production")
    df_delv = load_temporal_wide("Delivery To distributor.csv", "delivery")
    df_issue = load_temporal_wide("Factory Issue.csv", "factory_issue")

    df = df_sales.merge(df_prod, on=["date", "node_id"], how="left")
    df = df.merge(df_delv, on=["date", "node_id"], how="left")
    df = df.merge(df_issue, on=["date", "node_id"], how="left")
    df = df.sort_values(["date", "node_id"]).reset_index(drop=True)

    unique_dates = df["date"].drop_duplicates().sort_values().reset_index(drop=True)
    date_to_day = {d: i + 1 for i, d in unique_dates.items()}
    df["day"] = df["date"].map(date_to_day)

    df_meta = load_node_metadata()
    df = df.merge(df_meta, on="node_id", how="left")

    cols_order = [
        "node_id", "node_index", "date", "day",
        "sales_order", "production", "delivery", "factory_issue",
        "group", "sub_group", "plant", "storage_location",
    ]
    df = df[cols_order]
    df = (
        df
        .sort_values(["date", "node_id"])
        .drop_duplicates(subset=["date", "node_id"], keep="last")
    )
    return df

def add_lag_features(df: pd.DataFrame, lag_cols: list[str], max_lag: int = LAG_WINDOW) -> pd.DataFrame:
    df = df.sort_values(["node_id", "day"]).copy()
    for col in lag_cols:
        for lag in range(max_lag):
            df[f"{col}_lag{lag}"] = (
                df.groupby("node_id")[col].shift(lag)
            )
    return df

def add_rolling_stats(df: pd.DataFrame,
                      cols: list[str],
                      window: int = 7) -> pd.DataFrame:
    df = df.sort_values(["node_id", "day"]).copy()
    for col in cols:
        grp = df.groupby("node_id")[col]
        df[f"{col}_roll{window}_mean"] = grp.transform(
            lambda x: x.rolling(window).mean()
        )
        df[f"{col}_roll{window}_std"] = grp.transform(
            lambda x: x.rolling(window).std()
        )
        df[f"{col}_roll{window}_max"] = grp.transform(
            lambda x: x.rolling(window).max()
        )
        df[f"{col}_roll{window}_min"] = grp.transform(
            lambda x: x.rolling(window).min()
        )
    return df

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day_of_week"] = df["date"].dt.weekday
    df["is_weekend"] = df["day_of_week"] >= 5
    return df

def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["node_id", "day"]).copy()
    df["y_h1"] = df.groupby("node_id")["sales_order"].shift(-1)
    df["y_h7"] = df.groupby("node_id")["sales_order"].shift(-7)
    return df

def assign_splits(df: pd.DataFrame,
                  train_ratio: float = TRAIN_RATIO,
                  val_ratio: float = VAL_RATIO,
                  test_ratio: float = TEST_RATIO) -> pd.DataFrame:

    num_days = df["day"].max()
    train_end, val_end,test_end = compute_time_splits(
        num_days, train_ratio, val_ratio, test_ratio
    )

    df = df.copy()
    conditions = [
        df["day"] <= train_end,
        (df["day"] > train_end) & (df["day"] <= val_end),
        df["day"] > val_end,
    ]
    choices = ["train", "val", "test"]

    df["split"] = np.select(conditions, choices, default="test")

    return df

def filter_valid_samples(df: pd.DataFrame) -> pd.DataFrame:
    lag_cols = [c for c in df.columns if "lag" in c]
    label_cols = ["y_h1", "y_h7"]
    df = df.dropna(subset=lag_cols + label_cols)
    return df

def compute_graph_stats(edge_file: Path,
                        node_index_file: Path,
                        edge_type: str,
                        out_path: Path) -> None:
    edges = pd.read_csv(edge_file)
    if {"node1", "node2"}.issubset(edges.columns):
        u_col, v_col = "node1", "node2"
    else:
        raise ValueError(f"Unknown edge file format: {edge_file}")

    G = nx.Graph()
    for _, row in edges.iterrows():
        G.add_edge(int(row[u_col]), int(row[v_col]))

    deg = dict(G.degree())
    clustering = nx.clustering(G)
    closeness = nx.closeness_centrality(G)
    betweenness = nx.betweenness_centrality(G, normalized=True)

    df_stats = pd.DataFrame({
        "node_index": list(deg.keys()),
        f"{edge_type}_deg": list(deg.values()),
        f"{edge_type}_clustering": [clustering[n] for n in deg.keys()],
        f"{edge_type}_closeness": [closeness[n] for n in deg.keys()],
        f"{edge_type}_betweenness": [betweenness[n] for n in deg.keys()],
    })

    df_idx = pd.read_csv(node_index_file).rename(
        columns={"Node": "node_id", "NodeIndex": "node_index"}
    )
    df_stats = df_stats.merge(df_idx, on="node_index", how="left")

    df_stats.to_csv(out_path, index=False)
    print(f"Saved graph stats for {edge_type} to {out_path}")

def load_graph_stats() -> None:
    node_index_file = NODE_DIR / "NodesIndex.csv"
    edge_defs = [
        (EDGE_DIR / "Edges (Plant).csv", "plant"),
        (EDGE_DIR / "Edges (Product Group).csv", "product_group"),
        (EDGE_DIR / "Edges (Product Sub-Group).csv", "sub_group"),
        (EDGE_DIR / "Edges (Storage Location).csv", "storage_location"),
    ]

    for edge_file, et in edge_defs:
        out_path = PROC_DIR / "graph_stats" / f"graph_stats_{et}.csv"
        compute_graph_stats(
            edge_file=edge_file,
            node_index_file=node_index_file,
            edge_type=et,
            out_path=out_path,
        )

def merge_all_graph_stats(df: pd.DataFrame) -> pd.DataFrame:

    stats_dir = PROC_DIR / "graph_stats"
    edge_types = ["plant", "product_group", "sub_group", "storage_location"]

    for et in edge_types:
        path = stats_dir / f"graph_stats_{et}.csv"
        g = pd.read_csv(path)
        if "node_index" in g.columns:
            g = g.drop(columns=["node_index"])

        df = df.merge(g, on="node_id", how="left")

    return df

def build_xgboost_datasets() -> None:
    df = load_raw_data()
    lag_cols = ["sales_order", "production", "delivery", "factory_issue"]
    df = add_lag_features(df, lag_cols, max_lag=LAG_WINDOW)
    df = add_rolling_stats(df, ["sales_order", "production", "delivery", "factory_issue"], window=7)
    df = add_calendar_features(df)
    df = create_labels(df)
    df = assign_splits(df)
    df = filter_valid_samples(df)

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROC_DIR / "xgboost_base_filtered.parquet", index=False)

    load_graph_stats()
    df = merge_all_graph_stats(df)
    feature_cols = [
        c for c in df.columns
        if any(
            kw in c
            for kw in [
                "lag", "roll",
                "group", "sub_group",
                "plant", "storage_location",
                "day_of_week", "is_weekend", "month",
                "deg", "betweenness", "closeness", "clustering",
            ]
        )
    ]
    base_cols = ["node_id", "node_index", "date", "day", "split"]
    out_dir = PROC_DIR / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_h1 = df[base_cols + feature_cols + ["y_h1"]].rename(columns={"y_h1": "target"})
    df_h7 = df[base_cols + feature_cols + ["y_h7"]].rename(columns={"y_h7": "target"})
    
    df_h1.to_parquet(out_dir / "xgboost_h1.parquet", index=False)
    df_h7.to_parquet(out_dir / "xgboost_h7.parquet", index=False)
    print(f"Saved baseline datasets with horizon 1 to {out_dir} / xgboost_h1.parquet")
    print(f"Saved baseline datasets with horizon 7 to {out_dir} / xgboost_h7.parquet")

    df_h1.to_csv(out_dir / "xgboost_h1_full.csv", index=False)
    df_h7.to_csv(out_dir / "xgboost_h7_full.csv", index=False)

    print(f"Saved H=1 CSV to {out_dir / 'xgboost_h1_full.csv'}")
    print(f"Saved H=7 CSV to {out_dir / 'xgboost_h7_full.csv'}")
    
    print("\n=== Columns in baseline H=1 ===")
    print(list(df_h1.columns))

    print("\n=== Columns in baseline H=7 ===")
    print(list(df_h7.columns))


def main() -> None:
    build_xgboost_datasets()

if __name__ == "__main__":
    main()