"""
Plot family-level MAE by zero-ratio bucket as a single line chart.
Assumes you already ran the zero-bucket diagnostics script and have
`mae_by_zero_bucket_all_horizons_lags_unit_agg_over_seeds.csv` in
`BASE_DIR/diagnostics`.

Families:
- Plain XGBoost
- XGBoost + Graph (baseline_3 or graphfeat, take best of them)
- XGBoost + GNN embedding
- XGBoost + GNN residual
- Direct GNN

We plot mean MAE (averaged over all horizons/lags) vs zero_bucket.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(r"D:\SupplyGraph_experiment - S1\data\processed\predictions")
TEMPORAL_TYPE = "unit"


def load_agg_mae() -> pd.DataFrame:
    diag_dir = BASE_DIR / "diagnostics"
    path = diag_dir / f"mae_by_zero_bucket_all_horizons_lags_{TEMPORAL_TYPE}_agg_over_seeds.csv"
    df = pd.read_csv(path)
    return df


def add_family_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def to_family(model: str) -> str:
        m = model.strip()
        if m == "XGB":
            return "Plain XGBoost"
        if m.startswith("XGB+Graph"):
            return "XGB + Graph"
        if m.startswith("XGB + graphfeat"):
            return "XGB + Graph"  # same family as baseline_3
        if m.startswith("XGB + GNN embed"):
            return "XGB + GNN embedding"
        if m.startswith("Residual GNN"):
            return "XGB + GNN residual"
        # Direct GNN family: all pure GNNs (homo/hetero/proj)
        if m.startswith("GNN "):
            return "Direct GNN"
        if m.startswith("GNN proj"):
            return "Direct GNN"
        return "Other"

    df["family"] = df["model"].apply(to_family)

    keep = {
        "Plain XGBoost",
        "XGB + Graph",
        "XGB + GNN embedding",
        "XGB + GNN residual",
        "Direct GNN",
    }
    df = df[df["family"].isin(keep)].copy()
    return df


def pick_best_variant_per_family(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (horizon, lag_window, zero_bucket, family) keep the
    variant (model) with the lowest mean_MAE.
    """
    df = df.copy()
    df["rank"] = df.groupby(
        ["horizon", "lag_window", "zero_bucket", "family"]
    )["mean_MAE"].rank(method="first")
    df_best = df[df["rank"] == 1].drop(columns=["rank"])
    return df_best


def aggregate_over_horizons_lags(df_best: pd.DataFrame) -> pd.DataFrame:
    """
    Average MAE over horizons & lags to get one MAE per
    (zero_bucket, family).
    """
    df_agg = (
        df_best
        .groupby(["zero_bucket", "family"], as_index=False)
        .agg(mean_MAE=("mean_MAE", "mean"))
    )
    bucket_order = {"low_zero": 0, "mid_zero": 1, "high_zero": 2}
    df_agg["bucket_order"] = df_agg["zero_bucket"].map(bucket_order)
    df_agg = df_agg.sort_values(["bucket_order", "family"]).reset_index(drop=True)
    return df_agg


def plot_family_line(df_agg: pd.DataFrame, out_path: Path):
    plt.style.use("seaborn-v0_8-whitegrid")

    bucket_order = ["low_zero", "mid_zero", "high_zero"]
    x = list(range(len(bucket_order)))
    bucket_label_map = {
        "low_zero": "Low zero-ratio",
        "mid_zero": "Mid zero-ratio",
        "high_zero": "High zero-ratio",
    }

    fig, ax = plt.subplots(figsize=(6, 4))

    families = [
        "Plain XGBoost",
        "XGB + Graph",
        "XGB + GNN embedding",
        "XGB + GNN residual",
        "Direct GNN",
    ]

    for fam in families:
        sub = df_agg[df_agg["family"] == fam]
        if sub.empty:
            continue
        y = []
        for b in bucket_order:
            row = sub[sub["zero_bucket"] == b]
            y.append(row["mean_MAE"].iloc[0] if not row.empty else float("nan"))
        ax.plot(x, y, marker="o", label=fam)

    ax.set_xticks(x)
    ax.set_xticklabels([bucket_label_map[b] for b in bucket_order])
    ax.set_ylabel("Mean MAE (averaged over horizons/lags)")
    ax.set_xlabel("Zero-ratio bucket")
    ax.set_title("Model family performance across zero-ratio buckets")
    ax.legend(frameon=True, fontsize=8)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    df = load_agg_mae()
    df = add_family_column(df)
    df_best = pick_best_variant_per_family(df)
    df_agg = aggregate_over_horizons_lags(df_best)

    out_png = BASE_DIR / "diagnostics" / "family_zero_bucket_lineplot.png"
    plot_family_line(df_agg, out_png)
    print(f"[SAVE] Line plot saved to {out_png}")


if __name__ == "__main__":
    main()