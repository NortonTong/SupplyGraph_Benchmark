import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from config.config import PROC_DIR  # bạn đã dùng trong các script khác

# --------- load summary ----------
base_dir = Path(PROC_DIR) / "predictions" 
summary_path = base_dir / "summary_all_baselines.csv"

df = pd.read_csv(summary_path)

# chỉ lấy GNN, temporal_type=unit, horizon=7
df = df[
    (df["model_family"] == "gnn")
    & (df["temporal_type"] == "unit")
    & (df["horizon"] == 7)
]

METRIC = "RMSE_test"

def pick_best_by_graph_type(df_sub):
    """
    df_sub: filter theo 1 lag_window.
    Return: dict graph_type -> best RMSE entry (projected/homo5/hetero5)
    """
    out = {}

    # projected: có nhiều hàng với different edge_view / clip / softplus
    df_proj = df_sub[df_sub["graph_type"] == "projected"]
    if len(df_proj):
        best_proj = df_proj.loc[df_proj[METRIC].idxmin()]
        out["Projected"] = best_proj[METRIC]
    else:
        out["Projected"] = None

    # homo5: nhiều biến thể clip/softplus, lấy RMSE_test nhỏ nhất
    df_homo = df_sub[df_sub["graph_type"] == "homo5"]
    if len(df_homo):
        best_homo = df_homo.loc[df_homo[METRIC].idxmin()]
        out["Homogeneous"] = best_homo[METRIC]
    else:
        out["Homogeneous"] = None

    # hetero5: tương tự
    df_hetero = df_sub[df_sub["graph_type"] == "hetero5"]
    if len(df_hetero):
        best_hetero = df_hetero.loc[df_hetero[METRIC].idxmin()]
        out["Heterogeneous"] = best_hetero[METRIC]
    else:
        out["Heterogeneous"] = None

    return out

# pick best cho lag=7 và lag=14
df_lag7 = df[df["lag_window"] == 7]
df_lag14 = df[df["lag_window"] == 14]

best_lag7 = pick_best_by_graph_type(df_lag7)
best_lag14 = pick_best_by_graph_type(df_lag14)

print("Best RMSE_test (lag=7):", best_lag7)
print("Best RMSE_test (lag=14):", best_lag14)

# chuẩn bị data cho plot
graph_types = ["Projected", "Homogeneous", "Heterogeneous"]
y_lag7 = [best_lag7[g] for g in graph_types]
y_lag14 = [best_lag14[g] for g in graph_types]

# --------- plot bar chart ---------
plt.rcParams["figure.figsize"] = (6.5, 3.0)   # phù hợp 2-column, 1 hàng 2 subplot
plt.rcParams["axes.titlesize"] = 8
plt.rcParams["axes.labelsize"] = 7
plt.rcParams["xtick.labelsize"] = 7
plt.rcParams["ytick.labelsize"] = 7
plt.rcParams["legend.fontsize"] = 7

fig, axes = plt.subplots(1, 2, sharey=True)

# colors cố định cho 3 loại graph
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

# subplot (a): lag 7
ax0 = axes[0]
bars0 = ax0.bar(graph_types, y_lag7, color=colors)
ax0.set_title("Lag 7")
ax0.set_ylabel("RMSE (test)")

for bar in bars0:
    h = bar.get_height()
    ax0.text(
        bar.get_x() + bar.get_width() / 2,
        h,
        f"{h:.0f}",
        ha="center",
        va="bottom",
        fontsize=6,
    )

# subplot (b): lag 14
ax1 = axes[1]
bars1 = ax1.bar(graph_types, y_lag14, color=colors)
ax1.set_title("Lag 14")

for bar in bars1:
    h = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        h,
        f"{h:.0f}",
        ha="center",
        va="bottom",
        fontsize=6,
    )

fig.suptitle("Graph-type comparison (GNN, unit-scale, H=7)", fontsize=9)
plt.tight_layout()
out_path = base_dir / "graph_type_comparison_gnn_unit_h7_rmse.png"
plt.savefig(out_path, dpi=400)
plt.close()
print("Saved bar chart to", out_path)
