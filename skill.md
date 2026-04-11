# SupplyGraph Benchmark Skill

## 1. Project Goal

- Build a **benchmark** for demand forecasting on the SupplyGraph dataset: multi-node (SKU/plant), daily time series, with graph structure (plant, product group, etc).
- Ensure **fair comparison** between:
  - Tabular/time-series models (XGBoost, GRU, etc).
  - Graph models (GCN, GIN, etc) using node features, lag features, and edges.
- Provide a **reproducible pipeline**: same input, split, and metric, so all models are compared on the **same target** and **protocol**.

---


## 2. Data & Encoding

### 2.0. Node ID Standard

- The dataset contains exactly **40 original nodes**, defined in `NodesIndex.csv`.
- Use both `node_id` (string, e.g. SKU code) and `node_index` (integer 0..39) as logical IDs.
- **All graphs (edges) and embeddings are normalized to internal indices 0..39**.
- Any mapping, edge, or embedding must use this 0..39 index order, matching the 40-row `df_meta`.

### 2.1. Main File

- `gnn_data_encoded.pt` (in `PROC_DIR/gnn`): shared tensor file for all GNNs:
  - `X`: tensor `[T, N, F]` (T: days, N: nodes, F: features)
  - `Y_h1`, `Y_h7`: target tensors `[T, N]` (horizon 1/7)
  - `days`: `[T]` (day indices)
  - `split`: `[T]` ("train"/"val"/"test" per day)
  - `feature_cols`: feature names
- **All models** (XGBoost, GRU, GCN, GIN, etc) must use the same target and split for fair comparison.


### 2.2. Graph preprocessing / edge_index

- All Edges (*.csv) use `node_index` in range 0..39.
- When building `edge_index_views.pt`, node indices are mapped to positions 0..39 according to the order in the 40-row `df_meta` (from `NodesIndex.csv`).
- Thus, every `edge_index` in the pipeline is always in [0, 39].
### 2.3. preprocess_gnn.py: Node Embedding Generation

- `preprocess_gnn.py` **does not train GNN forecasting models**.
- Its role is to generate node embeddings for all 40 nodes using GCN/GIN encoders, for each of 4 graph views (plant, product_group, sub_group, storage_location).
- The output is `gnn_node_embeddings_all_views.parquet`, containing embeddings for all nodes and all views.
- These embeddings are then **merged as features into baseline models** (e.g., XGBoost) for fair comparison.

---

## 3. Evaluation Protocol (Do Not Change)

### 3.1. Split
- Split by **day** (T), not by node. All nodes at day `t` share the same split.
- Keep this split for all models.

### 3.2. Mask NaN
- `Y` may have NaN (nodes without target).
- Do **not** drop snapshots with partial NaN; mask `~torch.isnan(y)` per snapshot for loss/metric.
- All models must use the same masking.

### 3.3. Metric
- Use **global MAE, global RMSE** (not per-snapshot average):
  - Concatenate all `(y_true, y_pred)` for all days/nodes in the split (excluding NaN).
  - Compute:
    - `MAE = mean(|y_pred - y_true|)`
    - `RMSE = sqrt(mean((y_pred - y_true)^2))`
- Baselines (XGBoost/GRU) and GNNs must use **exactly** this method.

---


## 4. GNN Pipeline

### 4.1. Models (`gnn_model.py`)
- `GCNNodeRegressor`: multi-layer GCNConv, input `[N, F]`, output `[N]`.
- `GINNodeRegressor`: multi-layer GINConv+MLP, input/output as above.
- Both are **node regression** per snapshot (t), same as baselines.

### 4.2. build_dataset (in `train_gnn.py`)
- Load `gnn_data_encoded.pt`.
- Select `Y_h1` or `Y_h7` by horizon.
- **Normalize features using train days only** (mean/std per column, ignoring NaN).
- Load `edge_index_full` from `edge_index_views.pt`.
- Remap/trim `edge_index` to match nodes in `X` (shift to 0-based, mask out-of-range).
- For each day `t`:
  - `x_t = X[t]`, `y_t = Y[t]`, replace NaN/inf in `x_t` with 0.
  - Skip snapshot if `y_t` is all NaN.
  - Create `Data(x, edge_index, y)`, assign `data.day = days[t]`, append to train/val/test by split.
- Result: 3 lists (`data_train`, `data_val`, `data_test`), each is a list of daily snapshots with the same graph.

### 4.3. Train & Eval
- `train_epoch`: loop over loader (batch_size=1), mask NaN in `y`, compute loss on valid nodes.
- `eval_epoch`: collect all `y_true`, `y_pred` across batches, compute global MAE/RMSE.
- Early stopping on `val_rmse`, save best state.
- After training: evaluate on test, save metrics and predictions (CSV: day, node_index, y_true, y_pred).

### 4.4. Plotting
- Log `epoch`, `train_loss`, `val_mae`, `val_rmse`.
- Plot and save learning curves in `PROC_DIR/predictions_gnn/{tag}/`.
- All models must save metrics/curves in the same format for comparison.

---

## 5. Baselines & Fairness

When adding new models (e.g., GraphTransformer, TemporalGNN):
1. Use the same `gnn_data_encoded.pt` (or equivalent) for input, target, and split.
2. If using different normalization (batch/layer norm), still normalize input as in GNN pipeline for fairness.
3. Use **global MAE/RMSE** and same NaN masking.
4. Log test metrics and predictions in the same format/location (CSV: day, node_index, y_true, y_pred).
5. Ensure all models predict the **same target** (e.g., `sold_qty_h1` at day t), no shifting.

---

## 6. Guide for Agents/New Users

When adding/modifying a model:
1. Read `gnn_data_encoded.pt` to understand dimensions and targets.
2. Follow `build_dataset` for dataset definition, normalization, and split.
3. Follow `train_one_model` for training loop, logging, early stopping, and saving outputs.
4. **Do not** change split, metric calculation, or target definition. If protocol changes, create a new version and document it.

---

## 7. TL;DR (for agent)
- Standard dataset: `gnn_data_encoded.pt`.
- Standard graph: `edge_index_views.pt`.
- Split by day (not node).
- Mask NaN in `y` for loss/metric.
- Metric: **global MAE, global RMSE** over all (t, node) in test.
- Save prediction CSV and learning curves for each model/horizon.
- **Reuse this protocol** for new models to ensure fair, reproducible comparison.
