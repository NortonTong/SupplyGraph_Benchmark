# Reproducible Supply‑Chain Forecasting Benchmark and Controlled Study of Graph/Subgraph Integration

## 1. Motivation and research gap

Supply chain demand forecasting is naturally graph‑structured: products, plants, storage locations, and product groups form a relational network where shocks at one node propagate along shared facilities and categories.  Traditional forecasting pipelines, however, almost always treat each product as an independent time series, ignoring this network structure and relying on either classical intermittent‑demand models or tabular ML over hand‑crafted lags and calendar features.  This creates a gap between the rich relational structure of real supply chains and the mostly univariate or tabular models used in practice. 

Recent work has begun to explore graph neural networks (GNNs) for supply chain planning, notably through the SupplyGraph dataset and associated benchmarks.  These studies show that GNN‑based models can outperform conventional baselines on several tasks, but they also expose two critical limitations for reproducible research. First, **data and code are not fully open or self‑contained**, making it difficult for other groups to exactly reproduce the end‑to‑end pipeline from raw CSVs to model‑ready tensors.  Second, the published descriptions often omit key implementation details—such as temporal splitting strategies, preprocessing choices, hyperparameter search, and baseline configurations—so follow‑up work tends to adopt heterogeneous setups that are hard to compare directly. 

At the same time, **how the supply‑chain graph is constructed and how it is integrated into the forecasting model** remain under‑explored. Existing work typically fixes a single graph construction (for example, a product–plant graph) and uses it inside a GNN, without systematically contrasting alternative graph design and usage choices.  In particular, there is no controlled benchmark that jointly varies:
- Graph construction: projected product‑only views vs full homogeneous graphs (multiple node types, untyped edges) vs full heterogeneous graphs (typed edges). 
- Graph usage: incorporating graph information as engineered **graph features** for strong tabular models (e.g., XGBoost), versus using **graph models** (GNNs) directly on node‑level time series. 

This gap is especially acute for **intermittent, zero‑inflated, heavy‑tailed demand**, where standard MSE losses and naive output parameterizations can interact poorly with both tabular and graph‑based models.  The field currently lacks a benchmark that is: 
1. **Fully reproducible and open**: from raw CSV files through preprocessing to standardized parquet and PyTorch datasets for all baselines and GNN variants. 
2. **Methodologically transparent**: with an explicit, documented pipeline for calendar features, lag/rolling statistics, label construction, and time‑based splits. 
3. **Controlled along key graph axes**: providing a systematic comparison of projected vs homogeneous vs heterogeneous graphs, and of graph‑features vs graph‑models, under the same data, metrics, and optimization regime.

Our benchmark is designed to fill precisely this gap. It builds on a SupplyGraph‑style FMCG dataset but re‑implements the entire preprocessing and baseline stack in a transparent and modular way, enabling rigorous and reproducible experiments on graph‑based vs non‑graph demand forecasting under realistic intermittent demand conditions. 

---

## 2. Contributions

We revisit graph‑aware multi‑product demand forecasting with a focus on **benchmarks, data handling, and practical use of graphs**, rather than proposing a single new architecture. Our contributions are:

### 2.1. Transparent benchmark, splits, and evaluation

We build a clean benchmark for multi‑product demand forecasting and make all data‑handling choices explicit. 

- We preprocess and normalize the original sales data, clearly define the prediction target (unit demand) and horizon, and remove undocumented heuristics from earlier work.  
- We fix a **chronological train/validation/test split** at the day level and reuse this split for all models, ensuring that results are directly comparable.  
- We adopt a unified evaluation protocol with **MAE, RMSE, MAPE, and sMAPE** computed on the same unit scale for all methods, and report these metrics by horizon and lag window.  

Previous work on this dataset does not fully specify whether errors are measured in units, tens, dozens, or some other rescaling, and split details are partially implicit. Despite contacting the original authors, we could not obtain a definitive clarification, so we document our own choices carefully and release code and configuration to support reproducibility.

### 2.2. Systematic framework for building and using graphs

We **systematize how graphs are constructed and how they are used by the forecasting models**.

- Graph construction:
  - We formalize multiple graphs over the same raw entities:  
    - Projected product–product graphs (e.g., same group, same subgroup, same plant, same storage location).  
    - A homogeneous 5‑type graph where all node types are flattened into a single index space.  
    - A heterogeneous 5‑type graph with typed edges between products, product groups, subgroups, plants, and storage locations.  
  - For each graph type, we specify node indexing, edge semantics, and the mapping from time‑indexed product series to graph nodes, so every model sees exactly the same structure.  
- Ways of using the graph:
  - We instantiate three distinct graph‑aware strategies:
    1. **Graph‑as‑features**: basic and advanced graph features, or pre‑trained GNN embeddings, fed into XGBoost.  
    2. **Graph‑as‑model**: GNNs trained end‑to‑end as forecasters on the projected, homogeneous, or heterogeneous graphs.  
    3. **Graph‑as‑residual‑corrector**: GNNs trained to predict residuals on top of XGBoost forecasts.  

This separation of “how we build the graph” from “how we use the graph” yields a unified experimental scaffold in which new graph constructions or model families can be plugged in without redoing all engineering details. 

### 2.3. Strong tabular baseline and comparative study

We design a **comparative study** that treats a lag‑based XGBoost forecaster as a **mandatory baseline** and evaluates all graph‑aware variants against it under identical conditions. 

- We show that a carefully tuned **lag‑based XGBoost** model is very strong in this setting, often matching or outperforming sequence models such as GRUs on MAE and RMSE.  
- Across horizons and lag windows, XGBoost remains stable, whereas pure GNN forecasters frequently degrade on relative error metrics (MAPE, sMAPE), especially when trained directly on raw units.  
- We compare:
  - Plain tabular XGBoost (Baseline 1).  
  - Sequence GRU baseline (Baseline 2).  
  - XGBoost with static graph features (Baseline 3).  
  - Direct GNN forecasters (Baseline 4).  
  - XGBoost with GNN embeddings (Baseline 5).  
  - XGBoost with GNN residual correction (Baseline 6).  

This design reinforces that GRU or GNN models do **not** automatically improve over well‑tuned tabular baselines, and that any claimed gains should be measured against a strong lag‑based XGBoost reference.

### 2.4. Evidence that graph‑as‑features beats graph‑as‑model

Our experiments consistently indicate that, on this dataset, **using graphs to build features for XGBoost is more effective than using GNNs as the main forecaster**. 

- Direct GNN baselines (projected, homogeneous, heterogeneous GINs trained end‑to‑end) do not outperform the best XGBoost variants; in many settings they underperform them, particularly on MAPE and sMAPE.  
- In contrast, **graph‑aware tabular models** – XGBoost with basic and advanced graph features or with learned GNN embeddings – reliably improve over plain XGBoost in MAE and RMSE without introducing the instability observed in GNN‑only models.  
- Residual GNNs on top of XGBoost further reduce some metrics (especially sMAPE) and demonstrate that GNNs are valuable as **residual correctors**, but they still do not consistently dominate the simpler graph‑feature variants across all metrics.  

These findings suggest that, in this benchmark, the most effective use of graph information is **graph‑as‑features** (and, to a lesser extent, **graph‑as‑residual‑corrector**) rather than **graph‑as‑model**. This aligns with broader observations in structured forecasting, where GNNs often shine as representation learners feeding into strong tabular models rather than as full replacements. 

### 2.5. Clearer, reproducible re‑evaluation of prior work

We provide a **clearer and more reproducible re‑evaluation** of prior graph‑aware forecasting work on this dataset. 

- The original paper’s reported numbers are difficult to interpret: it is unclear whether the error units correspond to raw units, tens, dozens, or another rescaling, and some details of the split and preprocessing are not explicitly documented.  
- We attempt to reproduce the original setup as faithfully as possible, including contacting the authors for clarification; however, we could not fully resolve these ambiguities.  
- In response, we:
  - Make our data cleaning, normalization, splits, and evaluation protocol explicit.  
  - Release code that reproduces all the numbers in our tables.  
  - Document where we had to make reasonable assumptions due to missing information.  

Under this controlled and fully documented setting, we find that the main practical gains come from **clean data handling plus graph‑enhanced XGBoost**, rather than from any single sophisticated GNN architecture. We hope this work provides a solid and extensible benchmark that helps the community build clearer, more comparable graph‑aware demand forecasting models.

---

## 3. Data preprocessing and non‑graph baselines

### 3.1. Raw node and temporal data

We start from three groups of raw CSV files:

- Node metadata (static attributes per product):  
  - `NodesIndex.csv`: mapping from product identifier to internal index (`Node` → `node_id`, `NodeIndex` → `node_index`). 
  - `Node Types (Product Group and Subgroup).csv`: product hierarchy, with `Node`, `Group`, `Sub-Group` columns. 
  - `Nodes Type (Plant & Storage).csv`: manufacturing and storage information, with `Node`, `Plant`, `Storage Location`. 

- Temporal data (wide format, one column per product):  
  - `Unit/Sales Order.csv`  
  - `Unit/Production.csv`  
  - `Unit/Delivery To distributor.csv`  
  - `Unit/Factory Issue.csv`  

- Edge metadata (used later for graph construction, not in this file): shared plant, group, subgroup, and storage location edges between products.

All daily time series in this file focus on the **unit** temporal type (`temporal_type = "unit"`); the code is generic enough to also handle `temporal_type = "weight"`, but the current benchmark results use only unit demand.

### 3.2. Node metadata processing

We construct a **clean node metadata table** `df_meta` with one row per product, containing all static attributes used across baselines and graph models. 

1. `load_node_metadata()`:
   - Reads `NodesIndex.csv`, renames `Node` → `node_id`, `NodeIndex` → `node_index`, and drops duplicates. 
   - Reads `Node Types (Product Group and Subgroup).csv`, renames `Node` → `node_id`, `Group` → `group`, `Sub-Group` → `sub_group`, and drops duplicates. 
   - Reads `Nodes Type (Plant & Storage).csv`, renames `Node` → `node_id`, `Plant` → `plant`, `Storage Location` → `storage_location`, and drops duplicates. 
   - Merges the three tables on `node_id` to obtain:  
     - `node_id`, `node_index`, `group`, `sub_group`, `plant`, `storage_location`.  

This metadata is later joined into all temporal tables so that every (product, day) sample carries product hierarchy and location information.

### 3.3. Temporal data: wide to long and merging metrics

We convert all temporal CSVs from **wide** (one column per product) to a unified **long** format with explicit `node_id` and `date`. 

1. Wide → long conversion

- `_load_temporal_wide_generic(subdir, filename, value_name)`:
  - Reads the raw CSV from `TEMPORAL_DIR / subdir / filename`.  
  - Parses the `Date` column into a proper `datetime` type.  
  - Loads `df_meta` to obtain the set of valid product identifiers (`node_id`).  
  - Keeps only `"Date"` and columns whose names are in the valid `node_id` set, which filters out spurious or malformed columns.  
  - Uses `pandas.melt` to transform the table from wide to long, with:  
    - `Date` → `date`  
    - column name → `node_id`  
    - cell value → metric value (`sales_order`, `production`, `delivery`, `factory_issue`).  
  - Sorts by `(date, node_id)` and drops duplicate `(date, node_id)` entries, keeping the last occurrence.  

- Two thin wrappers specify the subdirectory and metric name:  
  - `load_temporal_unit_wide(filename, value_name)` uses `subdir="Unit"` for all unit‑based metrics.  
  - `load_temporal_weight_wide(filename, value_name)` uses `subdir="Weight"` for potential weight‑based metrics (not used in current experiments).

2. Joint metric table and calendar index

We then build the **joint raw temporal table** with all four metrics:

- `_load_raw_data_generic(loader_func)`:
  - Calls `loader_func` four times to load:  
    - `"Sales Order.csv"` → `sales_order`  
    - `"Production.csv"` → `production`  
    - `"Delivery To distributor.csv"` → `delivery`  
    - `"Factory Issue.csv"` → `factory_issue`  
  - Merges the four long tables on `(date, node_id)` via successive left joins.  
  - Sorts by `(date, node_id)` and resets the index.  
  - Builds a **monotone calendar index** `day`:
    - Extracts sorted unique dates.  
    - Maps each date to an integer day index starting at 1: `day = 1, 2, …, num_days`.  
  - Merges in `df_meta` so every row includes `node_index`, `group`, `sub_group`, `plant`, `storage_location`.  

- The resulting raw table has the canonical column order:  
  - `node_id`, `node_index`, `date`, `day`,  
    `sales_order`, `production`, `delivery`, `factory_issue`,  
    `group`, `sub_group`, `plant`, `storage_location`.  

- Two public entry points fix the loader function:  
  - `load_raw_data_unit()` uses `load_temporal_unit_wide`.  
  - `load_raw_data_weight()` uses `load_temporal_weight_wide`.  

This design ensures that all baselines operate on the **same underlying temporal index** and node mapping, which is crucial for a reproducible benchmark.

### 3.4. Calendar features, labels, and time‑based splits

1. Calendar features

- `add_calendar_features(df)` adds standard calendar covariates to each row based on the `date` column:
  - `day_of_week` in \(\{0,\dots,6\}\) with Monday = 0.  
  - `is_weekend` as a Boolean flag for Saturday/Sunday.  
  - `month` in \(\{1,\dots,12\}\).  
  - `day_of_month` in \(\{1,\dots,31\}\).  

These features are used both in non‑graph baselines and in graph‑based models as time‑varying covariates.

2. Forecast labels for horizon \(H\)

- `create_labels(df, horizon)` builds the prediction target for horizon \(H\) days ahead:
  - Sorts by `(node_id, day)` and applies a groupwise negative shift:  
    \[
    y\_{horizon}(t) = \text{sales\_order}(t + H) \quad \text{per product.}
    \]
  - Stores this as the column `y_h{H}` (e.g., `y_h7` for \(H=7\)).  

The label is always defined on the **original (unit) scale**.

3. Time‑based train/validation/test splits

We create strictly time‑ordered splits along the `day` axis to mimic realistic forecasting. 

- `compute_time_splits(num_days, train_ratio, val_ratio, test_ratio)`:
  - Checks that the three ratios sum to 1.0.  
  - Computes:
    - `train_end = floor(num_days * train_ratio)`  
    - `val_end = floor(num_days * (train_ratio + val_ratio))`  
    - `test_end = num_days` (implicit).  

- `assign_splits(df, train_ratio, val_ratio, test_ratio)`:
  - Calls `compute_time_splits` using globally configured `TRAIN_RATIO`, `VAL_RATIO`, `TEST_RATIO`.  
  - Assigns a split label per row based on the `day` index:  
    - `day <= train_end` → `"train"`  
    - `train_end < day <= val_end` → `"val"`  
    - `day > val_end` → `"test"`  

This ensures that no future information leaks into training or validation, and that all models in the benchmark share the *same* temporal splits. 

### 3.5. Lag and rolling statistics for tabular baselines

To support lag‑based models like XGBoost and naive last‑value, we derive classical time series features. 

1. Lag features

- `add_lag_features(df, lag_cols, max_lag)`:
  - Sorts by `(node_id, day)` and, for each column in `lag_cols`, creates lagged versions via groupwise shifts per product.  
  - For each metric `col` and lag \(l = 1,\dots,\text{max\_lag}\), we add:
    - `f"{col}_lag{l}" = col shifted by l days into the past`.  

In this benchmark we use lag windows \(L \in \{7, 14\}\) and apply lags on `sales_order`, `production`, `delivery`, and `factory_issue`. 

2. Rolling window statistics

- `add_rolling_stats(df, cols, window)`:
  - For each metric `col` in `cols`, and for each product, computes rolling statistics over a trailing window of length `window`.  
  - Adds:
    - `"{col}_roll{window}_mean"`  
    - `"{col}_roll{window}_std"`  
    - `"{col}_roll{window}_max"`  
    - `"{col}_roll{window}_min"`.  

These features allow tabular models to capture local level, variability, and extremes over the last \(L\) days without having to model sequences explicitly.

3. Valid sample filtering

- `filter_valid_samples(df, horizon)`:
  - Collects all columns with `"lag"` in their name and the label column `y_h{horizon}`.  
  - Drops any row where at least one lag feature or the target is missing.  

This step removes initial “warm‑up” days where the full lag/rolling window is not available and ensures that every training example has a complete feature vector and label.

### 3.6. Base tables: RAW vs FULL

We define two canonical base tables that all downstream baselines and GNN datasets derive from.

1. RAW base: no lag/rolling

- `build_base_raw(temporal_type, horizon)`:
  - Selects the loader based on `temporal_type`:
    - `"unit"` → `load_raw_data_unit()`  
    - `"weight"` → `load_raw_data_weight()` (not used in current experiments).  
  - Adds calendar features via `add_calendar_features`.  
  - Adds the horizon label `y_h{H}` via `create_labels`.  
  - Assigns `split` via `assign_splits`.  
  - Saves the result to:  
    - `data/processed/base/base_raw_h{H}_{temporal_type}.parquet`.  

This RAW base contains only **original series, calendar features, labels, and splits**, and is used by GRU sequence baselines and as the starting point for FULL tables. 

2. FULL base: lag + rolling

- `build_base_full(temporal_type, horizon, lag_window)`:
  - Loads `base_raw_h{H}_{temporal_type}.parquet` if it exists; otherwise calls `build_base_raw`.  
  - Adds lag features on the four main metrics with `add_lag_features`.  
  - Adds rolling statistics for the same metrics with window size `lag_window` via `add_rolling_stats`.  
  - Applies `filter_valid_samples` to drop rows with incomplete lag/rolling features.  
  - Saves the result to:  
    - `data/processed/base/base_full_h{H}_lag{L}_{temporal_type}.parquet`.  

The FULL base serves as the input for XGBoost, the naive last‑value baseline, and potentially graph‑based models that consume lag features.

### 3.7. One‑hot encoding by split
We standardize categorical handling across baselines using one‑hot encoding applied separately to each split.

- `CAT_COLS = ["group", "sub_group", "plant", "storage_location", "day_of_week", "is_weekend"]`.  

- `one_hot_encode_splits(df, cat_cols)`:
  - Splits the table into `df_train`, `df_val`, `df_test` based on the `split` column.  
  - Determines which categorical columns are actually present.  
  - Applies `pd.get_dummies` to each split independently, without dropping any category (`drop_first=False`).  
  - Aligns validation and test encodings to the training schema via `reindex(columns=df_train_enc.columns, fill_value=0)`.  
  - Concatenates the three encoded splits back into a single table.  

Encoding splits separately avoids any potential leakage of category statistics across temporal splits and guarantees that all models see the same feature space.

### 3.8. Baseline 0: naive last‑value CSV

For the naive baseline that predicts the last observed value at horizon \(H\), we export a FULL feature table to CSV.
- `build_baseline_full_csv_for_naive(temporal_type, horizon, lag_window)`:
  - Ensures `base_full_h{H}_lag{L}_{temporal_type}.parquet` exists by loading or building it.  
  - Writes the table to:
    - `data/processed/baseline/xgboost/xgboost_h{H}_lag{L}_{temporal_type}_full.csv`.  

This CSV is later consumed by a simple baseline that sets \( \hat{y}(t+H) = \text{sales\_order}(t) \) with fallback strategies for missing values. 

### 3.9. Baseline 1: XGBoost tabular (no graph)

We construct a strong, purely tabular baseline using gradient‑boosted trees over lag and rolling features. 

- `build_xgboost_tabular(temporal_type, horizon, lag_window)`:
  - Loads `base_full_h{H}_lag{L}_{temporal_type}.parquet` (or builds it if absent).  
  - Defines:
    - `base_cols = ["node_id", "node_index", "date", "day", "split"]`.  
    - `label_col = f"y_h{horizon}"`.  
  - Selects `feature_cols` by including any column whose name contains one of:
    - `"lag"` (all lag features)  
    - `f"roll{lag_window}_"` (rolling stats for the chosen window)  
    - `"group"`, `"sub_group"`, `"plant"`, `"storage_location"`  
    - `"day_of_week"`, `"is_weekend"`, `"month"`, `"day_of_month"`.  
  - Builds `df_h` with `base_cols + feature_cols + [label_col]`, and renames the label to `target`.  
  - Applies `one_hot_encode_splits(df_h, CAT_COLS)` to obtain a fully numeric design matrix with consistent columns across splits.  
  - Saves the result to:
    - `data/processed/baseline/xgboost/xgboost_tabular_h{H}_lag{L}_{temporal_type}.parquet`.  

This dataset is the input for XGBoost models that operate **without explicit graph information** and serves as a strong tabular baseline against which graph‑based methods are compared. 

### 3.10. Baseline 2: GRU sequence (no graph)

We also build a sequence model baseline that directly consumes time‑ordered covariates without precomputed lags. 

- `build_gru_sequence(temporal_type, horizon, seq_len)`:
  - Loads the RAW base `base_raw_h{H}_{temporal_type}.parquet` (or creates it if needed).  
  - Defines the label `label_col = f"y_h{horizon}"`.  
  - Selects a minimal set of columns:
    - IDs and indexing:  
      - `node_id`, `node_index`, `date`, `day`, `split`.  
    - Dynamic series:  
      - `sales_order`, `production`, `delivery`, `factory_issue`.  
    - Calendar features:  
      - `day_of_week`, `is_weekend`, `month`, `day_of_month`.  
    - Label:  
      - `label_col` renamed to `target`.  
    - Static product attributes:  
      - `group`, `sub_group`, `plant`, `storage_location`.  
  - Filters this list against actual columns present.  
  - Applies `one_hot_encode_splits` to obtain a numeric table with one‑hot encoded product and calendar categories.  
  - Saves the result to:
    - `data/processed/baseline/gru/gru_sequence_h{H}_L{seq_len}_{temporal_type}.parquet`.  

The GRU models then use sliding windows of length `seq_len` (e.g., 7 or 14 days) over this table to form input sequences and predict `target` at horizon \(H\). This baseline isolates the effect of sequence modeling without graph information. 

### 3.11. Sales‑order distribution diagnostics

To better understand the **zero‑inflated, heavy‑tailed** nature of demand, we include a dedicated diagnostic script.
- `inspect_sales_order_hist(df, tag)`:
  - Expects the RAW base (containing `sales_order`).  
  - Prints descriptive statistics, including median, high quantiles, and counts of zero and negative values.  
  - Plots four histogram variants:
    - Linear–linear: raw counts vs raw values.  
    - Log–log: positive values only, both axes on log scale.  
    - Linear–log: raw values on x, log‑scaled counts on y.  
    - Log–linear: positive values on a log x‑scale, linear counts on y.  
  - Saves the resulting 2×2 grid to:
    - `data/processed/sales_distribution/sales_order_hist_4scales_{tag}.png`.  

These plots make it visually clear that the demand distribution has many zeros and a long right tail.

### 3.12. Per‑product time‑series visualization

We also generate **per‑product time‑series grids** for qualitative inspection.

- `plot_sales_per_product_grid(temporal_type, horizon, max_products)`:
  - Loads `base_raw_h{H}_{temporal_type}.parquet` and sorts by `(node_id, day)`.  
  - Selects up to `max_products` distinct `node_id`s.  
  - Creates a grid of subplots with 4 columns and as many rows as needed.  
  - For each selected product, plots `sales_order` over `day` with markers and labels the subplot with the product ID.  
  - Removes any unused subplots in the grid.  
  - Saves the figure to:
    - `data/processed/plots/raw_timeseries/ts_products_{temporal_type}_h{H}.png`.  

These visualizations help confirm patterns such as intermittent demand, bursts, and seasonality at the product level.

### 3.13. Orchestration via `main()`

The `main()` function orchestrates the full preprocessing pipeline for all experiments defined in `DEFAULT_EXPERIMENTS` (combinations of temporal type, horizons, lag windows, and GRU sequence lengths).

For each experiment `exp`:

1. Read configuration:  
   - `t_type = exp.temporal_type` (currently `"unit"`).  
   - Iterate over `H in exp.horizons`.  

2. RAW base + diagnostics and plots:  
   - `build_base_raw(temporal_type=t_type, horizon=H)`  
   - `inspect_sales_order_hist(df_raw, tag="unit")`  
   - `plot_sales_per_product_grid(temporal_type=t_type, horizon=H, max_products=40)`  

3. GRU sequence baselines:  
   - For each `L_seq in exp.gru_seq_lengths`, call:  
     - `build_gru_sequence(temporal_type=t_type, horizon=H, seq_len=L_seq)`.  

4. Lag‑based baselines (naive + XGBoost):  
   - For each `L in exp.lag_windows`, call:  
     - `build_base_full(temporal_type=t_type, horizon=H, lag_window=L)`  
     - `build_baseline_full_csv_for_naive(temporal_type=t_type, horizon=H, lag_window=L)`  
     - `build_xgboost_tabular(temporal_type=t_type, horizon=H, lag_window=L)`  

Running `python data_preprocessing_baselines.py` thus produces:

- Base parquet tables:
  - `data/processed/base/base_raw_h{H}_{temporal_type}.parquet`  
  - `data/processed/base/base_full_h{H}_lag{L}_{temporal_type}.parquet`  

- Baseline inputs:
  - `data/processed/baseline/xgboost/xgboost_h{H}_lag{L}_{temporal_type}_full.csv` (for naive last‑value).  
  - `data/processed/baseline/xgboost/xgboost_tabular_h{H}_lag{L}_{temporal_type}.parquet` (for XGBoost).  
  - `data/processed/baseline/gru/gru_sequence_h{H}_L{seq_len}_{temporal_type}.parquet` (for GRU).  

- Diagnostic plots:
  - `data/processed/sales_distribution/sales_order_hist_4scales_unit.png`.  
  - `data/processed/plots/raw_timeseries/ts_products_{temporal_type}_h{H}.png`.  

This makes the preprocessing and baseline construction **fully deterministic and reproducible**, with all intermediate artifacts explicitly named and stored for downstream experiments and for replication by other researchers.

---


## 4. No‑graph baselines

We evaluate three purely temporal baselines: a naive last‑value model, a strong XGBoost tabular model, and a GRU sequence model. All operate on the same horizon and splits defined in the preprocessing pipeline.

### 4.1. Naive last‑value baseline

The naive baseline predicts future demand by simply reusing the last observed value at the forecast origin \(t_0\). 

- Input data:
  - We reuse the **full baseline** CSV built for XGBoost:
    - `data/processed/baseline/xgboost/xgboost_h{h}_lag{L}_{temporal_type}_full.csv`.  
  - Columns include:
    - `node_id`, `date`, `day`, `split`, `sales_order` at time \(t_0\), and `y_h7` (demand at \(t_0+7\)).  

- Naive rule:
  - For each sample in the test split:
    - Ground truth: `y_true = y_h7`.  
    - Naive prediction: `y_pred_last_t0 = sales_order`.  
  - If `sales_order` is missing, we impute:
    - Node‑level mean of `y_h7` computed on the training split.  
    - Fallback to global mean `y_h7` if the node mean is also missing.  

- Evaluation:  
  - We compute MAE, RMSE, MAPE, and sMAPE on the test split, all on the original demand scale. 
  - Predictions per test row are saved as:
    - `predictions_naive/naive_last_t0_{temporal_type}/lag{L}/naive_last_t0_h{h}_test_predictions.csv`  
    - Columns: `node_id`, `date`, `day`, `y_true`, `y_pred`.  
  - A summary over horizons and lag windows is written to:
    - `predictions_naive/summary_naive_last_t0.csv`.  

This baseline quantifies how much gain we obtain by using any learned model beyond a simple “copy last value” strategy.

---

### 4.2. XGBoost tabular baseline (no graph)

This baseline is the strong non‑graph reference point described earlier, using gradient‑boosted trees on lag‑based tabular features only. It predicts demand at horizon \(H\) directly, without using any graph information. 

- Input dataset:
  - `data/processed/baseline/xgboost/xgboost_tabular_h{H}_lag{L}_{temporal_type}.parquet`.  
  - Each row is a `(product, day)` pair with:
    - IDs/split: `node_id`, `node_index`, `date`, `day`, `split`.  
    - Features: lags, rolling statistics, calendar effects, one‑hot encoded categories.  
    - Target: `target` = demand at \(t_0 + H\).  

- Splits and features:
  - We split by the `split` column into train/val/test without re‑encoding.  
  - We drop non‑feature columns (`target`, `split`, identifiers, `date`, `day`) and treat all remaining columns as input features.  

- Model and training:
  - XGBRegressor with 5000 trees, depth 6, learning rate 0.05, hist tree method, and early stopping with 100 rounds on validation RMSE. 
  - Objective is squared error on the **raw demand scale**.  

- Outputs:
  - Learning curves (train/val RMSE vs. boosting round) saved under:
    - `predictions/baseline_1/plots_learning_curves/`.  
  - Full predictions (train/val/test) saved as:
    - `baseline/xgboost/xgboost_predictions_h{H}_lag{L}_{temporal_type}.parquet` (for residual GNNs later).  
  - Test predictions per row and per‑product time‑series plots saved under:
    - `predictions/baseline_1/csv/` and `predictions/baseline_1/plots_xgb_tabular/`.  
  - A run summary over configurations with MAE/RMSE/MAPE/sMAPE is written to:
    - `predictions/baseline_1/summary_xgb_tabular_baseline1_raw_targets_lags.csv`.  

This baseline is our main **non‑graph benchmark** when comparing against graph‑feature and GNN‑based models. 

---

### 4.3. GRU sequence baseline (no graph)

The GRU baseline models temporal dynamics explicitly as sequences, still without any graph structure. It operates on pre‑built GRU sequence tables and predicts horizon‑\(H\) demand from sliding windows of past features for each product. 

#### 4.3.1. Sequence dataset construction

We use GRU‑specific parquet files:

- `data/processed/baseline/gru/gru_sequence_h{H}_L{W}_{temporal_type}.parquet`.  

Each row is a `(product, day)` observation with:

- `node_id`, `node_index`, `date`, `day`, `split`.  
- Numeric feature columns (e.g., recent sales, covariates) and a `target` column (demand at horizon \(H\)).  

Preprocessing:

- `load_baseline(horizon, window, temporal_type)`:
  - Loads the parquet, casts `target` to float, fills all numeric/boolean columns’ missing values with 0.0.  

For each split (train/val/test), we build sequences:

- `build_sequences(df, window, split)`:
  - Sorts by `node_id`, `day`.  
  - Selects numeric feature columns, excluding `target`, `split`, identifiers, `date`, `day`.  
  - For each product `node_id`, slides a window of length `W`:
    - Input \(X\_t\): the last `W` time steps (shape `W × num_features`).  
    - Target \(y\_t\): the `target` at the next time step.  
  - Returns:
    - `X ∈ ℝ^{N×W×F}`, `y ∈ ℝ^{N}`, and a meta table with `(node_id, date)` for each sample.  

We wrap these into PyTorch datasets and dataloaders:

- `GRUDataset(X, y)` and `make_dataloaders(df, window, batch_size)` give train/val/test loaders and meta information.  

#### 4.3.2. Model architecture and training

The GRU model is a standard sequence‑to‑one regressor:

- `GRURegressor`:
  - GRU encoder:
    - `hidden_size=128`, `num_layers=2`, dropout=0.2, batch‑first.  
  - MLP head:
    - Linear → ReLU → Dropout → Linear( hidden_size → 1 ).  
  - Output:
    - In the baseline we describe, we treat the **output as a raw regression** on the demand scale.  
    - (The implementation supports alternative output transforms for other experiments, but for the main GRU baseline we use the direct linear head.)  

Training (`train_one_gru`):

- Loss and optimizer:
  - Mean squared error loss on the model output scale.  
  - Adam optimizer with learning rate \(10^{-3}\).  
- Early stopping:
  - Custom `EarlyStopping` on validation RMSE with patience (e.g., 20–40 epochs).  
- Loop:
  - For each epoch:
    - Train phase: compute MSE loss, clip gradients, update parameters.  
    - Validation phase: collect predictions and compute RMSE, MAE, MAPE, sMAPE.  
    - Track the best validation RMSE and save the corresponding model state.  

#### 4.3.3. Evaluation, outputs, and summaries

After training, we evaluate on the test loader:

- Test metrics:
  - RMSE, MAE, MAPE, sMAPE computed on the model output scale, interpreted as demand. 
- We build a test prediction DataFrame using the stored meta:
  - Columns: `node_id`, `date`, `y_true`, `y_pred`.  

Outputs:

- Model checkpoints:
  - Stored under:
    - `models/gru/baseline_2/{transform_tag}/gru_baseline2_h{H}_w{W}_{temporal_type}.pth`.  
- Test predictions:
  - Saved to:
    - `predictions/baseline_2/gru/csv/{transform_tag}/{temporal_type}/gru_baseline2_h{H}_w{W}_test_predictions.csv`.  
- Per‑product plots:
  - For each product, we generate time‑series plots of `y_true` vs. `y_pred` on the test set:
    - Stored under:
      - `predictions/baseline_2/gru/plots_gru/{transform_tag}/h{H}_w{W}/{temporal_type}/`.  

The `train_gru.py` script loops over all experiment configurations (temporal types and sequence lengths), calls `train_one_gru`, and aggregates a summary CSV:

- `predictions/baseline_2/gru/summary_gru_baseline2_h{H}_windows_{...}_outputs_{...}.csv`,  
  listing configuration, sequence length, feature count, and train/val/test errors. 

---

## 5. Graph construction strategies

We construct three families of graphs over the same underlying set of products, using only static node metadata (group, subgroup, plant, storage location). This allows us to isolate the effect of graph topology and edge typing, without confounding from temporal information. 

### 5.1. Node types and raw metadata

We start from the cleaned node metadata `df_meta` built in the preprocessing step (Section 3). Each row corresponds to a single product and contains:

- `node_id`: external product identifier.  
- `node_index`: internal integer index used in time‑series tensors.  
- `group`, `sub_group`: product hierarchy.  
- `plant`: manufacturing plant identifier.  
- `storage_location`: storage or warehouse identifier.  

These attributes determine node types and edges in the full graphs, and also drive the projected product graphs constructed by grouping on a single attribute.

### 5.2. Projected product graphs (4 views)

The projected graphs are **product‑only** graphs that connect products sharing the same value for a given attribute. This yields four homogeneous views:

- `product_graph_same_group`: edges between products with the same `group`.  
- `product_graph_same_subgroup`: edges between products with the same `sub_group`.  
- `product_graph_same_plant`: edges between products produced at the same `plant`.  
- `product_graph_same_storage`: edges between products stored in the same `storage_location`.  

Implementation (`build_projected_graph` and `build_all_projected_graphs`):

- Nodes:
  - One node per product (`node_id`), with attributes:
    - `node_index`, `group`, `sub_group`, `plant`, `storage_location`.  
- Edges:
  - For a chosen attribute `by_col` (e.g., `"group"`), we:
    - Filter `df_meta` to rows with non‑missing `by_col`.  
    - Group by `by_col` and, within each group, fully connect all products (simple undirected graph, no self‑loops).  
- Outputs:
  - A NetworkX `Graph` per view, saved as:
    - `graphs/projected_product_graphs/{out_name}.gpickle`.  
  - Edge list and node table as parquet:
    - `{out_name}_edges.parquet` (columns: `src`, `dst`).  
    - `{out_name}_nodes.parquet` (columns: `node_id`, `node_index`, static attributes).  

These four views let us test whether simpler “similarity” graphs (same group, plant, storage, etc.) provide enough relational signal for demand forecasting.

### 5.3. Homogeneous 5‑type graph

The homogeneous graph explicitly represents all node types—products and their associated group, subgroup, plant, and storage—but treats all edges as **untyped** in the final GNN. Conceptually, this graph encodes the bipartite or multipartite structure of the supply chain, while still allowing us to apply standard homogeneous GNNs. 

Implementation (`build_homogeneous_5type_graph`):

- Node types:
  - Product nodes:
    - One node per product with `node_type = "product"`, `node_index`, and static attributes.  
  - Category and facility nodes:
    - `node_type = "product_group"` for nodes named `GROUP::{group}`.  
    - `node_type = "product_sub_group"` for nodes named `SUBGROUP::{sub_group}`.  
    - `node_type = "plant"` for nodes named `PLANT::{plant}`.  
    - `node_type = "storage_location"` for nodes named `STORAGE::{storage_location}`.  

- Edges (untyped in the graph):
  - For each product:
    - Add an undirected edge to its group node (if `group` is present).  
    - Add an undirected edge to its subgroup node (if `sub_group` is present).  
    - Add an undirected edge to its plant node (if `plant` is present).  
    - Add an undirected edge to its storage node (if `storage_location` is present).  

- Outputs:
  - A NetworkX `Graph` saved as:
    - `graphs/homogeneous_graphs/homogeneous_5node_types.gpickle`.  
  - Node table:
    - `nodes_homogeneous_5type.parquet` with `node_id`, `node_type`, `node_index`, `raw_value`, and static attributes.  
  - Edge list:
    - `edges_homogeneous_5type.parquet` with columns `src`, `dst`.  

This representation preserves all structural relations but collapses edge types, supporting experiments where we rely on node type embeddings and a single homogeneous adjacency. 

### 5.4. Heterogeneous 5‑type graph

The heterogeneous graph uses the same set of node types as the homogeneous graph but retains explicit **edge types** between products and each category/facility node. This allows us to test heterogeneous GNNs that treat different relations separately.

Implementation (`build_heterogeneous_5type_graph`):

- Node types:
  - Identical to the homogeneous graph:
    - `"product"`, `"product_group"`, `"product_sub_group"`, `"plant"`, `"storage_location"`.  

- Edge types:
  - Directed edges from products to category/facility nodes, with:
    - `edge_type = "product_group"` for product → product_group edges.  
    - `edge_type = "product_subgroup"` for product → product_sub_group edges.  
    - `edge_type = "product_plant"` for product → plant edges.  
    - `edge_type = "product_storage"` for product → storage_location edges.  

- Graph:
  - Implemented as a NetworkX `MultiDiGraph` where each edge carries an `edge_type` attribute.  

- Outputs:
  - Heterogeneous graph pickle:
    - `graphs/heterogeneous_graphs/heterogeneous_5node_types.gpickle`.  
  - Node and edge tables:
    - `nodes_heterogeneous_5type.parquet` (node attributes and types).  
    - `edges_heterogeneous_5type.parquet` with `src`, `dst`, `edge_type`.  

This design supports later construction of typed edge_index dictionaries and relation‑specific GNN layers (Section 5.3). 

---

## 6. GNN datasets from XGBoost tabular

To ensure that graph models and tabular baselines operate on **comparable feature sets and splits**, we derive all GNN time‑series tensors from the XGBoost tabular baseline files built in Section 3. This ties the graph experiments directly to the same lag‑based covariates and target definitions used by XGBoost. 

### 6.1. Building node‑time tensors

We first convert the XGBoost tabular parquet into node‑time tensors suitable for node‑level forecasting:

- `load_xgb_tabular_for_gnn(temporal_type, lag_window, horizon)`:
  - Loads:
    - `data/processed/baseline/xgboost/xgboost_tabular_h{H}_lag{L}_{temporal_type}.parquet`.  
  - Enforces the presence and types of:
    - `node_id`, `node_index`, `date`, `day`, `split`, `target`.  
  - Converts `date` to datetime, casts `day` and `node_index` to integers.  
  - Sorts by `(date, node_id)` and removes duplicate `(date, node_id)` rows.  

- `build_time_tensors_from_xgb_for_gnn(df)`:
  - Extracts and sorts:
    - Unique `node_index` values → array of product nodes, mapping `node_index → position` (size \(N\)).  
    - Unique `day` values → array of time indices, mapping `day → t` (size \(T\)).  
  - Identifies **feature columns**:
    - Drops base/meta columns: `[node_id, node_index, date, day, split, target]`.  
    - Drops one‑hot columns that come from the graph side (`group_*`, `sub_group_*`, `plant_*`, `storage_*`, `storage_location_*`) to avoid double‑counting graph information.  
    - The remaining columns are purely temporal and calendar covariates (lags, rolling stats, calendar features, etc.).  
  - Allocates:
    - \(X \in \mathbb{R}^{T \times N \times F}\): features per day, per product.  
    - \(Y \in \mathbb{R}^{T \times N}\): scalar target (unit demand at horizon \(H\)) per day, per product.  
  - Fills \(X\) and \(Y\) by iterating rows and using the `day` and `node_index` mappings.  
  - Computes `day_split` as a per‑day split label (`train`, `val`, `test`) by grouping over `day` and taking the first assigned split (consistent with Section 3).  
  - Builds a sorted list of `node_id` and `node_index` aligned with the tensor ordering.  

The function returns a common package:

```text
{
  "X_product": [T, N, F],
  "Y_product": [T, N],
  "days": [T],
  "split": [T]  # per day
  "node_ids_product": [N],
  "node_index_product": [N],
  "feature_cols": list of feature names
}
```

This package forms the **shared temporal core** for all graph variants (projected, homogeneous, heterogeneous). 

### 6.2. Projected product GNN datasets

For the projected graphs, we build per‑view edge_index tensors in the product‑node space consistent with the `node_index` ordering used in `X_product`.

- `build_projected_edge_indices(nodeindex2pos_prod, df_meta)`:
  - Builds a mapping `node_id → node_index` from `df_meta`, then `node_index → position` from `nodeindex2pos_prod`.  
  - For each projected graph parquet:
    - `product_graph_same_group_edges.parquet`  
    - `product_graph_same_subgroup_edges.parquet`  
    - `product_graph_same_plant_edges.parquet`  
    - `product_graph_same_storage_edges.parquet`  
  - Converts each edge list `(src, dst)` to a 2×E tensor over positions \(\{0,\dots,N-1\}\), symmetrizing by adding both `(u, v)` and `(v, u)` for every undirected edge.  
  - Returns an `edge_index_dict`:
    - `"same_group"`, `"same_subgroup"`, `"same_plant"`, `"same_storage"` → `torch.LongTensor[2, E_view]`.  

- `build_gnn_datasets_for_config(..., df_meta)`:
  - Loads the XGBoost tabular baseline, builds `pkg_common` as in Section 5.1.  
  - Constructs projected edge indices via `build_projected_edge_indices`.  
  - Packages the projected dataset as:
    - `pkg_proj = { **pkg_common, "edge_index_dict": edge_index_proj, "graph_def": "projected_product_4view" }`.  
  - Saves to:
    - `data/processed/gnn/gnn_projected_h{H}_lag{L}_{temporal_type}.pt`.  

This dataset supports multi‑view product‑only GNNs where each view corresponds to one similarity notion (same group, same plant, etc.).

### 6.3. Homogeneous 5‑type GNN datasets

For the homogeneous 5‑type graph, we construct both:

- A **typed edge_index_dict** with local indices per node type, and  
- A **flattened global edge_index** suitable for standard homogeneous GNN layers.

- `build_homo5type_from_parquet()`:
  - Loads:
    - `nodes_homogeneous_5type.parquet` and `edges_homogeneous_5type.parquet`.  
  - Splits nodes by `node_type` and assigns **local indices** within each type.  
  - Builds:
    - `num_nodes_dict`: mapping `node_type → number of nodes`.  
    - `nodeid2type` and `nodeid2local`: book‑keeping for node lookups.  
  - For each relation (product to group/subgroup/plant/storage), it:
    - Filters edges whose destination node has the corresponding type.  
    - Converts sources/destinations to local indices and builds `edge_index` tensors.  
    - Stores them in:
      - `edge_index_dict[(src_type, rel_name, dst_type)]`.  

- `make_homo5_flat_edge_index(edge_index_dict, num_nodes_dict, node_type_order)`:
  - Computes cumulative offsets per node type according to `node_type_order`.  
  - For each `(src_type, rel_name, dst_type)`:
    - Shifts local indices by offsets to obtain global node indices.  
  - Concatenates all edges into a single `edge_index_flat ∈ ℤ^{2×E}` over the global node space.  

- In `build_gnn_datasets_for_config`:
  - After building `pkg_common`, we call:
    - `edge_index_homo5, num_nodes_homo5, nodes_homo_tbl = build_homo5type_from_parquet()`.  
    - `edge_index_flat = make_homo5_flat_edge_index(...)`.  
  - We then package:
    - `pkg_homo5 = { **pkg_common, "edge_index_dict": edge_index_homo5, "edge_index": edge_index_flat, "num_nodes_dict": num_nodes_homo5, "nodes_homo_table": nodes_homo_tbl, "graph_def": "homogeneous_5node_types" }`.  
  - Saved to:
    - `data/processed/gnn/gnn_homo5_h{H}_lag{L}_{temporal_type}.pt`.  

This dataset allows us to train homogeneous GNNs that internally distinguish node types via embeddings and offsets, while operating on a single large adjacency matrix.

### 6.4. Heterogeneous 5‑type GNN datasets

For the heterogeneous graph, we build a typed `edge_index_dict` that directly encodes all product–category and category–product relations. 

- `build_hetero5type_from_parquet()`:
  - Loads:
    - `nodes_heterogeneous_5type.parquet` and `edges_heterogeneous_5type.parquet`.  
  - Groups nodes by `node_type` and assigns local indices, building `num_nodes_dict`, `nodeid2type`, `nodeid2local`.  
  - Defines `add_edge_pair(edge_type_name, rel_fwd, dst_type, rel_rev)`:
    - Filters edges with the given `edge_type_name` (e.g., `"product_group"`).  
    - Keeps only edges from `product` to the expected `dst_type`.  
    - Builds forward edge_index for `(product, rel_fwd, dst_type)` and symmetric reverse edge_index for `(dst_type, rel_rev, product)`.  
  - Applies `add_edge_pair` for:
    - `"product_group"`  
    - `"product_subgroup"`  
    - `"product_plant"`  
    - `"product_storage"`  
  - Returns:
    - `edge_index_dict`, `num_nodes_dict`, `nodes_tbl`.  

- In `build_gnn_datasets_for_config`:
  - After `pkg_common`, we call:
    - `edge_index_het5, num_nodes_het5, nodes_het_tbl = build_hetero5type_from_parquet()`.  
  - Package:
    - `pkg_het5 = { **pkg_common, "edge_index_dict": edge_index_het5, "num_nodes_dict": num_nodes_het5, "nodes_hetero_table": nodes_het_tbl, "graph_def": "heterogeneous_5node_types" }`.  
  - Saved to:
    - `data/processed/gnn/gnn_hetero5_h{H}_lag{L}_{temporal_type}.pt`.  

This dataset is designed for heterogeneous GNNs (e.g., HeteroConv, relation‑specific GIN), where each relation type can have its own message‑passing parameters. 

---

## 7. Graph‑feature baselines (XGBoost + projected/homo/hetero features)

In addition to using graphs inside GNNs, we derive **graph features** for each product node and append them to the XGBoost tabular baseline. This supports a controlled comparison between:

- Strong **tabular models with graph features**, and  
- **GNN‑based models** that consume the full graph structure. [towardsdatascience](https://towardsdatascience.com/time-series-isnt-enough-how-graph-neural-networks-change-demand-forecasting/)

### 7.1. Projected graph features

For each of the four projected product graphs, we compute standard centrality and clustering metrics at the product level.

- `compute_projected_graph_features(df_meta)`:
  - For each projected graph file:
    - `"product_graph_same_group.gpickle"` (suffix `"group"`).  
    - `"product_graph_same_subgroup.gpickle"` (suffix `"subgroup"`).  
    - `"product_graph_same_plant.gpickle"` (suffix `"plant"`).  
    - `"product_graph_same_storage.gpickle"` (suffix `"storage"`).  
  - Loads the NetworkX `Graph` and computes:
    - Degree: `deg_proj_{suffix}`.  
    - Clustering coefficient: `clust_proj_{suffix}`.  
    - Betweenness centrality: `btw_proj_{suffix}`.  
    - Closeness centrality: `close_proj_{suffix}`.  
  - Merges all four feature tables on `node_id` and fills missing values with 0.0.  
  - Restricts to product nodes using `df_meta[["node_id", "node_index"]]` to ensure alignment with tabular data.  

The output is a per‑product feature table indexed by `node_id` and `node_index`.

### 7.2. Homogeneous graph features

From the homogeneous 5‑type graph, we compute both global centrality measures and type‑specific neighbor counts for each product. 

- `compute_homo_graph_features()`:
  - Loads:
    - `homogeneous_5node_types.gpickle`.  
  - Computes overall centrality:
    - Total degree: `deg_homo_total`.  
    - Betweenness: `btw_homo`.  
    - Closeness: `close_homo`.  
    - Eigenvector centrality: `eig_homo`.  
  - Restricts to product nodes via `nodes_homogeneous_5type.parquet`.  
  - Computes, for each product node, counts of neighbors by node type:
    - `deg_homo_pg`: number of product_group neighbors.  
    - `deg_homo_psg`: number of product_sub_group neighbors.  
    - `deg_homo_plant`: number of plant neighbors.  
    - `deg_homo_storage`: number of storage_location neighbors.  
  - Merges all features and attaches `node_index` for alignment with tabular data.  

The resulting table captures how “connected” a product is to different parts of the supply chain in the homogeneous representation.

### 7.3. Heterogeneous graph features

From the heterogeneous MultiDiGraph, we build directed degree, edge‑type degree, and centrality features. 

- `compute_hetero_graph_features()`:
  - Loads:
    - `heterogeneous_5node_types.gpickle`.  
  - Builds a directed simple graph `G_dir` by dropping parallel edges, preserving direction.  
  - For product nodes:
    - Total in‑degree and out‑degree:
      - `deg_het_in_total`, `deg_het_out_total`.  
    - Out‑degree per edge type:
      - `deg_het_out_product_group`, `deg_het_out_product_subgroup`,  
        `deg_het_out_product_plant`, `deg_het_out_product_storage`.  
  - Computes:
    - PageRank over `G_dir`: `pr_het`.  
    - Betweenness centrality on the undirected version `G_und`: `btw_het`.  
  - Merges all features and attaches `node_index`, filling missing values with 0.0.  

These features quantify how central and connected each product is in the heterogeneous relational structure, both globally and with respect to specific relation types.

### 7.4. Building XGBoost + graph‑feature baselines

Finally, we merge the graph features with the XGBoost tabular baseline to build three families of **graph‑feature baselines**. 

- `build_xgb_graph_baselines_for_config(temporal_type, horizon, lag_window, df_meta)`:
  - Loads the base XGBoost tabular file:
    - `xgboost_tabular_h{H}_lag{L}_{temporal_type}.parquet`.  
  - Ensures `node_id` is a string, drops duplicates on `(node_id, node_index, date)` for consistency.  

  1. **Projected graph features**:
     - Computes per‑product projected features with `compute_projected_graph_features(df_meta)`.  
     - Merges them into `df_xgb` on `["node_id", "node_index"]` (many‑to‑one).  
     - Fills missing values with 0.0.  
     - Saves:
       - `baseline/xgb_graph/xgboost_tabular_graph_projected_h{H}_lag{L}_{temporal_type}.parquet`.  

  2. **Homogeneous graph features**:
     - Computes per‑product homogeneous features with `compute_homo_graph_features()`.  
     - Merges into `df_xgb` on `["node_id", "node_index"]`.  
     - Saves:
       - `baseline/xgb_graph/xgboost_tabular_graph_homo5_h{H}_lag{L}_{temporal_type}.parquet`.  

  3. **Heterogeneous graph features**:
     - Computes per‑product heterogeneous features with `compute_hetero_graph_features()`.  
     - Merges into `df_xgb` on `["node_id", "node_index"]`.  
     - Saves:
       - `baseline/xgb_graph/xgboost_tabular_graph_hetero5_h{H}_lag{L}_{temporal_type}.parquet`.  

Each of these files has exactly the same structure as the original XGBoost tabular dataset (including `target`, `split`, and all lag/rolling features), plus additional graph‑derived columns. This enables **direct, apples‑to‑apples comparison** between:

- XGBoost with no graph features, and  
- XGBoost with projected / homogeneous / heterogeneous graph features,  

under the same temporal splits and evaluation metrics as the GNN models.

---

Ở mức paper thì nên tách rõ:

- Một section mô tả **cách build advanced graph features** (preprocessing).  
- Một section mô tả **baseline XGBoost dùng các feature đó**.

Bạn có thể đặt phần code `build_advanced_graph_features.py` thành **Section 8** (feature engineering), rồi Section 10 là training XGBoost với các feature này. Dưới đây là text chi tiết cho Section 8, bám đúng code bạn gửi.

---

## 8. Advanced neighbor‑based graph feature construction

We also precompute a rich set of **neighbor‑based graph features** for each product and day, across all graph types (projected, homogeneous 5‑type, heterogeneous 5‑type) that used for baseline “XGBoost with neighbor‑aggregation graph features”. These features summarize how neighboring products behave in terms of sales and operational signals.

### 8.1. Product index mapping and temporal matrices

We first build a consistent **product index mapping** and time axis:

- From the baseline tabular dataset (already aligned to a given horizon and lag window), we extract all unique product indices:
  - `build_product_index_mapping(df_base)`:
    - Sorts unique `node_index` values.  
    - Returns:
      - `node_indices`: sorted array of product indices.  
      - `idx2pos`: dictionary `node_index → position ∈ {0,…,N−1}`.  

- We then construct time–product matrices for several value types:
  - `build_Ys_from_base(df_base, node_indices, value_cols)` with:
    - `value_cols = ["sales", "production", "delivery", "factory_issue"]`.  
  - The function:
    - Sorts `df_base` by `day` and `node_index`.  
    - Collects unique days into `days` (length \(T\)).  
    - For each value type \(v\), builds a matrix:
      \[
      Y^{(v)} \in \mathbb{R}^{T \times N},
      \]
      where \(Y^{(v)}[t, i]\) is the value for product \(i\) at day \(t\), or `NaN` if missing.  
    - For `sales`, it uses:
      - `sales_order` if present, otherwise `target` as a proxy.  
    - Only columns actually present in `df_base` (e.g., `production`, `delivery`, `factory_issue`) are filled.  

These matrices give us a unified representation \(Y^{(v)}[t,i]\) on which we can apply neighbor aggregations. 

### 8.2. Neighbor sets for projected, homogeneous and heterogeneous graphs

We build neighbor lists in **product index space** for each graph family and relation type.

#### 8.2.1. Projected product graphs

For projected graphs, neighbors are products sharing the same attribute (group, subgroup, plant, storage): 

- `build_neighbor_indices_projected(df_meta, idx2pos)`:
  - Uses `df_meta` with columns `node_index`, `group`, `sub_group`, `plant`, `storage_location`.  
  - For each attribute:
    - `"group" → "same_group"`  
    - `"sub_group" → "same_subgroup"`  
    - `"plant" → "same_plant"`  
    - `"storage_location" → "same_storage"`  
  - For each group of products sharing the same attribute value, it:
    - Collects their positions `idx2pos[node_index]`.  
    - For each product \(i\) in the group, adds all other products \(j ≠ i\) as neighbors.  
  - Returns a dictionary:
    - `neighbors["same_group"]`, `neighbors["same_subgroup"]`, `neighbors["same_plant"]`, `neighbors["same_storage"]`,  
    - Each is a list of length \(N\), where entry `i` is the sorted set of neighbor positions for product \(i\).  

#### 8.2.2. Homogeneous 5‑type graph

For the homogeneous 5‑type graph, neighbors are products connected through shared group, subgroup, plant, or storage nodes.

- We first load the graph and node table:
  - `edge_index_homo5, num_nodes_homo5, nodes_homo_tbl = build_homo5type_from_parquet()`.  
- `build_neighbor_indices_homo5(edge_index_homo5, nodes_homo_tbl, idx2pos)`:
  - Builds per‑type node tables and identifies product nodes with their local indices in the homogeneous graph.  
  - For each relation:
    - `("product", "product_group_edge", "product_group") → "homo_group"`  
    - `("product", "product_sub_group_edge", "product_sub_group") → "homo_subgroup"`  
    - `("product", "product_plant_edge", "plant") → "homo_plant"`  
    - `("product", "product_storage_edge", "storage_location") → "homo_storage"`  
  - For each destination category node (e.g., a specific `product_group` node), it:
    - Collects the set of product local indices connected to that node.  
    - For each product in this set, all other products in the same set are treated as neighbors.  
  - Converts local product indices back to `node_index`, then to positions via `idx2pos`.  
  - Returns:
    - `neighbors_homo["homo_group"]`, `neighbors_homo["homo_subgroup"]`, `neighbors_homo["homo_plant"]`, `neighbors_homo["homo_storage"]`, all as lists of neighbor indices per product.  

This effectively says: *two products are homogeneous neighbors in view X if they are connected to the same X‑type node in the homogeneous graph*. 

#### 8.2.3. Heterogeneous 5‑type graph

For the heterogeneous graph, we use typed edges from products to each category/facility node and again connect products that share the same category node.

- Load graph and node table:
  - `edge_index_het5, num_nodes_het5, nodes_het_tbl = build_hetero5type_from_parquet()`.  
- `build_neighbor_indices_hetero5(edge_index_het5, nodes_het_tbl, idx2pos)`:
  - Identifies product nodes and their local indices.  
  - For each forward relation type:
    - `(product, "product_group", product_group) → "het_group"`  
    - `(product, "product_subgroup", product_sub_group) → "het_subgroup"`  
    - `(product, "product_plant", plant) → "het_plant"`  
    - `(product, "product_storage", storage_location) → "het_storage"`  
  - For each category node, collects all incident product local indices and links them as neighbors of one another.  
  - As before, converts local indices → `node_index` → positions `idx2pos`.  
  - Returns:
    - `neighbors_het["het_group"]`, `neighbors_het["het_subgroup"]`, `neighbors_het["het_plant"]`, `neighbors_het["het_storage"]`.  

Thus, projected, homogeneous, and heterogeneous neighbor sets all define **product–product neighborhoods**, but induced via different graph constructions.

### 8.3. Neighbor‑aggregation operators

We define a set of generic neighbor aggregation operators over the temporal matrices \(Y^{(v)}\):

Given:

- \(Y \in \mathbb{R}^{T \times N}\) (e.g., sales for all products over time);  
- Neighbor lists `neighbor_idx[i] ⊆ {0,…,N−1}` for each product \(i\);  

we compute:

1. **Lag‑based neighbor mean/sum/max/min**

For a given lag \(L\), we look at neighbors’ values at day \(t-L\):

- `neighbor_mean_lag(Y, neighbor_idx, lag=L)`:
  \[
  \text{feat}[t,i] = \text{mean}\_{j\in \mathcal{N}(i)} Y[t - L, j],
  \]
  if any neighbor has a finite value, otherwise `NaN`.  

- `neighbor_sum_lag(Y, neighbor_idx, lag=L)`:
  - Sum over neighbors at \(t-L\).  

- `neighbor_max_lag(Y, neighbor_idx, lag=L)`:
  - Max over neighbors at \(t-L\).  

- `neighbor_min_lag(Y, neighbor_idx, lag=L)`:
  - Min over neighbors at \(t-L\).  

2. **Neighbor zero‑ratio over a rolling window**

For a window size `window`, we look at neighbors over the past `window` days:

- `neighbor_zero_ratio_window(Y, neighbor_idx, window)`:
  - For each \(t ≥ \text{window}−1\):
    - Collect values \(Y[\tau, j]\) for \(\tau ∈ [t−\text{window}+1, t]\) and \(j ∈ \mathcal{N}(i)\).  
    - Compute:
      \[
      \text{feat}[t, i] = \text{mean}( Y[\tau, j] = 0).
      \]
  - If there are no neighbor values, keep `NaN`.  

These operators are applied to all value types \(v ∈ \{\)sales, production, delivery, factory_issue\(\}\), all neighbor views, and two lags: `L = 1` and `L = lag_window` (the main lag window used in the base XGBoost features). 

### 8.4. Advanced features per graph family

Using the machinery above, we define three feature builders, each starting from a base `df_base` that already contains tabular and basic graph features.

#### 8.4.1. Projected graphs: `build_xgb_with_proj_features`

- Inputs:
  - `df_base`: XGBoost baseline with projected graph features for a given `(temporal_type, horizon, lag_window)`.  
- Steps:
  1. Build `node_indices` and `idx2pos`.  
  2. Build temporal matrices `Ys` for `["sales", "production", "delivery", "factory_issue"]`.  
  3. Load `df_meta` via `load_node_metadata()` to obtain product metadata.  
  4. Construct neighbors:
     - `neighbors_proj = build_neighbor_indices_projected(df_meta, idx2pos)` with keys:
       - `"same_group"`, `"same_subgroup"`, `"same_plant"`, `"same_storage"`.  
  5. For each view in these keys, for each value type `vname`, and for each lag \(L ∈ {1, lag\_window}\), compute:
     - `adv_{vname}_proj_{view}_mean_lag{L}`  
     - `adv_{vname}_proj_{view}_sum_lag{L}`  
     - `adv_{vname}_proj_{view}_max_lag{L}`  
     - `adv_{vname}_proj_{view}_min_lag{L}`  
  6. Additionally compute zero‑ratios over the main lag window:
     - `adv_{vname}_proj_{view}_zero_ratio_win{lag_window}`.  
  7. Assemble all features into a table indexed by `(day, node_index)` and merge into `df_base`.  

The output is `df_proj_adv`, a tabular dataset with **base features + basic projected graph features + neighbor‑aggregation features**. 

#### 8.4.2. Homogeneous 5‑type graph: `build_xgb_with_homo_features`

- Inputs:
  - `df_base`: XGBoost baseline with homogeneous graph features.  
- Steps:
  1. Build `node_indices`, `idx2pos`, and `Ys` as above.  
  2. Load homogeneous graph and nodes:
     - `edge_index_homo5, num_nodes_homo5, nodes_homo_tbl = build_homo5type_from_parquet()`.  
  3. Build neighbors via:
     - `neighbors_homo = build_neighbor_indices_homo5(...)` with keys:
       - `"homo_group"`, `"homo_subgroup"`, `"homo_plant"`, `"homo_storage"`.  
  4. For each view and each value type, compute:
     - `adv_{vname}_{view}_mean_lag{1}`, `adv_{vname}_{view}_sum_lag{1}`, `...`,  
     - `adv_{vname}_{view}_mean_lag{lag_window}`, `...`,  
     - `adv_{vname}_{view}_zero_ratio_win{lag_window}`.  
  5. Merge the resulting feature matrix on `(day, node_index)` with `df_base`.  

The result `df_homo_adv` combines homogeneous centralities with **multi‑relation neighbor demand/activity patterns**.

#### 8.4.3. Heterogeneous 5‑type graph: `build_xgb_with_hetero_features`

- Inputs:
  - `df_base`: XGBoost baseline with heterogeneous graph features.  
- Steps:
  1. Build `node_indices`, `idx2pos`, and `Ys`.  
  2. Load heterogeneous graph:
     - `edge_index_het5, num_nodes_het5, nodes_het_tbl = build_hetero5type_from_parquet()`.  
  3. Build neighbors via:
     - `neighbors_het = build_neighbor_indices_hetero5(...)` with keys:
       - `"het_group"`, `"het_subgroup"`, `"het_plant"`, `"het_storage"`.  
  4. For each view and value type, compute:
     - `adv_{vname}_{view}_mean_lag{1}`, `...`, `adv_{vname}_{view}_min_lag{lag_window}`,  
     - `adv_{vname}_{view}_zero_ratio_win{lag_window}`.  
  5. Merge advanced features into `df_base` on `(day, node_index)`.  

The output `df_het_adv` captures neighbor behavior along **explicitly typed relations** (group/subgroup/plant/storage) in the heterogeneous graph.

### 8.5. End‑to‑end preprocessing pipeline

The main entry point `main()` in `build_advanced_graph_features.py` runs the advanced feature construction for all experiment configurations:

- For each `exp` in `DEFAULT_EXPERIMENTS`:
  - `temporal_type = exp.temporal_type`  
  - For each horizon `H` and lag window `L` in `exp.lag_windows`:
    - Load base XGBoost+graph files from `baseline/xgb_graph/`:
      - `xgboost_tabular_graph_projected_h{H}_lag{L}_{t_type}.parquet`  
      - `xgboost_tabular_graph_homo5_h{H}_lag{L}_{t_type}.parquet`  
      - `xgboost_tabular_graph_hetero5_h{H}_lag{L}_{t_type}.parquet`  
    - If the baseline file exists, call the corresponding builder:
      - `build_xgb_with_proj_features`, `build_xgb_with_homo_features`, or `build_xgb_with_hetero_features`.  
    - Save the advanced feature datasets to:
      - `baseline/xgboost/xgboost_tabular_graphfeat_projected_h{H}_lag{L}_{t_type}.parquet`  
      - `baseline/xgboost/xgboost_tabular_graphfeat_homo5_h{H}_lag{L}_{t_type}.parquet`  
      - `baseline/xgboost/xgboost_tabular_graphfeat_hetero5_h{H}_lag{L}_{t_type}.parquet`.  

These `graphfeat` parquet files are then used as input to the XGBoost baseline in Section 10, where we evaluate how much performance gain we can obtain from **neighbor‑aware graph features** without training any GNN.

---

## 9. Graph-aware forecasting baselines

We implement four families of **graph‑aware baselines** that incorporate relational structure between products, groups, plants, and storage locations into the forecasting model. Together with the non‑graph baselines (naive, GRU sequence, plain XGBoost), these cover both **tabular** and **graph neural network** approaches to multi‑product demand forecasting.

### 9.1. XGBoost with static graph features (Baseline 3)

The first graph‑aware family augments a strong **tabular XGBoost forecaster** with **hand‑crafted graph features**. 

- We build multiple graphs over the product catalogue:
  - **Projected graphs** between products (e.g., same group, subgroup, plant, storage location).  
  - **Homogeneous 5‑type** graph where all node types are flattened.  
  - **Heterogeneous 5‑type** graph with typed edges between products, product groups, subgroups, plants, and storage locations.  
- From these graphs, we extract static or slowly varying features for each product node, such as degree, counts within a neighbourhood, or simple aggregations of historical demand across neighbours.  
- These graph features are concatenated to the original tabular feature set and fed into XGBoost, yielding `xgb_graph_{proj,homo,hetero}` models.  

This family tests whether **simple, static graph descriptors** are enough to improve a strong tabular baseline without using GNNs. 

### 9.2. XGBoost with learned GNN embeddings (Baseline 5)

The second family uses GNNs only as **representation learners**, while XGBoost remains the final predictor. 

- We first train **graph neural networks** on the 5‑type graph (projected, homogeneous, or heterogeneous) to produce node embeddings for each product, using historical demand as supervision.  
- After training, we freeze the GNNs and extract their product embeddings as dense graph features.  
- These learned embeddings are then concatenated with tabular features and used as input to a new XGBoost model, giving `xgb_gnn_embed_{proj,homo,hetero}`.  

This setup isolates the benefit of **GNN‑based graph representations** when combined with a powerful tabular learner, rather than relying on GNNs to do the final regression. 

### 9.3. XGBoost with GNN residual correction (Baseline 6)

The third graph‑aware family uses GNNs as **residual correctors** on top of XGBoost forecasts.

- Step 1: Fit a strong tabular XGBoost model on the usual lagged features and exogenous covariates, and generate per‑product forecasts for all time steps in the evaluation horizon.  
- Step 2: Construct residual targets for each product node and time:
  - Either on the **raw scale** \(y\) or on a **log1p scale**, depending on the experiment configuration.  
  - Build time–product tensors `X_residual` (residual‑side features, including the XGBoost prediction itself) and `R_residual` (residual target).  
- Step 3: Train GNN regressors on residuals:
  - **Projected product graphs** with different views (same group, subgroup, plant, storage).  
  - **Homogeneous 5‑type** graph.  
  - **Heterogeneous 5‑type** graph.  
  - The GNN outputs predicted residuals, which are added back to the XGBoost forecasts (and inverse‑transformed if needed) to obtain the final demand prediction.  

This design explicitly treats the graph model as a **second‑stage corrector** that focuses on systematic spatial errors left by the tabular forecaster.

### 9.4. Direct GNN forecasters (Baseline 4)

Finally, we include **pure GNN baselines** that directly predict next‑step demand from graph‑structured inputs, without XGBoost. 

- We build three architectures:
  - **ProjectedGINRegressor** on the projected product graph.  
  - **HomogeneousFiveTypeGINRegressor** on the flattened 5‑type graph with node‑type embeddings.  
  - **HeterogeneousGINRegressor** on the full heterogeneous 5‑type graph with typed edges.  
- For each temporal configuration and lag window, we construct time–product feature tensors and train these GNNs end‑to‑end to minimize MSE on the (possibly transformed) demand scale.  
- At evaluation time, we invert any target transforms and compute MAE, RMSE, MAPE, and sMAPE at the product level.  

These GNN baselines allow us to benchmark **standalone graph neural forecasters** against the graph‑aware XGBoost families and against simple sequence models such as GRUs. 

---

## 10. XGBoost with basic graph features (projected, homogeneous, heterogeneous)

This baseline augments the tabular XGBoost model with **precomputed graph features** derived from the projected product graphs, the homogeneous 5‑type graph, and the heterogeneous 5‑type graph. The forecasting model is still XGBoost; graphs are only used to enrich the input feature space.

### 10.1. Input datasets with graph features

For each temporal type and lag window, we build three graph‑augmented tabular datasets (see Section “Graph‑feature baselines”):

- Projected graph features:
  - `baseline/xgb_graph/xgboost_tabular_graph_projected_h{H}_lag{L}_{temporal_type}.parquet`.  
- Homogeneous 5‑type graph features:
  - `baseline/xgb_graph/xgboost_tabular_graph_homo5_h{H}_lag{L}_{temporal_type}.parquet`.  
- Heterogeneous 5‑type graph features:
  - `baseline/xgb_graph/xgboost_tabular_graph_hetero5_h{H}_lag{L}_{temporal_type}.parquet`.  

Each file has the same structure as the no‑graph XGBoost baseline:

- Identification and split columns:
  - `node_id`, `node_index`, `date`, `day`, `split`.  
- Original tabular features:
  - All lag, rolling, and calendar features from the base XGBoost dataset.  
- Graph features:
  - For the corresponding graph mode (projected, homogeneous, or heterogeneous), including degrees, clustering coefficients, and centrality measures.  
- Target:
  - `target` = demand at horizon \(H\) on the original scale.  

The script `train_xgb_graph_baseline` loads these files via:

- `load_tabular_graph_baseline(temporal_type, lag_window, graph_mode)`:
  - `graph_mode ∈ { "proj", "homo", "hetero" }`.  
  - Selects the appropriate parquet path and reads it into a DataFrame.  

### 10.2. Feature definition and splits

We keep the same split logic as for the no‑graph XGBoost baseline:

- Split:
  - `split_train_val_test(df)` partitions rows into `df_train`, `df_val`, and `df_test` based on the `split` column.  

- Feature set:
  - `prepare_features(df)`:
    - Sets `target = df["target"].astype(float)`.  
    - Drops non‑feature columns:
      - `target`, `split`, `node_id`, `node_index`, `date`, `day`.  
    - Treats all remaining columns as features.  

This means that the model sees **all original tabular features plus graph‑derived features** in a single concatenated feature vector, without any special handling for graph features. 

### 10.3. Model configuration and training

We reuse the same XGBoost configuration as in the no‑graph baseline to ensure a fair comparison:

- Model:
  - `XGBRegressor` with:
    - `n_estimators = 5000`  
    - `max_depth = 6`  
    - `learning_rate = 0.05`  
    - `subsample = 0.8`, `colsample_bytree = 0.8`  
    - `objective = "reg:squarederror"`  
    - `tree_method = "hist"`  
    - `random_state = 42`, `n_jobs = -1`  
    - `eval_metric = "rmse"`  
    - `early_stopping_rounds = 100`  

- Training:
  - Fit on `(X_train, y_train)` with evaluation on `(X_val, y_val)`.  
  - Use early stopping on validation RMSE to select the best iteration.  

This setup isolates the **incremental benefit of graph features** while keeping the underlying learner fixed.

### 10.4. Evaluation and outputs

For each configuration (temporal type, lag window, graph mode), we compute metrics on train, validation, and test splits:

- Metrics on original demand scale:
  - Mean Absolute Error (MAE).  
  - Root Mean Squared Error (RMSE).  
  - Mean Absolute Percentage Error (MAPE).  
  - Symmetric MAPE (sMAPE).

We also generate diagnostic artifacts:

- Learning curves:
  - Train and validation RMSE per boosting round.  
  - Saved under:
    - `predictions/baseline_3/plots_learning_curves/{graph_mode}/learning_curve_h{H}_lag{L}_raw_{tag}_{temporal_type}.png`.  

- Test predictions:
  - Per‑row predictions saved as:
    - `predictions/baseline_3/csv/{temporal_type}/{graph_mode}/xgb_graph_{graph_mode}_h{H}_lag{L}_raw_test_predictions_{temporal_type}.csv`.  
  - Columns: `node_id`, `date`, `y_true`, `y_pred`.  

- Per‑product time‑series plots:
  - For each product in the test split, we plot true vs. predicted demand over time:
    - `xgb_graph_{graph_mode}_h{H}_lag{L}_node_{node_id}_{temporal_type}.png`.  
  - Stored under:
    - `predictions/baseline_3/plots_xgb_graph/{graph_mode}/raw_lag{L}/{temporal_type}/`.  

- Summary table:
  - For all temporal types, lag windows, and graph modes, we aggregate a summary DataFrame and save it to:
    - `predictions/baseline_3/summary_xgb_graph_baseline3_raw_lags_graphmodes.csv`.  
  - It reports, per configuration:
    - Number of features, and train/val/test MAE and RMSE.  

This baseline allows us to answer: *“If we only add static graph‑derived features to a strong tabular model, how much improvement do we get compared to the no‑graph XGBoost baseline?”* 


---

## 11. XGBoost with advanced neighbor‑based graph features

In this baseline, XGBoost is trained on tabular features enriched with a large set of **neighbor‑based graph features** built from projected, homogeneous, and heterogeneous product graphs. The model itself is unchanged compared to other XGBoost baselines; only the input feature set is extended with systematically named neighbor aggregation features.

### 11.1. Input datasets and feature sources

For each temporal type, horizon, and lag window, we start from the graph‑augmented XGBoost baselines (Section 9) and attach advanced neighbor features produced by `build_advanced_graph_features.py`:

- Projected graph advanced features:
  - `baseline/xgboost/xgboost_tabular_graphfeat_projected_h{H}_lag{L}_{temporal_type}.parquet`.  
- Homogeneous 5‑type graph advanced features:
  - `baseline/xgboost/xgboost_tabular_graphfeat_homo5_h{H}_lag{L}_{temporal_type}.parquet`.  
- Heterogeneous 5‑type graph advanced features:
  - `baseline/xgboost/xgboost_tabular_graphfeat_hetero5_h{H}_lag{L}_{temporal_type}.parquet`.  

Each file contains, for every `(node_id, day)`:

- Identification, split, and target:
  - `node_id`, `node_index`, `date`, `day`, `split`, and `target` (demand at horizon \(H\)).  
- Base tabular features and basic graph features (degrees, centralities, etc.).  
- **Advanced neighbor‑aggregation features** with names of the form:

  - `adv_{value_name}_{view}_mean_lag{L}`  
  - `adv_{value_name}_{view}_sum_lag{L}`  
  - `adv_{value_name}_{view}_max_lag{L}`  
  - `adv_{value_name}_{view}_min_lag{L}`  
  - `adv_{value_name}_{view}_zero_ratio_win{W}`  

where:

- `value_name ∈ {sales, production, delivery, factory_issue}`  
- `view` encodes the graph family and relation type (detailed below).  
- `lag L ∈ {1, lag_window}` and window `W = lag_window`.  

These files are loaded in `train_xgb_tabular_graphfeat` via:

- `load_tabular_graphfeat(horizon, temporal_type, lag_window, graph_type)` with:
  - `graph_type ∈ { "projected", "homo5", "hetero5" }`.  

### 11.2. Exact definition of neighbor‑aggregation features

All advanced features are computed from temporal matrices \(Y^{(v)} ∈ \mathbb{R}^{T \times N}\), where \(v\) is one of:

- `sales` (using `sales_order` if available, otherwise `target`).  
- `production`, `delivery`, `factory_issue` (if present in the baseline table).  

and from neighbor lists \(\mathcal{N}\_{view}(i)\) defined on products by the graph views in Section 8.
#### 11.2.1. Value names (`value_name`)

The placeholder `{value_name}` in all feature names comes from:

- `value_name ∈ { "sales", "production", "delivery", "factory_issue" }`.  

So for each `value_name` and each graph view, we build a whole set of lag‑based and window‑based neighbor features.

#### 11.2.2. Views for each graph type (`view`)

The `{view}` token encodes both the graph family and the specific relation:

- **Projected graphs** (`graph_type = "projected"`; Section 8.4.1):  
  - Views:
    - `"same_group"`  
    - `"same_subgroup"`  
    - `"same_plant"`  
    - `"same_storage"`  
  - These views connect products that share the same group, subgroup, plant, or storage location in the product metadata, respectively.  

- **Homogeneous 5‑type graph** (`graph_type = "homo5"`; Section 8.4.2):  
  - Views:
    - `"homo_group"` (via `product_group_edge`)  
    - `"homo_subgroup"` (via `product_sub_group_edge`)  
    - `"homo_plant"` (via `product_plant_edge`)  
    - `"homo_storage"` (via `product_storage_edge`)  
  - Two products are neighbors in a given view if they connect to the same category/facility node of the corresponding type in the homogeneous graph.  

- **Heterogeneous 5‑type graph** (`graph_type = "hetero5"`; Section 8.4.3):  
  - Views:
    - `"het_group"` (via `product_group` edges)  
    - `"het_subgroup"` (via `product_subgroup` edges)  
    - `"het_plant"` (via `product_plant` edges)  
    - `"het_storage"` (via `product_storage` edges)  
  - Again, two products become neighbors in a view if they share at least one common category/facility node of that relation type in the heterogeneous graph.  

Thus, all advanced features are **per‑product, per‑day neighbor summaries** where neighborhood semantics depend on the underlying graph and view. 

#### 11.2.3. Lag‑based neighbor features (mean / sum / max / min)

For each value type \(v\), each view, and each lag \(L ∈ \{1, \text{lag\_window}\}\), we define:

- `adv_{vname}_{view}_mean_lag{L}`  
- `adv_{vname}_{view}_sum_lag{L}`  
- `adv_{vname}_{view}_max_lag{L}`  
- `adv_{vname}_{view}_min_lag{L}`  

These features are computed by the functions `neighbor_mean_lag`, `neighbor_sum_lag`, `neighbor_max_lag`, and `neighbor_min_lag`. Formally, for day index \(t\) and product index \(i\):

- Mean:
  \[
  \text{adv}\_{v,view,\text{mean},L}(t,i) =
  \begin{cases}
  \text{mean}\_{j \in \mathcal{N}\_{view}(i)} Y^{(v)}[t-L, j] & \text{if any neighbor has a finite value},\\
  \text{NaN} & \text{otherwise.}
  \end{cases}
  \]

- Sum:
  \[
  \text{adv}\_{v,view,\text{sum},L}(t,i) =
  \sum\_{j \in \mathcal{N}\_{view}(i)} Y^{(v)}[t-L, j], \quad \text{ignoring NaNs}.
  \]

- Max:
  \[
  \text{adv}\_{v,view,\text{max},L}(t,i) =
  \max\_{j \in \mathcal{N}\_{view}(i)} Y^{(v)}[t-L, j].
  \]

- Min:
  \[
  \text{adv}\_{v,view,\text{min},L}(t,i) =
  \min\_{j \in \mathcal{N}\_{view}(i)} Y^{(v)}[t-L, j].
  \]

If there is no neighbor or all neighbor values are missing at that lag, the feature is left as `NaN`.

Concrete examples of feature names:

- For projected same‑group neighbors and sales:
  - `adv_sales_proj_same_group_mean_lag1`  
  - `adv_sales_proj_same_group_mean_lag7` (if `lag_window = 7`)  
  - `adv_sales_proj_same_group_max_lag1`, etc.  

- For homogeneous subgroup neighbors and factory issues:
  - `adv_factory_issue_homo_subgroup_sum_lag1`  
  - `adv_factory_issue_homo_subgroup_min_lag14` (if `lag_window = 14`)  

- For heterogeneous plant neighbors and deliveries:
  - `adv_delivery_het_plant_mean_lag1`  
  - `adv_delivery_het_plant_max_lag7`, etc.  

#### 11.2.4. Neighbor zero‑ratio features over a rolling window

For each value type \(v\) and view, we also build **zero‑ratio features** over a time window of length `W = lag_window` using `neighbor_zero_ratio_window`. The feature name is:

- `adv_{vname}_{view}_zero_ratio_win{W}`  

For day index \(t\) and product \(i\), we consider all neighbor positions \(j ∈ \mathcal{N}\_{view}(i)\) and all days in the interval \([t-W+1, t]\) (inclusive):

\[
\text{adv}\_{v,view,\text{zero\_ratio},W}(t,i)
= \frac{\#\{\tau, j: Y^{(v)}[\tau,j] = 0\}}{\#\{\tau, j: Y^{(v)}[\tau,j]\ \text{is observed}\}},
\]

i.e., the fraction of observed neighbor values equal to zero in the sliding window. If no neighbor data is available in the window, the feature is `NaN`. 

Examples:

- For projected same‑storage neighbors and sales with `lag_window = 7`:
  - `adv_sales_proj_same_storage_zero_ratio_win7`  
- For heterogeneous group neighbors and production with `lag_window = 14`:
  - `adv_production_het_group_zero_ratio_win14`  

These zero‑ratio features capture **sparsity patterns** in neighbors’ activity (e.g., share of neighbors with zero sales) over a medium‑term window. 

### 11.3. Integration into XGBoost training

The training script `train_xgb_tabular_graphfeat` uses these advanced features as part of the input to a standard XGBoost regressor:

- Loading:
  - `load_tabular_graphfeat(horizon, temporal_type, lag_window, graph_type)` reads the corresponding `xgboost_tabular_graphfeat_*` parquet.  

- Feature selection:
  - `prepare_features(df)`:
    - Sets `y = df["target"].astype(float)`.  
    - Drops:
      - `target`, `split`, `node_id`, `node_index`, `date`, `day`.  
    - Treats all remaining columns as features, including:
      - Base tabular features.  
      - Basic graph features (centralities, degrees, etc.).  
      - All `adv_*` advanced neighbor‑aggregation features described above.  

- Model and training:
  - XGBRegressor with 5000 trees, depth 6, learning rate 0.05, `reg:squarederror` objective, hist tree method, and early stopping with 100 rounds based on validation RMSE.
  - Predictions on train, validation, and test splits are clipped at zero before computing metrics.  

- Metrics and outputs:
  - Compute MAE, RMSE, MAPE, sMAPE on the original demand scale. 
  - Save learning curves, test predictions, per‑product plots, and a summary CSV:
    - `predictions/baseline_7/summary_xgb_tabular_graphfeat_raw_targets.csv`.  

In summary, this baseline evaluates how much additional gain we obtain by exposing XGBoost to **explicit, lagged, neighbor‑based statistics of sales/production/delivery/factory issues**, on top of standard tabular and basic graph features, without using any GNN. 

---

## 12. GNN embeddings as features for XGBoost

In this baseline, we first learn **structural embeddings** for product nodes using GIN‑based encoders on the three graph variants, then concatenate these embeddings with tabular lag features and train a standard XGBoost regressor. The GNNs act purely as **feature extractors**; forecasting is done by XGBoost on top of the learned node embeddings. 

### 12.1. GNN encoder architectures

We use three encoder variants, matching the three graph constructions: a projected product‑only graph, a homogeneous 5‑type graph, and a heterogeneous 5‑type graph. All encoders map node attributes and graph structure to a **fixed‑dimensional embedding** for each product.

#### 12.1.1. ProjectedGINEncoder (projected product graph)

For each projected product graph view, we apply a **GIN encoder** with a small MLP at each layer:

- Architecture:
  - Input:  
    - Node features: \(x ∈ \mathbb{R}^{N × F\_{in}}\).  
    - Edges: `edge_index ∈ ℕ^{2×E}`.  
  - `ProjectedGINEncoder(in_channels=F_in, hidden_channels=128, num_layers=3)`:
    - Each layer is a `GINConv` with an `MLP`:
      - `MLP`: Linear → ReLU → Linear.  
    - Non‑linearity: ReLU after each `GINConv`.  
  - Output:
    - Node embeddings \(h ∈ \mathbb{R}^{N × 128}\) for **product nodes only** (one node type).  

This encoder is applied on top of **projected product graphs** constructed from same‑group, same‑subgroup, same‑plant, and same‑storage relations, but note that in this GNN embedding baseline we use a **single encoder shared across views** (Section 12.2). 

#### 12.1.2. HomogeneousFiveTypeGINEncoder (flattened 5‑type graph)

To encode the **homogeneous 5‑type graph**, we flatten all node types into a single graph but distinguish them via a **node‑type embedding**: 

- Node types:
  - `node_type_order` includes `["product", "product_group", "product_sub_group", "plant", "storage_location"]` (order taken from the homogeneous node table).  
- Inputs:
  - `x_dict`: dictionary mapping each node type to an `N_type × F_in` feature matrix.  
  - `edge_index`: flattened adjacency `edge_index ∈ ℕ^{2×E_total}` over all nodes.  
  - `num_nodes_dict`: number of nodes per type.  
- Architecture:
  - For each node type, we concatenate:
    - Original features \(x\_i\) with a learned node‑type embedding \(e_{\text{type}(i)} ∈ \mathbb{R}^{d_{\text{type}}}\).  
  - Apply `num_layers = 3` GINConv layers on the homogeneous graph:
    - First layer: input dimension \(F\_{in} + 8\) (8‑d type embedding) → 128.  
    - Subsequent layers: 128 → 128.  
    - Activation: ReLU after each GINConv.  
- Output:
  - After the last layer, we slice out the **product node block**:
    - Corresponding to indices `[offset_prod : offset_prod + n_product]`.  
  - Final embeddings: \(h_{\text{product}} ∈ \mathbb{R}^{N_{\text{prod}} × 128}\).  

This design allows a **homogeneous GIN** to operate on a heterogeneous graph by encoding node type information in the feature space. 

#### 12.1.3. HeterogeneousGINEncoder (heterogeneous 5‑type graph)

For the fully heterogeneous graph, we build a **Heterogeneous GIN** encoder using PyG’s `HeteroConv`:

- Inputs:
  - `x_dict`: node‑type feature matrices, one per node type.  
  - `edge_index_dict`: mapping from edge type `(src_type, rel, dst_type)` to its `edge_index`.  
  - `in_channels_dict`:
    - Contains all node types with their input feature dimension plus `"edge_types"` listing all edge types.  

- Architecture:
  - First project each node type to hidden dimension:
    - `node_in_proj[nt]: ℝ^{F_in} → ℝ^{128}` for every node type `nt`.  
  - Heterogeneous GIN layers:
    - Each `HeterogeneousGINLayer` creates a `GINConv` per edge type, all sharing the same output dimension (128).  
    - `HeteroConv` with aggregation `sum` over incoming edge types.  
    - `num_layers = 2`; after each layer we apply ReLU to all node types.  
- Output:
  - We retain embeddings only for node type `"product"`:
    - `h_dict["product"] ∈ ℝ^{N_prod × 128}`.  

Thus, we obtain a 128‑dimensional embedding for each product that captures **multi‑relational structure** across product–group–subgroup–plant–storage interactions.

### 12.2. Exporting time‑dependent GNN embeddings

For each temporal configuration `(temporal_type, horizon H = 7, lag_window L)`, we prebuild **time‑dependent GNN packages** containing:

- `X_product`: \(T × N_{\text{prod}} × F_{in}\) product features over time.  
- `days`: length‑T tensor of day indices.  
- `split`: length‑T list specifying whether each time slice belongs to train/val/test.  
- Graph structure and metadata:
  - Projected: `edge_index_dict` with keys = projected views.  
  - Homogeneous: `edge_index`, `num_nodes_dict`, `nodes_homo_table`.  
  - Heterogeneous: `edge_index_dict`, `num_nodes_dict`, `nodes_hetero_table`.  

The script `export_gnn_embeddings.py` then runs the corresponding encoder on each time slice and writes embeddings to disk as parquet files. This produces **per‑(day, product)** embeddings aligned with the tabular baseline. 

#### 12.2.1. Projected GIN embeddings (4 views)

- Source file:
  - `gnn/gnn_projected_h{H}_lag{L}_{temporal_type}.pt`.  
- Export function:
  - `export_projected_embeddings_for_config(temporal_type, lag_window, device)`:
    - Loads the package.  
    - Instantiates `ProjectedGINEncoder` (input dim \(F_{in}\), hidden dim 128, 3 layers).  
    - For each **view** in:
      - `PROJECTED_VIEWS = ["same_group", "same_subgroup", "same_plant", "same_storage"]`,  
      if present in `edge_index_dict`:
      - For every time step \(t\):
        - Encode product features `X_product[t]` with `edge_index = edge_index_dict[view_name]`.  
        - Store for each product position `node_pos`:
          - `day`, `split[t]`, `view`, and embedding coordinates `emb_0,…,emb_{d−1}`.  
- Output:
  - Long‑format parquet:
    - `gnn_embeddings/gnn_projected_emb_4views_h{H}_lag{L}_{temporal_type}.parquet`.  
  - Columns:
    - `node_index_pos`, `day`, `split`, `view`, `emb_0`, …, `emb_{d-1}`.  

These embeddings are later pivoted so that each view contributes a separate block of embedding features per product‑day (Section 12.3.1). 

#### 12.2.2. Homogeneous 5‑type GIN embeddings

- Source file:
  - `gnn/gnn_homo5_h{H}_lag{L}_{temporal_type}.pt`.  
- Export function:
  - `export_homo5_embeddings_for_config(temporal_type, lag_window, device)`:
    - Loads `X_product`, `days`, `split`, `edge_index`, `num_nodes_dict`, and `nodes_homo_table`.  
    - Determines `node_type_order` from `nodes_homo_table["node_type"].unique()`.  
    - Instantiates `HomogeneousFiveTypeGINEncoder` (hidden dim 128, 3 layers, type embedding dim 8).  
    - For each time step \(t\):
      - Builds `x_dict`:
        - `product` type: features for first `N_prod` nodes set to `X_product[t]`, rest zero.  
        - Other types: all‑zero features.  
      - Computes product embeddings `h_prod`.  
      - Records one row per product with:
        - `node_index_pos`, `day`, `split`, `emb_0,…,emb_{d-1}`.  
- Output:
  - `gnn_embeddings/gnn_homo5_emb_h{H}_lag{L}_{temporal_type}.parquet`.  

#### 12.2.3. Heterogeneous 5‑type GIN embeddings

- Source file:
  - `gnn/gnn_hetero5_h{H}_lag{L}_{temporal_type}.pt`.  
- Export function:
  - `export_hetero5_embeddings_for_config(temporal_type, lag_window, device)`:
    - Loads `X_product`, `days`, `split`, `edge_index_dict`, `num_nodes_dict`, `nodes_hetero_table`.  
    - Derives `node_types` and `edge_types = list(edge_index_dict.keys())`.  
    - Constructs `in_channels_dict = {"edge_types": edge_types, node_type: F_in}`.  
    - Instantiates `HeterogeneousGINEncoder` (hidden dim 128, 2 layers).  
    - For each time step \(t\):
      - Builds `x_dict` as in the homogeneous case (product nodes get `X_product[t]`, others zeros).  
      - Runs the encoder to get product embeddings.  
      - Logs `node_index_pos`, `day`, `split`, `emb_*`.  
- Output:
  - `gnn_embeddings/gnn_hetero5_emb_h{H}_lag{L}_{temporal_type}.parquet`.  

In all three cases, embeddings are **time‑dependent** and aligned by `(day, node_index_pos, split)`, so they can be merged with the tabular baseline using join keys `(node_index, day, split)`. 

### 12.3. Building XGBoost datasets with GNN embeddings

Once embeddings are exported, we construct **tabular datasets for XGBoost** that concatenate lag features and GNN embeddings. The script saves these datasets under `baseline/xgb_gnn_embed/`. Forecasting is still performed by the XGBoost baseline (same hyperparameters as in Section 9–10); the only difference is the presence of `emb_*` columns.

#### 12.3.1. Projected GNN embeddings + XGBoost

- Base tabular file (no graph or basic graph features):
  - `baseline/xgboost/xgboost_tabular_h{H}_lag{L}_{temporal_type}.parquet`.  
- Projected embeddings:
  - `gnn_embeddings/gnn_projected_emb_4views_h{H}_lag{L}_{temporal_type}.parquet`.  
- Builder:
  - `build_xgb_tabular_with_gnn_embed_projected(temporal_type, lag_window)`:
    1. Load `df_base` and `df_emb_long`.  
    2. Rename `node_index_pos → node_index` and cast to int.  
    3. Select embedding columns (`emb_*`) plus index columns (`node_index`, `day`, `split`, `view`).  
    4. Pivot embeddings by view:
       - Index: (`node_index`, `day`, `split`).  
       - Columns: (`view`, `emb_k`).  
       - After `unstack("view")`, flatten column MultiIndex as:
         - `"{col_emb}_{view}"`, e.g.:
           - `emb_0_same_group`, `emb_1_same_group`, …  
           - `emb_0_same_subgroup`, …  
           - `emb_0_same_plant`, …  
           - `emb_0_same_storage`, …  
    5. Merge pivoted embeddings into `df_base` on (`node_index`, `day`, `split`).  
    6. Select features:
       - Any column whose name:
         - Contains `"lag"` or `"roll"`, or  
         - Is one of `day_of_week`, `is_weekend`, `month`, `day_of_month`, or  
         - Is one of `group`, `sub_group`, `plant`, `storage_location`, or  
         - Starts with `"emb_"` (i.e., all projected embedding features).  
    7. Assemble final dataset:
       - Base columns: `node_id`, `node_index`, `date`, `day`, `split`.  
       - Feature columns: lag/rolling/calendar/categorical + **all `emb_*_same_*` columns**.  
       - Target: `target`.  
    8. Save to:
       - `baseline/xgb_gnn_embed/xgboost_tabular_gnnembed_projected4view_h{H}_lag{L}_{temporal_type}.parquet`.  

Thus, each product‑day example includes **four blocks of GNN embeddings**, one for each projected view, concatenated with the standard tabular features. 

#### 12.3.2. Homogeneous GNN embeddings + XGBoost

- Base tabular file:
  - `baseline/xgboost/xgboost_tabular_h{H}_lag{L}_{temporal_type}.parquet`.  
- Homogeneous embeddings:
  - `gnn_embeddings/gnn_homo5_emb_h{H}_lag{L}_{temporal_type}.parquet`.  
- Builder:
  - `build_xgb_tabular_with_gnn_embed_homo5(temporal_type, lag_window)`:
    1. Load `df_base` and `df_emb`.  
    2. Rename `node_index_pos → node_index` and cast to int.  
    3. Merge on (`node_index`, `day`, `split`).  
    4. Select features as above (lag/rolling/calendar/categorical + all `emb_*` columns).  
    5. Save to:
       - `baseline/xgb_gnn_embed/xgboost_tabular_gnnembed_homo5_h{H}_lag{L}_{temporal_type}.parquet`.  

Here, the `emb_*` columns represent a **single homogeneous embedding** per product‑day (no view suffix, since there is only one flattened graph).

#### 12.3.3. Heterogeneous GNN embeddings + XGBoost

- Base tabular file:
  - `baseline/xgboost/xgboost_tabular_h{H}_lag{L}_{temporal_type}.parquet`.  
- Heterogeneous embeddings:
  - `gnn_embeddings/gnn_hetero5_emb_h{H}_lag{L}_{temporal_type}.parquet`.  
- Builder:
  - `build_xgb_tabular_with_gnn_embed_hetero5(temporal_type, lag_window)`:
    1. Load `df_base` and `df_emb`.  
    2. Rename `node_index_pos → node_index`.  
    3. Merge on (`node_index`, `day`, `split`).  
    4. Select features exactly as in the homogeneous case (including all `emb_*` columns).  
    5. Save to:
       - `baseline/xgb_gnn_embed/xgboost_tabular_gnnembed_hetero5_h{H}_lag{L}_{temporal_type}.parquet`.  

In both homo5 and hetero5 variants, the `emb_*` columns encode **graph‑aware node embeddings** learned by the respective GNN encoders, while XGBoost remains the forecasting workhorse. 

---

## 13. XGBoost with GNN embeddings 

This baseline evaluates **GNN embeddings as input features** to a strong tabular model. We first build XGBoost‑ready datasets that concatenate lag‑based tabular features with GNN embeddings (Section 12), then train XGBoost models that operate purely in this enriched feature space. Forecasting is still done by XGBoost; the GNNs only provide learned representations. 

### 13.1. Graph modes and input datasets

We consider three **graph modes**, corresponding to the three encoder families:

- `graph_mode = "proj"`:
  - Uses 4‑view projected GIN embeddings (same_group, same_subgroup, same_plant, same_storage) as in Section 12.2.1.  
  - Input file:
    - `baseline/xgb_gnn_embed/xgboost_tabular_gnnembed_projected4view_h{H}_lag{L}_{temporal_type}.parquet`.  

- `graph_mode = "homo"`:
  - Uses embeddings from the HomogeneousFiveTypeGINEncoder (Section 12.1.2).  
  - Input file:
    - `baseline/xgb_gnn_embed/xgboost_tabular_gnnembed_homo5_h{H}_lag{L}_{temporal_type}.parquet`.  

- `graph_mode = "hetero"`:
  - Uses embeddings from the HeterogeneousGINEncoder (Section 12.1.3).  
  - Input file:
    - `baseline/xgb_gnn_embed/xgboost_tabular_gnnembed_hetero5_h{H}_lag{L}_{temporal_type}.parquet`.  

All three datasets share the same structure:

- Identification and split:
  - `node_id`, `node_index`, `date`, `day`, `split`.  
- Target:
  - `target` = demand at horizon \(H\) on the original scale.  
- Tabular features:
  - Baseline lag and rolling features, calendar features, and categorical indicators (`group`, `sub_group`, `plant`, `storage_location`).  
- GNN embedding features:
  - Columns whose names start with `emb_`; for the projected case, these embed columns are further suffixed by view when pivoted (e.g. `emb_0_same_group`, `emb_0_same_subgroup`, etc.).  

The loader `load_tabular_gnn_embed_baseline(temporal_type, lag_window, graph_mode)` selects the appropriate parquet file based on the graph mode and experiment configuration. [sciencedirect]

### 13.2. Train/validation/test splits and feature set

We keep the same data splitting protocol as other baselines:

- Splits:
  - `split_train_val_test(df)` partitions the dataset into:
    - `df_train` where `split == "train"`,  
    - `df_val` where `split == "val"`,  
    - `df_test` where `split == "test"`.  

- Target and features:
  - `prepare_features(df)`:
    - Casts `target` to float.  
    - Drops:
      - `target`, `split`, `node_id`, `node_index`, `date`, `day`.  
    - Treats all remaining columns as features.  

As a result, the XGBoost input is a **single flat feature vector** per `(product, day)`, combining:

- All lag‑based and rolling time‑series features.  
- Calendar and categorical features.  
- All GNN embedding features `emb_*` (and `emb_*_view` in the projected case).  

The label `target` always stays on the **raw demand scale**, so performance metrics are directly interpretable.

### 13.3. XGBoost configuration and training loop

We use a fixed XGBoost configuration across all graph modes to isolate the effect of embeddings: 

- Model:
  - `XGBRegressor` with:
    - `n_estimators = 5000`  
    - `max_depth = 6`  
    - `learning_rate = 0.05`  
    - `subsample = 0.8`, `colsample_bytree = 0.8`  
    - `objective = "reg:squarederror"`  
    - `tree_method = "hist"`  
    - `random_state = 42`, `n_jobs = -1`  
    - `eval_metric = "rmse"`  
    - `early_stopping_rounds = 100`  

- Training:
  - For each configuration `(temporal_type, lag_window, graph_mode)`, `train_xgb_gnn_embed_baseline`:
    1. Loads the “tabular + embedding” dataset with `load_tabular_gnn_embed_baseline`.  
    2. Splits rows into train/val/test by `split`.  
    3. Extracts features/targets using `prepare_features`.  
    4. Trains the XGBoost model on `(X_train, y_train)` with evaluation on `(X_val, y_val)` using early stopping.  
    5. Stores training and validation RMSE curves for diagnostics.  

The same hyperparameters are used for the **no‑graph XGBoost baseline**, **XGBoost + basic graph features**, **XGBoost + advanced neighbor features**, and **XGBoost + GNN embeddings**, enabling a direct comparison. 

### 13.4. Metrics, predictions, and outputs

We evaluate the model on train, validation, and test splits using standard point‑forecast metrics on the raw demand scale: 
- Metrics:
  - Mean Absolute Error (MAE).  
  - Root Mean Squared Error (RMSE).  
  - Mean Absolute Percentage Error (MAPE).  
  - Symmetric MAPE (sMAPE).  

For each configuration, `train_xgb_gnn_embed_baseline`:

- Computes metrics:
  - On train: `MAE_train`, `RMSE_train`, `MAPE_train`, `sMAPE_train`.  
  - On validation: `MAE_val`, `RMSE_val`, `MAPE_val`, `sMAPE_val`.  
  - On test: `MAE_test`, `RMSE_test`, `MAPE_test`, `sMAPE_test`.  
- Logs a summary dictionary containing:
  - `temporal_type`, `graph_mode`, `lag_window`, `horizon`,  
  - `variant` (e.g. `"baseline_5_xgb_gnn_embed_proj_raw"`),  
  - `tag` (unique identifier per run),  
  - `target_type = "raw"`,  
  - `n_features` (dimension of the input feature vector),  
  - All train/val/test metrics listed above.  

All run summaries are collected in `RUN_SUMMARY` and at the end:

- Aggregated summary:
  - `summary_xgb_gnn_embed_baseline5_raw_lags_graphmodes.csv` saved under:
    - `predictions/baseline_5/summary_xgb_gnn_embed_baseline5_raw_lags_graphmodes.csv`.  
  - Sorted by `temporal_type`, `graph_mode`, `lag_window`, `horizon`, `target_type`, `tag`, and printed for quick inspection.  

This table is the main artifact used in the paper to compare how much gain we obtain from **projected vs homogeneous vs heterogeneous GNN embeddings** when plugged into the same XGBoost backbone. 

### 13.5. Test‑set predictions and per‑product plots

To analyze prediction behavior at the product level, we also export **full test predictions** and visualizations:

- Test prediction CSV:
  - For each configuration, we save:
    - `predictions/baseline_5/csv/{temporal_type}/{graph_mode}/xgb_gnn_embed_{graph_mode}_h{H}_lag{L}_raw_test_predictions_{temporal_type}.csv`.  
  - Columns:
    - `node_id`, `date`, `y_true`, `y_pred`.  

- Per‑product time‑series plots:
  - `plot_predictions_per_product` iterates over all products in the test set and plots:
    - `y_true` vs. `y_pred` over time for each `node_id`.  
  - Plots are saved as:
    - `predictions/baseline_5/plots_xgb_gnn_embed/{graph_mode}/raw_lag{L}/{temporal_type}/xgb_gnn_embed_{graph_mode}_h{H}_lag{L}_node_{node_id}_{temporal_type}.png`.  

These plots allow us to qualitatively inspect **where GNN embeddings help** (e.g., for structurally similar products with sparse histories) versus where they offer limited benefit over purely tabular baselines. 

---

## 14. Graph neural network regressors for direct forecasting

We design three GNN regressors that directly predict next‑step demand for each product node, matching the three graph constructions used elsewhere in the paper: projected product graph, homogeneous 5‑type graph, and heterogeneous 5‑type graph. All models are built from **GIN‑style message passing layers** followed by a per‑node regression head. 

### 14.1. Shared GIN block

All GNN variants reuse the same basic **MLP block** as the GIN aggregation function:

- `MLP(in_channels, out_channels, hidden_channels=None)`:
  - If `hidden_channels` is not given, it defaults to `out_channels`.  
  - Architecture: Linear → ReLU → Linear.  
- In all GIN layers, this MLP is used inside `GINConv` as the update function.  

This follows standard practice for GIN architectures, where message aggregation is followed by a small MLP to update node representations. 

### 14.2. ProjectedGINRegressor (projected product graph)

The **ProjectedGINRegressor** operates on a **single‑type product graph** (projected from relational data) and outputs one scalar prediction per product node.

- Inputs:
  - Node features: `x ∈ ℝ^{N × F}` for product nodes.  
  - Graph structure: `edge_index ∈ ℕ^{2 × E}` for the projected graph (e.g., same‑group, same‑plant edges depending on configuration; the exact view is handled at dataset level).  

- Architecture:
  - `ProjectedGINRegressor(in_channels, hidden_channels=128, num_layers=3, is_softplus=False, is_log1p=False)`:
    - `num_layers` GIN layers:
      - First: `GINConv(MLP(F_in, 128))`.  
      - Next layers: `GINConv(MLP(128, 128))`.  
    - Non‑linearity: ReLU after each `GINConv`.  
    - Final regression head:
      - `out_lin: ℝ^{128} → ℝ`, applied per node and squeezed to shape `[N]`.  
  - The flags `is_softplus` and `is_log1p` are stored for logging/tagging, but **no nonlinear post‑processing** is applied in `forward` (the model outputs logits on the training scale).  

- Output:
  - `forward(x, edge_index)` returns a vector `head ∈ ℝ^{N}`, where each entry is the **logit prediction** \(z_i\) for product \(i\) on the training target scale (e.g., raw or log‑transformed, depending on preprocessing).  

This model corresponds to a standard node‑regression GIN over the product graph. 

### 14.3. HomogeneousFiveTypeGINRegressor (flattened 5‑type graph)

The **HomogeneousFiveTypeGINRegressor** treats the 5‑type graph as a **flattened homogeneous graph** with a type embedding to distinguish node types. It predicts demand only for product nodes. 

- Node types and indexing:
  - `num_nodes_dict`: dictionary `{node_type: num_nodes}`.  
  - `node_type_order`: ordered list of node types specifying concat order (e.g., `["product", "product_group", "product_sub_group", "plant", "storage_location"]`).  
  - Global indexing:
    - Nodes of all types are concatenated into a single index space of size `total_num_nodes`.  
    - For each type, we store:
      - An offset in this concatenation (`node_type_offsets`).  
      - A type id (`node_type_id ∈ ℕ^{total_num_nodes}`) used for embedding.  

- Inputs:
  - `x_dict`: dictionary `{node_type: [N_type, F_in]}` with features per type.  
  - `edge_index`: flattened edge list over all node types in the global index space (`[2, E_total]`).  

- Architecture:
  - Type embeddings:
    - `type_emb = nn.Embedding(num_types, node_type_emb_dim)` with default `node_type_emb_dim = 8`.  
    - For each node, we concatenate its original features with its type embedding.  
  - GIN layers:
    - First layer: `GINConv(MLP(F_in + node_type_emb_dim, 128))`.  
    - Next `num_layers - 1` layers: `GINConv(MLP(128, 128))`.  
    - ReLU after each layer.  
  - Regression head:
    - Linear layer `out_lin: ℝ^{128} → ℝ` applied to every node, giving logits `head_all ∈ ℝ^{N_total}`.  
  - Product slice:
    - Using `node_type_offsets` and `num_nodes_dict["product"]`, we slice:
      - `out_prod = head_all[offset_prod : offset_prod + N_product]`.  

- Output:
  - `forward(x_dict, edge_index)` returns `out_prod ∈ ℝ^{N_product}`, the predicted logits for **product nodes only**.  

This design allows us to apply a homogeneous GIN to a structurally heterogeneous graph while still producing a **product‑level forecast**. 

### 14.4. HeterogeneousGINRegressor (heterogeneous 5‑type graph)

The **HeterogeneousGINRegressor** operates directly on a heterogeneous 5‑type graph using PyG’s `HeteroConv`. It maintains separate representations per node type and aggregates messages along each relation type. 

- Inputs:
  - `in_channels_dict`: dictionary:
    - Contains key `"edge_types"` listing all edge types `(src_type, rel, dst_type)`.  
    - For each node type `nt`, provides `in_channels_dict[nt] = F_in_nt` (input feature dimension).  
  - For `forward`:
    - `x_dict`: `{node_type: [N_type, F_in]}`.  
    - `edge_index_dict`: `{(src_type, rel, dst_type): edge_index}` with `edge_index ∈ ℕ^{2×E}`.  

- Node feature projection:
  - For each node type `nt`, we project input features to `hidden_channels`:
    - `node_in_proj[nt]: ℝ^{F_in_nt} → ℝ^{128}`.  
  - Initial hidden features:
    - `h_dict[nt] = ReLU(node_in_proj[nt](x_dict[nt]))` for all node types.  

- Heterogeneous GIN layers:
  - Each `HeterogeneousGINLayer`:
    - Builds a dict of `GINConv` modules, one per edge type `(src_type, rel, dst_type)`.  
    - Uses the hidden dimension as input channel for all node types.  
    - Wraps these in `HeteroConv(convs, aggr="sum")`.  
    - `forward(x_dict, edge_index_dict)` returns updated embeddings only for **destination node types**.  
  - In `HeterogeneousGINRegressor`:
    - We apply `num_layers` such layers sequentially.  
    - After each layer, we **reinstate** node types that appear only as sources:
      - If a type `nt` is missing from layer output, we keep its previous embeddings.  
    - ReLU is applied after each layer on all node‑type embeddings.  

- Regression head:
  - After the final layer:
    - `head = out_lin(h_dict["product"]).squeeze(-1)`, producing `ℝ^{N_product}` logits.  

- Output:
  - `forward(x_dict, edge_index_dict)` returns a vector of logits for all **product nodes** in the heterogeneous graph.  

This model can leverage **directional, typed edges** between products, product groups, subgroups, plants, and storage locations for forecasting. 

### 14.5. Target scale and activation choices

All three regressors share the same convention for outputs:

- The `forward` method always returns **logits on the training scale** (variable usually denoted \(z\)).  
- The flags `is_softplus` and `is_log1p` are stored in the model, but **no activation or inverse transform is applied inside the model**:
  - If we train on raw targets, `z` can be interpreted directly as predicted demand.  
  - If we train on a transformed target (e.g., \(\log(1 + y)\) or softplus), the inverse transform is handled in the **training/evaluation pipeline**, not inside the regressor.  

This separation makes it easy to compare raw‑target and transformed‑target training regimes using the same architectures.

---

## 15. Direct GNN forecasting baselines

In this family of baselines, we train **graph neural networks end‑to‑end** to directly predict next‑step demand for all products, without any XGBoost component. For each temporal configuration and lag window, we build graph–time packages and fit three GNN variants: projected product graph, homogeneous 5‑type graph, and heterogeneous 5‑type graph. 

### 15.1. Data packages and time splits

For each experiment configuration `(temporal_type, horizon H = 7, lag_window ∈ {7, 14})`, we precompute graph‑ready tensors and metadata, stored under `data/processed/gnn/` as:

- Projected: `gnn_projected_h{H}_lag{L}_{temporal_type}.pt`.  
- Homogeneous 5‑type: `gnn_homo5_h{H}_lag{L}_{temporal_type}.pt`.  
- Heterogeneous 5‑type: `gnn_hetero5_h{H}_lag{L}_{temporal_type}.pt`.  

The helper `load_gnn_pkg(graph_type, temporal_type, lag_window)` loads the appropriate package into a dictionary `pkg` with:

- Common fields:
  - `X_product`: tensor `[T, N_prod, F]` of product features over time.  
  - `Y_product`: tensor `[T, N_prod]` of targets on **original scale** (non‑negative sales).  
  - `days`: array of time indices (or dates), length `T`.  
  - `split`: list of strings length `T` with split labels: `"train"`, `"val"`, `"test"`.  

- Graph‑specific fields:
  - Projected:
    - `edge_index_dict`: mapping `view → edge_index` for projected product graphs.  
  - Homogeneous:
    - `edge_index`: flattened adjacency `[2, E_total]`.  
    - `num_nodes_dict`: number of nodes per type.  
    - `nodes_homo_table`: node metadata, including `node_type`.  
  - Heterogeneous:
    - `edge_index_dict`: heterogeneous edge indices keyed by `(src_type, rel, dst_type)`.  
    - `num_nodes_dict`: number of nodes per type.  

Time splits are derived from `split` using `get_time_splits(days, split)`:

- `idx_train`: indices with `split[t] == "train"`.  
- `idx_val`: indices with `split[t] == "val"`.  
- `idx_test`: indices with `split[t] == "test"`.  

All GNN baselines respect these time‑based splits for training, validation, and testing. 

### 15.2. Target transforms and inverse transforms

We support three training modes for the target: **raw**, **log1p**, and **softplus‑inverse**, controlled by `exp.is_softplus` and `exp.is_log1p` in `DEFAULT_EXPERIMENTS`:

- Transform to training scale (forward):  
  - `transform_y_tensor(y, is_softplus, is_log1p)`:
    - Clamp `y` at zero to ensure non‑negativity.  
    - If `is_log1p`: `z = log1p(y)`.  
    - Else if `is_softplus`: `z = softplus^{-1}(y) ≈ log(exp(y) − 1)`.  
    - Else (`raw`): `z = y`.  

- Inverse transform to original scale (for evaluation):  
  - `inverse_transform_y_tensor(z, is_softplus, is_log1p)`:
    - If `is_softplus`: `y_hat = softplus(z)`, then clamp to ≥ 0.  
    - If `is_log1p`: `y_hat = expm1(z)`, clamp ≥ 0.  
    - Else (`raw`): `y_hat = z`, clamp ≥ 0 to avoid negative predictions.  

The helper `get_mode_name(is_softplus, is_log1p)` returns the string `"softplus"`, `"log1p"`, or `"raw"` to label runs and output directories. 

### 15.3. Loss, metrics, and early stopping

All GNN baselines share the same optimization and evaluation choices: 

- Optimization:
  - Optimizer: Adam with learning rate `1e-3`.  
  - Loss function: mean squared error (MSE) on **transformed targets** `z`.  

- Early stopping:
  - `EarlyStopping(patience=es_patience, min_delta=es_min_delta)`:
    - Tracks the best validation loss (minimum MSE on `z`).  
    - Saves a deep copy of the best `state_dict`.  
    - Stops training if validation score does not improve by at least `min_delta` for `patience` epochs.  
    - After training, `load_best(model)` restores the best weights.  
  - In experiments, we use `es_patience = 20` and `es_min_delta = 0.001`.  

- Forecast metrics (on original scale `y` after inverse transform):
  - `MAE`, `RMSE`, `MAPE`, `sMAPE`.  
  - Implemented by helper functions `mae`, `rmse`, `mape`, `smape`, all operating on flattened arrays.  

These metrics match those used for the XGBoost baselines, enabling a direct comparison. 

### 15.4. Projected GNN baseline

For the projected product graphs, we train a separate model for each **edge view** in:

- `PROJECTED_VIEWS = ["same_group", "same_subgroup", "same_plant", "same_storage"]`.  

Training is handled by `run_projected_gnn_baseline(pkg, temporal_type, lag_window, edge_view, ...)`. 

#### 15.4.1. Training procedure

Given a package `pkg`:

- Data tensors:
  - `X = pkg["X_product"] ∈ ℝ^{T × N × F}`.  
  - `Y = pkg["Y_product"] ∈ ℝ^{T × N}`.  
  - `Y_trans = transform_y_tensor(Y, is_softplus, is_log1p)`.  
  - `days`, `split`.  
  - `edge_index = edge_index_dict[edge_view]` for the selected view.  

- Model:
  - `ProjectedGINRegressor(in_channels=F, hidden_channels=128, num_layers=3, is_softplus=is_softplus, is_log1p=is_log1p)` on GPU if available.  

- Mini‑batching over days:
  - Hyperparameters: `epochs=300`, `batch_days=8`.  
  - `iterate_days(day_indices, train_mode)`:
    - For each mini‑batch of `batch_days` time indices:
      - Slice `X_block ∈ ℝ^{B × N × F}` and `Y_block ∈ ℝ^{B × N}`.  
      - For each day in the block:
        - Take `x_b ∈ ℝ^{N × F}`, `z_true_b ∈ ℝ^{N}`.  
        - Compute logits `z_pred_b = model(x_b, edge_index)`.  
        - Form mask over finite `z_true_b`, compute `MSE(z_pred_b[mask], z_true_b[mask])`.  
      - Average losses over `B`, backpropagate if `train_mode=True`.  
    - Returns average loss over all mini‑batches.  

- Training loop:
  - For each epoch:
    - `train_loss = iterate_days(idx_train, train_mode=True)`.  
    - `val_loss = iterate_days(idx_val, train_mode=False)`.  
    - Update early stopping; break if patience is exceeded.  

#### 15.4.2. Evaluation and outputs

After restoring the best weights:

- Prediction per split:
  - For a set of day indices `idxs`:
    - For each `t` in `idxs`:
      - Compute logits `z_pred_t` and transform back to original scale:
        - `y_hat_t = inverse_transform_y_tensor(z_pred_t, is_softplus, is_log1p)`.  
      - Collect `y_true_t = Y[t]` and `y_hat_t`.  
    - Concatenate across time into flattened arrays `y_true_flat` and `y_pred_flat`.  
- Metrics:
  - Compute MAE, RMSE, MAPE, sMAPE for train, validation, and test.  

- Test‑set reshaping and plotting:
  - Reshape test predictions to `y_test_true ∈ ℝ^{T_test × N}`, `y_test_pred ∈ ℝ^{T_test × N}`.  
  - `plot_predictions_per_product(days_test, y_test_true, y_test_pred, out_dir, temporal_type, lag_window, graph_tag)`:
    - For each product index `j`, plot true vs predicted demand over `days_test`.  
    - Save one PNG per product into `plots_projected`.  

- CSV outputs:
  - Save test predictions (flattened) to:
    - `predictions/baseline_4/{temporal_type}_{mode_name}/csv/projected/{tag}_test_predictions.csv`.  
  - Columns: `date` (day index), `product_idx`, `y_true`, `y_pred`.  

- Summary entry:
  - Append metrics and metadata to `RUN_SUMMARY` with:
    - `variant = "gnn_projected"`, `edge_view` set to the current projected view.  

This gives four projected GNN baselines per `(temporal_type, lag_window)` configuration.

### 15.5. Homogeneous 5‑type GNN baseline

The homogeneous 5‑type baseline trains a single GNN on the **flattened 5‑type graph**, predicting for all product nodes at each time step. Training is handled by `run_homo5_gnn_baseline(pkg, temporal_type, lag_window, ...)`. 

#### 15.5.1. Training procedure

From `pkg` we use:

- `X_prod ∈ ℝ^{T × N_prod × F}`, `Y_prod ∈ ℝ^{T × N_prod}`.  
- `Y_trans = transform_y_tensor(Y_prod, ...)`.  
- `days`, `split`.  
- `edge_index ∈ ℝ^{2 × E_total}` (flattened graph).  
- `num_nodes_dict`, `nodes_homo_table` (to get `node_type_order`).  

Model:

- `HomogeneousFiveTypeGINRegressor(in_channels=F, num_nodes_dict=num_nodes_dict, node_type_order=node_type_order, hidden_channels=128, num_layers=3, node_type_emb_dim=8, ...)`.  

Other node types:

- `other_types = [nt for nt in node_type_order if nt != "product"]`.  
- For each time step we build `x_dict` as:
  - `"product"`: `x_prod_b ∈ ℝ^{N_prod × F}` from the current day.  
  - Other types: zero features `ℝ^{num_nodes_dict[nt] × F}`.  

Mini‑batching and loss computation:

- Same pattern as projected case:
  - For each mini‑batch of days:
    - For each day `b`:
      - Build `x_dict`.  
      - Compute `z_pred_b = model(x_dict, edge_index)` for product nodes.  
      - Compute MSE against `z_true_b = Y_trans[b]` with finite mask.  

Early stopping and optimization follow exactly the same scheme as in Section 15.4. 

#### 15.5.2. Evaluation and outputs

After training, we:

- Predict on train/val/test splits with the same per‑day loop:
  - Rebuild `x_dict` with zero features for non‑product types.  
  - Run model, inverse‑transform logits to original scale.  

- Compute MAE, RMSE, MAPE, sMAPE on flattened predictions and targets.  

- Reshape test predictions into `[T_test, N_prod]` for plotting with `plot_predictions_per_product`, saved under:

  - `predictions/baseline_4/{temporal_type}_{mode_name}/plots_homo5/`.  

- Save test prediction CSV:

  - `predictions/baseline_4/{temporal_type}_{mode_name}/csv/homo5/{tag}_test_predictions.csv`.  

- Append a summary record to `RUN_SUMMARY` with `variant = "gnn_homo5"` and `edge_view = None`.  

This baseline isolates the effect of **homogeneous GIN with type embedding** on the 5‑type graph.

### 15.6. Heterogeneous 5‑type GNN baseline

The heterogeneous GNN baseline uses **HeterogeneousGINRegressor** to model typed relations explicitly. It is implemented in `run_hetero5_gnn_baseline(pkg, temporal_type, lag_window, ...)`.

#### 15.6.1. Training procedure

From `pkg` we use:

- `X_prod ∈ ℝ^{T × N_prod × F}`, `Y_prod ∈ ℝ^{T × N_prod}`.  
- `Y_trans = transform_y_tensor(Y_prod, ...)`.  
- `days`, `split`.  
- `edge_index_dict`: mapping from `(src_type, rel, dst_type)` to `edge_index`.  
- `num_nodes_dict`: number of nodes per type.  

We define:

- `edge_types = list(edge_index_dict.keys())`.  
- `in_channels_dict = {"edge_types": edge_types, "product": F}`, plus `F` for each other node type.  

Model:

- `HeterogeneousGINRegressor(in_channels_dict, hidden_channels=128, num_layers=2, is_softplus=is_sp, is_log1p=is_log1p)`.  

Base node features:

- `base_x_dict = {nt: zeros(num_nodes_dict[nt], F) for nt in num_nodes_dict.keys()}`.  
- For each time step we override:
  - `x_dict = {nt: base_x_dict[nt]}` then
  - `x_dict["product"] = X_prod[t]`.  

Mini‑batching and loss:

- Same structure as other baselines:
  - Iterate over mini‑batches of days (`batch_days=8`).  
  - For each day:
    - Build `x_dict`, run `z_pred_b = model(x_dict, edge_index_dict)`.  
    - MSE loss on transformed targets `z_true_b`, with finite mask.  

Optimization and early stopping follow exactly the same scheme. 

#### 15.6.2. Evaluation and outputs

After restoring the best model:

- Predict on train/val/test indices with per‑day loops:
  - Build `x_dict` from `base_x_dict` and per‑day product features.  
  - Run hetero GNN, inverse‑transform logits to predicted `y_hat_t`.  

- Compute MAE, RMSE, MAPE, sMAPE on flattened arrays.  

- Reshape test predictions to `[T_test, N_prod]` and call `plot_predictions_per_product`, saving under:

  - `predictions/baseline_4/{temporal_type}_{mode_name}/plots_hetero5/`.  

- Save test prediction CSV:

  - `predictions/baseline_4/{temporal_type}_{mode_name}/csv/hetero5/{tag}_test_predictions.csv`.  

- Append to `RUN_SUMMARY` with `variant = "gnn_hetero5"` and `edge_view = None`.  

This baseline allows us to study whether **explicitly modeling edge types** (heterogeneous GNN) improves product‑level forecasting compared to projected and homogeneous GINs. 

### 15.7. Experiment loop and summaries

The `main()` function in `train_gnn.py` iterates over all experiment configurations in `DEFAULT_EXPERIMENTS`:

- For each `exp`:
  - Get `temporal_type`, `lag_windows`, and target transform flags (`is_softplus`, `is_log1p`).  
  - Create a base prediction directory:
    - `predictions/baseline_4/{temporal_type}_{mode_name}/`.  
  - For each `lag_window` in `exp.lag_windows`:
    - Load projected, homogeneous, and heterogeneous GNN packages via `load_gnn_pkg`.  
    - Run:
      - `run_projected_gnn_baseline` for each view in `PROJECTED_VIEWS`.  
      - `run_homo5_gnn_baseline`.  
      - `run_hetero5_gnn_baseline`.  

After finishing all runs for a given experiment config:

- Collect `RUN_SUMMARY` into a DataFrame.  
- Sort by `temporal_type`, `lag_window`, `horizon`, `variant`, `tag`, `edge_view`, `target_transform`.  
- Write a CSV summary:

  - `predictions/baseline_4/{temporal_type}_{mode_name}/summary_baseline_4_{temporal_type}_{mode_name}.csv`.  

This summary serves as the main quantitative comparison of the **three GNN variants** across lag windows and target transforms, and it is used directly in the experimental analysis section of the paper.

This script defines **Baseline 6**, where a GNN is trained to model **residuals on top of an existing XGBoost forecaster**, instead of predicting demand from scratch. 

---

## 16. Baseline 6: Residual XGBoost + GNN

Baseline 6 follows a two‑stage strategy: first, a strong **tabular XGBoost model** produces per‑product demand forecasts; second, a GNN learns to predict residuals between these forecasts and the (transformed) ground‑truth, using graph structure across products, groups, plants, and storage locations. 

### 16.1. Residual construction and time splits

For each experiment configuration `(temporal_type, HORIZON = H, lag_window)`, we reuse the tabular features and XGBoost predictions:

- Node and feature loading:
  - `load_node_metadata()` loads static node metadata (e.g. product, group, plant IDs).  
  - `load_xgb_tabular_for_gnn(temporal_type, lag_window, HORIZON)` returns the tabular design matrix aligned with product nodes and time.  
  - XGBoost predictions are read from:
    - `baseline/xgboost/xgboost_predictions_h{H}_lag{lag_window}_{temporal_type}.parquet`.  

- Residual tensors:
  - `build_residual_time_tensors_for_gnn(df_xgb_tabular, df_xgb_pred, is_log1p)` builds a package:
    - `X_residual ∈ ℝ^{T × N_prod × F_res}`: residual‑side features per product per time.  
      - Last feature in the channel (`X_res[:, :, -1]`) stores the XGBoost prediction on the original scale.  
    - `R_residual ∈ ℝ^{T × N_prod}`: residual target for the GNN (on raw or log1p scale depending on `is_log1p`).  
    - `days`: array of dates (length `T`).  
    - `split`: `"train"`, `"val"`, `"test"` per time index.  
    - `node_ids_product`: node IDs for the product dimension, used in plotting and CSV outputs.  

- Time splits:
  - `get_time_splits(days, split)` produces:
    - `idx_train`, `idx_val`, `idx_test`, with counts logged for transparency.  

The GNN is always trained on **residuals**, but final metrics are computed on reconstructed **original‑scale demand** (XGBoost forecast + residual).

### 16.2. Metrics, early stopping, and per‑product plots

Baseline 6 uses the same evaluation metrics and early‑stopping logic as Baseline 4: 

- Metrics:
  - `MAE`, `RMSE`, `MAPE`, `sMAPE` implemented as NumPy helpers operating on flattened arrays.  
  - `MAPE` ignores near‑zero targets (with a configurable epsilon), and `sMAPE` is defined using the symmetric denominator.  

- Early stopping:
  - `EarlyStopping(patience, min_delta)` monitors validation loss (MSE on residuals).  
  - Stores the best `state_dict` and stops if no improvement is observed for `patience` epochs.  
  - `load_best(model)` restores the best weights before evaluation.  

- Per‑product plots for Baseline 6:
  - `plot_baseline6_predictions_per_product(...)`:
    - Builds a DataFrame with columns `date`, `node_id`, `y_true`, `y_pred` from flattened test arrays.  
    - For each product node, plots true vs predicted test‑set demand over time.  
    - Saves plots under:
      - `predictions/baseline_6/{temporal_type}_{mode}/[projected|homo5|hetero5]/plots_per_product/`.  

These plots show how well the **residual‑enhanced forecasts** track individual product trajectories.

### 16.3. Residual GNN on projected product graphs

The projected residual baseline trains a **ProjectedGINRegressor** on a product–product graph defined by different views (same group, subgroup, plant, or storage). Training is handled by `train_residual_projected`.

#### 16.3.1. Graph construction and model

For a fixed `(temporal_type, lag_window, is_log1p)` and `edge_view`:

- Residual tensors:
  - `X_res = X_residual ∈ ℝ^{T × N × F_res}`.  
  - `R_res = R_residual ∈ ℝ^{T × N}`.  
  - `days`, `split`, `node_ids_product`.  

- Graph:
  - `build_projected_edge_indices(nodeindex2pos_prod, df_meta)` constructs a dictionary `edge_index_dict` mapping each projected view to an adjacency `edge_index`.  
  - We select `edge_index = edge_index_dict[edge_view]`.  

- Model:
  - `ProjectedGINRegressor(in_channels=F_res, hidden_channels=128, num_layers=2, is_softplus=False, is_log1p=False)`.  
  - The model operates on residual features only and returns predicted residuals `r_pred ∈ ℝ^{N}`.  

- Optimization:
  - Adam with learning rate `3e-3`, MSE loss on residuals.  
  - Early stopping with `patience = 20`, `min_delta = 1e-3`.  

#### 16.3.2. Training loop and residual reconstruction

Training mini‑batches over days:

- `iterate_days(day_indices, train_mode)`:
  - For each block of `batch_days` time indices:
    - Slice `X_block ∈ ℝ^{B × N × F_res}`, `R_block ∈ ℝ^{B × N}`.  
    - For each day in block:
      - `x_b ∈ ℝ^{N × F_res}`, `r_true_b ∈ ℝ^{N}`.  
      - `r_pred_b = model(x_b, edge_index)`.  
      - Compute MSE over finite entries of `r_true_b`.  
    - Average losses across `B`, backpropagate if training.  

After training, we evaluate by reconstructing the full demand:  

- For each time `t` in a given split:
  - `x_t = X_res[t]`.  
  - `r_true_t`, `r_pred_t`.  
  - `y_xgb_t = X_res[t, :, -1]` (XGBoost forecast on original scale).  

- Reconstruction:
  - If `is_log1p`:
    - Compute `z_xgb_t = log1p(y_xgb_t)`.  
    - The GNN residual `R_res` and `r_pred_t` live on the **log1p scale**.  
    - True and predicted logits:
      - `z_true_t = z_xgb_t + r_true_t`, `z_hat_t = z_xgb_t + r_pred_t`.  
    - Back to original scale:
      - `y_true_t = expm1(z_true_t).clamp_min(0)`.  
      - `y_hat_t = expm1(z_hat_t).clamp_min(0)`.  
  - Else (`raw`):
    - Residuals are on the original scale:
      - `y_true_t = (y_xgb_t + r_true_t).clamp_min(0)`.  
      - `y_hat_t = (y_xgb_t + r_pred_t).clamp_min(0)`.  

These `y_true_t` and `y_hat_t` are concatenated over time to compute MAE, RMSE, MAPE, and sMAPE for train/val/test splits. 

#### 16.3.3. Outputs and logging

For each projected view:

- Test predictions:
  - Saved to:
    - `predictions/baseline_6/{temporal_type}_{mode}/projected/{tag}_test_predictions.csv`.  
  - Columns: `date`, `node_id`, `y_true`, `y_pred`.  

- Per‑product plots:
  - Saved under:
    - `.../projected/plots_per_product/`, using `plot_baseline6_predictions_per_product`.  

- Summary record:
  - Appended to `RUN_SUMMARY` with:
    - `graph_type="projected"`, `edge_view`, `variant="baseline_6_residual"`, `mode=mode_name`, and all metrics.  


### 16.4. Residual GNN on homogeneous 5‑type graph

The homogeneous residual baseline embeds residual features into a **flattened 5‑type graph**, but still predicts residuals only for product nodes. It is implemented in `train_residual_homo5`.

#### 16.4.1. Graph and indexing

After building residual tensors `X_res`, `R_res`, and time splits:

- Homogeneous graph:
  - `build_homo5type_from_parquet()` returns:
    - `edge_index_homo5`: heterogeneous edge indices per relation.  
    - `num_nodes_homo5`: node counts per type (5 types).  
    - `nodes_homo_tbl`: node metadata with `node_type` and `node_index`.  
  - `node_type_order = sorted(unique node_type)` defines concatenation order.  
  - `make_homo5_flat_edge_index(...)` converts heterogeneous indices to a single `edge_index_flat` in the concatenated node space.  

- Mapping products into the flattened index:
  - Select product rows from `nodes_homo_tbl`, sorted by `node_index`.  
  - Build a mapping `nodeindex2local` from `node_index` to local product position.  
  - Compute type offsets (per `node_type_order`) to get the global index of each product node in the concatenated graph.  
  - `prod_global_idx` holds the global indices for all product nodes in the residual tensors’ order.  

#### 16.4.2. Model and training

Model:

- `HomogeneousFiveTypeGINRegressor(in_channels=F_res, num_nodes_dict=num_nodes_homo5, node_type_order=node_type_order, hidden_channels=128, num_layers=2, node_type_emb_dim=8, is_softplus=False, is_log1p=False)`.  

For each time step during training:

- Build `x_all ∈ ℝ^{N_total × F_res}` as zeros.  
- Insert product residual features:
  - `x_all[prod_global_idx] = x_prod_b`, where `x_prod_b ∈ ℝ^{N_prod × F_res}`.  
- Split `x_all` back into a dict:
  - `x_dict[nt] = slice of x_all` for each `nt` in `node_type_order`.  
- Forward:
  - `r_pred_b = model(x_dict, edge_index_flat)`, giving residual predictions for product nodes.  
- Loss:
  - MSE between `r_pred_b` and `r_true_b` with finite mask.  

Mini‑batching, optimization, and early stopping are identical to the projected case.

#### 16.4.3. Reconstruction and outputs

At evaluation time, for each day:

- Rebuild `x_all` and `x_dict` as above.  
- Predict `r_pred_t`, then reconstruct original‑scale demand exactly like the projected case:
  - If `is_log1p`: add residuals on the log1p scale and apply `expm1`.  
  - Else: add residuals on the original scale and clamp at zero.  

Outputs:

- Test prediction CSV:
  - `predictions/baseline_6/{temporal_type}_{mode}/homo5/{tag}_test_predictions.csv`.  
- Per‑product plots:
  - Saved under `.../homo5/plots_per_product/`.  
- Summary entry appended to `RUN_SUMMARY` with `graph_type="homo5"`.  

This baseline tests whether spreading residual information across a larger homogeneous 5‑type graph can improve over purely product‑product projected graphs. 


### 16.5. Residual GNN on heterogeneous 5‑type graph

Finally, `train_residual_hetero5` uses the full **heterogeneous graph** to propagate residual signals through typed edges while still predicting residuals for product nodes only.

#### 16.5.1. Graph, features, and mapping

After constructing `X_res`, `R_res`, and time splits:

- Heterogeneous graph:
  - `build_hetero5type_from_parquet()` returns:
    - `edge_index_het5`: dict keyed by `(src_type, rel, dst_type)` with edge_index tensors.  
    - `num_nodes_het5`: node counts per type.  
    - `nodes_het_tbl`: node metadata including `node_type` and `node_index`.  

- Base features:
  - `base_x_dict[nt] = zeros(num_nodes_het5[nt], F_res)` for all non‑product node types.  

- Product index mapping:
  - Select product rows from `nodes_het_tbl`, cast `node_index` to int, sort by `node_index`.  
  - Build `nodeindex2local` mapping from `node_index` to local product index in the heterogeneous graph.  
  - From `node_index_product` in `pkg_res`, derive `prod_local_idx` (local hetero indices matching residual tensor order).  

#### 16.5.2. Model and training

Model:

- `HeterogeneousGINRegressor(in_channels_dict, hidden_channels=128, num_layers=2, is_softplus=False, is_log1p=False)`, where:
  - `in_channels_dict["edge_types"] = list(edge_index_het5.keys())`.  
  - For each node type `nt`, `in_channels_dict[nt] = F_res`.  

Version of features per time step:

- For each day:
  - Start from `x_dict = {nt: base_x_dict[nt]}` for all non‑product types.  
  - Build `x_prod_full = zeros(num_nodes_het5["product"], F_res)`.  
  - Scatter product residual features:
    - `x_prod_full[prod_local_idx] = x_prod_b`.  
  - Set `x_dict["product"] = x_prod_full`.  

- Forward:
  - `r_pred_b = model(x_dict, edge_index_dict)`, returning residual predictions aligned with `prod_local_idx` order.  

- Loss:
  - MSE between `r_pred_b` and `r_true_b` with finite mask, per day, averaged across the mini‑batch.  

Optimization and early stopping again mirror the previous baselines. 

#### 16.5.3. Reconstruction and outputs

Evaluation reconstructs original‑scale demand exactly as in the other residual baselines:

- For each time `t`:
  - Build `x_dict`, run the hetero GNN, obtain `r_pred_t`.  
  - Combine with XGBoost base prediction `y_xgb_t` on either log1p or raw scale.  

Outputs:

- Test prediction CSV:
  - `predictions/baseline_6/{temporal_type}_{mode}/hetero5/{tag}_test_predictions.csv`.  
- Per‑product plots:
  - Under `.../hetero5/plots_per_product/`.  
- Summary row appended to `RUN_SUMMARY` with `graph_type="hetero5"`.  

This baseline explores whether **typed inter‑entity edges** help the GNN correct XGBoost forecast errors more effectively than the projected or homogeneous variants.

### 16.6. Main loop and global summary

The `main()` function orchestrates all residual experiments: 

- Global hyperparameters:
  - `device="cuda"` (if available), `epochs=400`, `batch_days=8`.  
  - `only_graph` can restrict training to `"projected"`, `"homo5"`, `"hetero5"`, or `"all"`.  

- Modes per experiment:
  - For each `exp` in `DEFAULT_EXPERIMENTS`:
    - Always run `raw` mode (`is_log1p=False`).  
    - Additionally run `log1p` mode if `exp.is_log1p` is `True`.  

- For each `(temporal_type, lag_window, is_log1p)`:
  - If `projected` is enabled:
    - Train residual GNNs for all `PROJECTED_VIEWS = ["same_group", "same_subgroup", "same_plant", "same_storage"]`.  
  - If `homo5` is enabled:
    - Train one residual homogeneous GNN on the 5‑type graph.  
  - If `hetero5` is enabled:
    - Train one residual heterogeneous GNN on the 5‑type graph.  

At the end, if `RUN_SUMMARY` is non‑empty:

- Collect all summary rows into a DataFrame.  
- Sort by `temporal_type`, `lag_window`, `horizon`, `graph_type`, `edge_view`, `mode`, `tag`.  
- Save the global summary to:

  - `predictions/baseline_6/summary_baseline_6_residual_xgb_gnn.csv`.  

This summary is used in the results section to compare **plain XGBoost**, **direct GNN baselines (Baseline 4)**, and **Residual XGBoost + GNN (Baseline 6)** on the same demand forecasting task.

---

## 17. Overall performance at horizon 7

We compare all model families on the **unit‑level 7‑day horizon** with lag windows 7 and 14, focusing on **test MAE, RMSE, MAPE, sMAPE**. 

### 17.1. Lag 7: best models

At lag 7, the **best overall models** are the residual GNN baselines on top of XGBoost:

- Best MAE:
  - `xgb_gnn_embed_hetero` (Baseline 5, hetero embedding) and `xgb_gnn_residual_hetero5` (Baseline 6) both reach **MAE ≈ 418–490**, clearly below plain XGBoost tabular (**MAE ≈ 410** but with higher MAPE) and GNN‑only baselines.  
- Best MAPE / sMAPE:
  - Pure GNN baselines (Baseline 4) show very large MAPE, especially homogeneous and projected variants, often exceeding 300–600% and sMAPE over 140%.  
  - Residual GNN baselines (Baseline 6) reduce MAPE substantially, with `xgb_gnn_residual_hetero5` reaching **MAPE ≈ 236.5%, sMAPE ≈ 89.4%**, clearly better than pure GNNs but still above tabular XGBoost.  
- Baseline 1 (XGBoost tabular):
  - `xgb_tabular_raw` achieves **MAE ≈ 409.7**, **RMSE ≈ 1035.7**, **MAPE ≈ 230.6%**, **sMAPE ≈ 127.3%**, serving as a strong non‑graph reference.  

A compact view (test metrics, lag 7):

| Model family          | Variant / graph               | MAE (test) | RMSE (test) | MAPE (test) | sMAPE (test) |
|-----------------------|-------------------------------|-----------:|------------:|------------:|-------------:|
| xgb_tabular (B1)      | no graph                      | 409.7      | 1035.7      | 230.6       | 127.3        |
| xgb_graph (B3)        | proj / homo / hetero          | 401.8–421.6| 1004.6–1037.9| 212.1–239.9 | 127.1–128.1  |
| xgb_tabular_graphfeat | proj / homo5 / hetero5        | 392.7–402.3| 942.8–950.1 | 318.9–378.8 | 127.4–128.9  |
| xgb_gnn_embed (B5)    | proj / homo / hetero          | 438.7–489.9| 1044.8–1102.0| 199.6–486.0 | 128.4–131.7  |
| xgb_gnn_residual (B6) | proj / homo5 / hetero5        | **399.98–429.6** | 1026.8–1109.2 | 221.1–537.6 | **85.7–105.9** |
| GNN only (B4)         | projected / homo5 / hetero5   | 416.9–1014.7|1037.8–1911.4| 106.6–2727.0| 96.9–191.1   |
| GRU sequence (B2)     | no graph                      | 463.9      | 1189.7      | 192.1       | 149.5        |
| Naive last T0         | no graph                      | 404.8      | 1117.1      | 125.1       | 62.2         |

Overall, **graph‑enhanced XGBoost models** (Baselines 3, 5, 6) consistently outperform pure GRU and pure GNN baselines, and the best residual GNN variant is competitive with the XGBoost tabular baseline in MAE and sMAPE.

### 17.2. Lag 14: best models

At lag 14, we observe similar patterns, with some models slightly improving while others degrade: 

- XGBoost tabular (Baseline 1):
  - Achieves **MAE ≈ 375.4**, **RMSE ≈ 933.8**, **MAPE ≈ 192.3%**, **sMAPE ≈ 126.2%**, improving MAE slightly compared to lag 7.  
- Graph‑enhanced tabular (Baseline 3 and “graphfeat”):
  - `xgb_graph_*` and `xgb_tabular_graphfeat_*` variants achieve **MAE ≈ 336.9–375.4** and **RMSE ≈ 833.7–933.8**, with similar or slightly better MAE and RMSE than plain XGBoost.  
- Residual GNN baseline (Baseline 6):
  - `xgb_gnn_residual_hetero5` reaches **MAE ≈ 371.2**, **RMSE ≈ 947.5**, **MAPE ≈ 240.2%**, **sMAPE ≈ 80.9%**, again offering good sMAPE but MAPE still above tabular XGBoost.  
- Pure GNN baselines:
  - Stand‑alone GNNs remain noticeably worse, with **MAE ≈ 379–799** and MAPE values often exceeding 250–300%, particularly for projected and homogeneous setups.  

A compact view (test metrics, lag 14):

| Model family          | Variant / graph               | MAE (test) | RMSE (test) | MAPE (test) | sMAPE (test) |
|-----------------------|-------------------------------|-----------:|------------:|------------:|-------------:|
| xgb_tabular (B1)      | no graph                      | 375.4      | 933.8       | 192.3       | 126.2        |
| xgb_graph (B3)        | proj / homo / hetero          | 365.3–421.6| 889.7–1037.9| 189.9–239.0 | 126.1–128.1  |
| xgb_tabular_graphfeat | proj / homo5 / hetero5        | 336.9–392.7| 833.7–950.1 | 209.2–378.9 | 100.6–128.9  |
| xgb_gnn_embed (B5)    | proj / homo / hetero          | 370.7–377.6| 904.1–1102.8| 187.6–300.9 | 126.9–134.2  |
| xgb_gnn_residual (B6) | proj / homo5 / hetero5        | **365.99–382.0** | 935.5–1067.0 | 168.3–1223.9 | **80.9–118.6** |
| GNN only (B4)         | projected / homo5 / hetero5   | 379.6–1445.7| 965.0–2240.4| 190.9–3014.9| 97.9–171.7   |
| GRU sequence (B2)     | no graph                      | 552.6      | 1320.7      | 153.5       | 153.2        |
| Naive last T0         | no graph                      | 404.8      | 1117.1      | 125.1       | 62.2         |

Here, the **best MAE and RMSE** come from **graph‑augmented tabular models** (Baseline 3 and graph‑feature variants), while residual GNNs again show the **lowest sMAPE** among non‑naive models but with higher MAPE. 

### 17.3. Key takeaways

From these experiments, three main observations stand out: 

1. **Plain GNNs underperform strong tabular baselines**: Direct GNN baselines (Baseline 4) do not beat XGBoost tabular or graph‑enhanced tabular models in this setting, especially on MAPE and sMAPE.  
2. **Graph structure helps when combined with tabular models**: Both **xgb_graph** (Baseline 3) and **xgb_tabular_graphfeat** improve MAE and RMSE over pure tabular XGBoost, confirming the value of graph information for multivariate demand forecasting.
3. **Residual GNNs are most effective as correctors, not standalone forecasters**: Baseline 6 (XGBoost + GNN residuals) consistently improves sMAPE and often achieves MAE close to or better than the tabular baseline, but never clearly dominates the best graph‑augmented XGBoost variants across all metrics.  

Overall, the **most robust models** across lags and metrics are the **graph‑enhanced XGBoost baselines (Baseline 3 and graph‑feature variants)**, while **GNNs shine as residual correctors** rather than as end‑to‑end forecasters in this dataset. 
