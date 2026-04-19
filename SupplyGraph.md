# Reproducible Supply-Chain Forecasting Benchmark and Controlled Study of Graph/Subgraph Integration

## 1. Motivation and Research Gap

Supply chain planning là một bài toán tự nhiên cho **graph**: sản phẩm, nhà máy, kho, nhóm sản phẩm tạo thành một mạng lưới có cấu trúc rõ ràng. Tuy nhiên, nghiên cứu về graph-based supply chain forecasting vẫn còn mới và thiếu benchmark chuẩn.  

Benchmark hiện có nổi bật là **SupplyGraph** – một dataset thực tế từ FMCG dùng cho supply chain planning bằng GNNs. Nhưng trong thực hành, benchmark này có các vấn đề:  

- Data và code không hoàn toàn mở, khó tái lập kết quả một cách đầy đủ.  
- Paper không cung cấp đủ chi tiết về toàn bộ pipeline (preprocessing, temporal split, tuning), khiến việc reproduce và mở rộng nghiên cứu trở nên khó khăn.  
- Các nghiên cứu sau đó không có một chuẩn chung rõ ràng để so sánh, tạo thành nút thắt cho hướng nghiên cứu GNN trong supply chain.  

Ngoài ra, trong chủ đề này, **cách build graph** (projected vs full homogeneous vs full heterogeneous) và **cách use graph** (graph features vs graph models) là điểm mới quan trọng nhưng chưa có một so sánh chi tiết và hệ thống.

***

## 2. Contributions

Dự án này xây dựng một benchmark **reproducible, open** cho demand forecasting (unit) trong supply chain dựa trên dataset kiểu SupplyGraph, với các đóng góp:

- Một pipeline preprocessing **minh bạch, có code đầy đủ**, từ raw CSV đến `.parquet` và `.pt` cho từng baseline/GNN, với schema rõ ràng và documented bằng markdown.  
- Chuẩn hóa và so sánh **3 chiến lược build graph × 2 chiến lược use graph**:
  - 3 cách build graph:
    1. Projected product graphs (4 views).
    2. Full homogeneous graph (5 loại node).
    3. Full heterogeneous graph (5 loại node, nhiều loại edge).
  - 2 cách use graph:
    1. Graph features: trích xuất thống kê/tính chất từ graph và đưa vào tabular model.
    2. Graph models: dùng GNN trực tiếp trên graph + time series node features.  
- Thiết kế bộ baseline chuẩn:
  - Tabular time series (no graph) – XGBoost.
  - Temporal sequence (no graph) – GRU.
  - Time series with graph features – XGBoost với projected/homo/hetero features.
  - Time series with graph model – GNN (3 kiểu graph). 
- Phân tích phân phối target `sales_order` (unit), thiết kế **output parameterization** cho GNN (raw vs softplus) phù hợp phân phối **zero-inflated + heavy-tailed** của demand.  

Toàn bộ benchmark hiện tại tập trung vào **temporal_type = unit**, để đơn giản hóa phân tích; forecast theo weight được để lại cho công việc tương lai.

***

## 3. Data Overview

Dataset theo đúng spirit của SupplyGraph: **nodes = products**, edges = quan hệ giữa products. Ba thành phần chính:  

- **Nodes**: mỗi node là một sản phẩm.
- **Edges**: quan hệ giữa sản phẩm gồm:
  - Shared plant.
  - Product group / sub-group.
  - Storage location.  
- **Temporal data**: cho mỗi (product, day), có:
  - `sales_order` (demand, unit).
  - `production`.
  - `delivery` (to distributor).
  - `factory_issue`.  

### 3.1. Raw files

- Node metadata:
  - `NodesIndex.csv` – mapping `Node` → `node_id`, `NodeIndex` → `node_index`.
  - `Node Types (Product Group and Subgroup).csv` – `Node`, `Group`, `Sub-Group`.
  - `Nodes Type (Plant & Storage).csv` – `Node`, `Plant`, `Storage Location`.  
- Temporal data (Unit, dạng wide, mỗi product là một cột):
  - `Unit/Sales Order.csv`, `Production.csv`, `Delivery To distributor.csv`, `Factory Issue.csv`.  
- Edge metadata:
  - `Edges (Plants).csv`.
  - `Edges (Product Group).csv`.
  - `Edge (Product Sub-group).csv`.
  - `Edges (Storage Location).csv`.  

***

## 4. Data Processing Pipeline

### 4.1. Node metadata

- Đọc:
  - `NodesIndex.csv` → chuẩn hóa: `node_id`, `node_index`.
  - `Node Types (Product Group and Subgroup).csv` → `node_id`, `group`, `sub_group`.
  - `Nodes Type (Plant & Storage).csv` → `node_id`, `plant`, `storage_location`.  
- Merge 3 bảng trên `node_id` để có `df_meta` (1 dòng / product) chứa toàn bộ thuộc tính tĩnh.

### 4.2. Temporal data (Unit)

#### 4.2.1. Load và chuẩn hóa wide → long

- `_load_temporal_wide_generic(subdir, filename, value_name)`:
  - Đọc file, parse `Date` → datetime.
  - Giữ `Date` + các cột tên trùng `node_id` valid (từ `df_meta`) để loại bỏ noise.  
  - Melt:
    - `Date` → `date`.
    - Column product → `node_id`.
    - Value → metric tương ứng (`sales_order`, `production`, `delivery`, `factory_issue`).
  - Sort (`date`, `node_id`), drop duplicates.  
- `load_temporal_unit_wide` là wrapper cho nhánh Unit.

#### 4.2.2. Kết hợp temporal series

- `load_raw_data_unit`:
  - Merge 4 metrics trên (`date`, `node_id`) (left join).
  - Sort, reset index.
  - Tạo day index:
    - `unique_dates` sort tăng dần.
    - Mỗi `date` → `day = 1..num_days`.
  - Merge với `df_meta` để có `node_index`, `group`, `sub_group`, `plant`, `storage_location`.
  - Reorder:

    ```text
    node_id, node_index, date, day,
    sales_order, production, delivery, factory_issue,
    group, sub_group, plant, storage_location
    ```

  - Sort và drop duplicates trên (`date`, `node_id`).  

### 4.3. Calendar, labels, splits

- `add_calendar_features`:
  - `day_of_week = date.dt.weekday` (0–6).
  - `is_weekend = (day_of_week >= 5)`.
  - `month = date.dt.month`.
  - `day_of_month = date.dt.day`.  
- `create_labels`:
  - Sort theo (`node_id`, `day`).
  - `y_h7 = sales_order.shift(-7)` per `node_id` (forecast horizon H=7 ngày vào tương lai).  
- `assign_splits`:
  - `compute_time_splits(num_days, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)`:
    - `train_end = floor(num_days * TRAIN_RATIO)`.
    - `val_end = floor(num_days * (TRAIN_RATIO + VAL_RATIO))`.
    - `test_end = num_days`.
  - Gán:
    - `day <= train_end` → `train`.
    - `train_end < day <= val_end` → `val`.
    - `day > val_end` → `test`.

### 4.4. Base RAW vs FULL

- `build_base_raw()`:
  - Dùng `load_raw_data_unit`.
  - + `add_calendar_features`, `create_labels`, `assign_splits`.
  - **Không** tạo lag/rolling.
  - Lưu: `base_raw_unit.parquet`.
- `build_base_full()`:
  - Đọc `base_raw_unit.parquet`.
  - + `add_lag_features` cho 4 metrics với `LAG_WINDOW ∈ {7,14}` (lag0–lag(L−1)).
  - + `add_rolling_stats` với window tương ứng (mean, std, max, min).
  - + `filter_valid_samples`: drop mọi dòng có NaN ở lag/rolling/label.
  - Lưu: `base_full_unit.parquet`.

### 4.5. Histogram và phân phối Sales Order

- `data_distribution.py`:
  - Dùng `build_base_raw_for_inspect` để lấy `sales_order` theo (node_id, day) cho unit.
  - Vẽ histogram với 4 scaling:
    - linear–linear.
    - log–log.
    - linear–log.
    - log–linear.  
  - Lưu: `data/processed/plots/sales_order_hist_4scales_unit.png`.

Kết quả: phân phối **zero-inflated**, right-skewed rất mạnh (heavy tail), mean lớn nhưng median thấp, tail dài ở phía giá trị lớn.  

***

## 5. Baselines (No Graph)

### 5.1. Baseline 1 – Tabular time series – XGBoost

#### 5.1.1. Dataset construction

- Input: `base_full_unit.parquet`.
- `build_xgboost_tabular()`:
  - Feature columns:
    - Tất cả cột chứa `"lag"` hoặc `"roll"`.
    - Cột categorical: `"group"`, `"sub_group"`, `"plant"`, `"storage_location"`.
    - Calendar: `"day_of_week"`, `"is_weekend"`, `"month"`, `"day_of_month"`.
  - Base columns: `["node_id", "node_index", "date", "day", "split"]`.
  - Label `"y_h7"` → `"target"`.
  - Tạo `df_h7 = base_cols + feature_cols + ["target"]`.

#### 5.1.2. One-hot encoding

- Categorical: `["group", "sub_group", "plant", "storage_location", "day_of_week", "is_weekend"]`.
- `one_hot_encode_splits(df, CAT_COLS)`:
  - Tách `df_train`, `df_val`, `df_test` theo `split`.
  - `pd.get_dummies` trên từng split.
  - Align val/test theo schema train bằng `reindex(..., fill_value=0)`.
  - Gộp lại `df_all_enc` → lưu `xgboost_tabular_unit.parquet`. 

#### 5.1.3. XGBoost training

- Model: `XGBRegressor` (ví dụ 5000 trees, max_depth=6, learning_rate=0.05, tree_method="hist", sớm dừng qua early_stopping_rounds trên validation).  
- Train trên mẫu `split == "train"`, early stopping trên `val`, report trên `test`.
- Tùy chọn: rolling-origin evaluation, với origin \(T\), train `day <= T`, test `day == T+7`.

### 5.2. Baseline 2 – Temporal sequence – GRU

#### 5.2.1. Dataset construction

- Input: `base_raw_unit.parquet` (không lag/rolling).
- `build_gru_sequence()`:
  - Chọn:

    ```text
    node_id, node_index, date, day, split,
    sales_order, production, delivery, factory_issue,
    day_of_week, is_weekend, month, day_of_month,
    y_h7 → target,
    group, sub_group, plant, storage_location
    ```

  - One-hot các cột categorical như XGBoost.
  - Lưu: `gru_sequence_unit.parquet`. 

#### 5.2.2. GRU usage

- Sort (`node_id`, `day`), group theo `node_id`, tạo sequence `(T_i, D)` per product.
- GRU input: các dynamic features + calendar + OHE category; horizon H=7 target tại timestep tương ứng.
- Model: GRU encoder + MLP head, loss MSE trên target gốc. 

***

## 6. Graph Construction Strategies

### 6.1. Projected product graphs (4 views)

4 homogeneous product graphs, mỗi graph 1 loại “similarity”:

- `same_plant`: cạnh giữa products cùng plant.
- `same_group`: products cùng group.
- `same_subgroup`: products cùng subgroup.
- `same_storage`: products cùng storage_location.  

Mỗi view lưu riêng `edge_index` cho product graph.

### 6.2. Full homogeneous graph

- Node types: `product`, `plant`, `group`, `subgroup`, `storage_location`.
- Edges: tất cả quan hệ product–(plant/group/subgroup/storage) và các edges giữa non-product nodes, nhưng **không phân biệt loại** (untyped).
- Toàn bộ graph flatten thành 1 `edge_index` với global node index, dùng type embedding để phân biệt node type trong GNN.  

### 6.3. Full heterogeneous graph

- Node types: như trên (5 loại).
- Edge types, ví dụ:
  - `product_plant`, `product_group`, `product_subgroup`, `product_storage`.
  - `plant_storage`, `plant_group`, `plant_subgroup`.
  - `group_subgroup`, `group_storage`, `subgroup_storage`.  
- Sử dụng `edge_index_dict[(src_type, rel, dst_type)]` trong Heterogeneous GNN.

Mục tiêu là so sánh chi tiết (projected vs homogeneous vs heterogeneous) × (graph features vs GNN).

***

## 7. Graph-based Baselines – GNN

### 7.1. Node features cho GNN

Cho mỗi graph-type (projected/homo/hetero), node features của `product` bao gồm:

- 4 temporal metrics ở time t: `sales_order`, `production`, `delivery`, `factory_issue`.
- Calendar: `day_of_week`, `is_weekend`, `month`, `day_of_month`.
- Không dùng lag/rolling trong phiên bản GNN baseline hiện tại (cho phép GNN học từ context graph + snapshot features).  

Các node-type khác (plant/group/subgroup/storage):

- Có thể dùng feature zeros cộng với type embeddings đơn giản, để cô lập hiệu ứng cấu trúc graph so với thông tin feature.

### 7.2. Saved GNN packages

`build_gnn_dataset.py` tạo các file:

- Projected: `gnn_projected_h{H}_lag{L}_unit.pt`.
- Homogeneous: `gnn_homo5_h{H}_lag{L}_unit.pt`.
- Heterogeneous: `gnn_hetero5_h{H}_lag{L}_unit.pt`.  

Mỗi package chứa:

- `X_product`: `[T, N_prod, F]`.
- `Y_product`: `[T, N_prod]` (target horizon H).
- `days`: list ngày (hoặc index).
- `split`: list `"train" / "val" / "test"` per day.
- `edge_index_dict`: tùy loại graph:
  - Projected: `{view_name: edge_index}` với `view_name` ∈ {`same_group`, `same_subgroup`, `same_plant`, `same_storage`}.
  - Homo/Hetero: `{(src_type, rel, dst_type): edge_index}`.
- `num_nodes_dict`: `{node_type: N_type}` cho homo/hetero.

***

## 8. Output Parameterizations for GNN (Raw vs Softplus)

Phân phối `sales_order` có đặc trưng:

- Rất nhiều zero (zero-inflated).
- Right-skewed mạnh, tail dài, nhiều outliers lớn.  

Với regression models (XGBoost/GRU/GNN), output linear có thể sinh ra giá trị âm dù demand thực tế không âm. Điều này không sai về mặt loss, nhưng kém tự nhiên về mặt business và gây vấn đề cho các metric như MAPE/sMAPE.  
Vì vậy, với GNN baseline, chúng tôi so sánh **hai chế độ output**:

### 8.1. Raw output

- Output layer là `Linear`: \(z = W h + b\).
- Dự báo cuối cùng là \(y_{\text{pred}} = z\) (không ràng buộc).  
- Dùng cho setting `target_transform = "raw"`, với loss MSE trên target gốc.

### 8.2. Softplus output

- Output layer: \(z = \text{Linear}(h)\).
- Dự báo cuối cùng: \(y_{\text{pred}} = \text{Softplus}(z) = \log(1 + \exp(z))\).  
- Đảm bảo dự báo không âm một cách “mềm” (smooth), tránh gradient chết như ReLU và ổn định hơn trong practice.  
- Được bật qua tham số `use_softplus_output: bool` trong `ProjectedGINRegressor`, `HomogeneousFiveTypeGINRegressor`, `HeterogeneousGINRegressor`.


***

## 9. GNN Training Pipeline

### 9.1. Early stopping và splits

- Dùng `split` theo ngày:
  - `train`: ngày sớm.
  - `val`: ngày giữa, dùng để early stopping.
  - `test`: ngày cuối, chỉ dùng để report.  
- Early stopping:
  - Monitor `val_loss` (MSE trên target gốc).
  - Nếu `val_loss` không cải thiện thêm sau `patience` epoch (với `min_delta`), dừng training.
  - Lưu và restore **best model weights** theo validation loss. 

### 9.2. Projected GNN baseline

- Mỗi view product graph (4 view: `same_group`, `same_subgroup`, `same_plant`, `same_storage`) chạy **hai chế độ output**:
  - `raw`: linear output.
  - `softplus`: linear + Softplus.  
- Với mỗi `(lag_window ∈ {7,14})`:
  - Load `gnn_projected_h7_lag{L}_unit.pt`.
  - Train:
    - Loop theo `day` như mini-batch (block size = `batch_days`).
    - Mỗi timestep: truyền `X[t]` và `edge_index[view]` vào `ProjectedGINRegressor`.
    - Loss MSE giữa dự báo và target gốc.
  - Sau training: evaluate trên train/val/test, compute MAE/RMSE/MAPE/sMAPE cho cả chế độ raw và softplus. 

### 9.3. Homogeneous 5-type GNN baseline

- Build global homogeneous graph:
  - Concatenate node types theo `node_type_order`.
  - Tạo offset để map local index → global index.
  - Symmetrize edge list (add reverse edges).
- Model `HomogeneousFiveTypeGINRegressor`:
  - Input: concat features + type embedding (embedding theo node_type).
  - Output: chỉ lấy phần tương ứng với product nodes.
- Train và evaluate:
  - Với hai chế độ output: raw vs softplus.

### 9.4. Heterogeneous 5-type GNN baseline

- Dùng `edge_index_dict[(src_type, rel, dst_type)]` từ gói heterogeneous.
- `HeterogeneousGINRegressor`:
  - Một HeteroConv/GIN layer per block, mỗi edge-type có conv riêng.
  - Output trên node-type `product`.
- Train và evaluate như trên, với hai chế độ output: raw vs softplus.

***

## 10. Time Series Visualization per Product

Để hiểu rõ hơn pattern demand theo thời gian và hành vi mô hình, pipeline GNN hiện tại tự động **plot time series per product (subset)**:

- Với mỗi `(lag_window ∈ {7,14})`:
  - Sau khi load `gnn_projected_h7_lag{L}_unit.pt`, gọi:

    ```python
    plot_product_time_series(pkg_proj, temporal_type="unit", lag_window=L, max_products=20)
    ```

  - `plot_product_time_series`:
    - Dùng `Y_product` (shape `[T, N_prod]`) và `days`.
    - Vẽ line chart cho tối đa `max_products` sản phẩm đầu tiên, dùng `matplotlib.subplots` để tạo grid subplots.  
    - Lưu: `data/processed/predictions/gnn_baselines/plots/ts_products_unit_lag{L}.png`.

Mục đích:

- Visual kiểm tra seasonality, spikes, periods nhiều zero.
- Hỗ trợ debug các sản phẩm có forecast khó (zero-inflated/hard tail) và so sánh qualitative giữa raw vs softplus.

***

## 11. File Structure (Updated)

- `data/raw/NODE_DIR/...` – node metadata CSVs.
- `data/raw/TEMPORAL_DIR/Unit/...` – temporal CSVs (unit).
- `data/raw/edges/...` – edge metadata.  

- `data/processed/base_raw_unit.parquet` – raw sequences + calendar + labels + splits.
- `data/processed/base_full_unit.parquet` – thêm lag/rolling.  

- `data/processed/baseline/xgboost_tabular_unit.parquet` – XGBoost tabular (OHE).
- `data/processed/baseline/gru_sequence_unit.parquet` – GRU base (OHE, no lag/rolling).
- `data/processed/plots/sales_order_hist_4scales_unit.png` – target histogram (4 scales).  

- `data/processed/gnn/gnn_projected_h7_lag{L}_unit.pt` – projected graph packages.
- `data/processed/gnn/gnn_homo5_h7_lag{L}_unit.pt` – homogeneous graph packages.
- `data/processed/gnn/gnn_hetero5_h7_lag{L}_unit.pt` – heterogeneous graph packages.

Thêm subsection này ngay sau phần File Structure là hợp lý. Bạn có thể chèn vào cuối README / report:

***

## 12. Key Empirical Findings (Unit Demand, H=7)

Các thực nghiệm ban đầu tập trung vào **unit demand** với horizon 7 ngày và hai lag window (7, 14), so sánh giữa no-graph, graph-features và GNN (raw vs softplus). Dưới đây là những quan sát chính, dựa trên MAE/RMSE/MAPE/sMAPE – các metric chuẩn trong forecast accuracy cho supply chain. 

### 12.1. Baselines đơn giản đã tương đối mạnh

- Naive last-t0 (dùng giá trị gần nhất làm forecast) đã đạt MAE ≈ 405 và RMSE ≈ 1117 trên test, với sMAPE ≈ 62 cho unit demand, cho thấy bản thân chuỗi có mức autocorrelation nhất định. 
- XGBoost tabular với lag 14 ngày trên features (no-graph) giảm đáng kể RMSE so với naive, dù sMAPE vẫn nằm quanh 120–130, phù hợp với bối cảnh demand biến động và zero-inflated.  
### 12.2. GRU không vượt XGBoost trong cấu hình hiện tại

- GRU sequence trên target raw thể hiện RMSE và sMAPE kém hơn XGBoost tabular, cả ở lag 7 và 14, gợi ý rằng trên dataset này, **lag-based gradient boosting với OHE categorical** là baseline rất cạnh tranh.  
- Điều này phù hợp với các báo cáo gần đây về demand forecasting, nơi các phương pháp ML tabular cấu hình tốt vẫn là baseline khó đánh bại trên dữ liệu vừa và nhỏ. 

### 12.3. Graph features + XGBoost mang lại lợi ích ổn định

- Khi thêm graph features (từ projected, homogeneous hoặc heterogeneous graphs) vào XGBoost, RMSE test tiếp tục giảm so với XGBoost no-graph, đặc biệt ở lag 14, trong khi MAE và sMAPE cũng được cải thiện nhẹ.  
- Khác biệt giữa ba kiểu graph features (proj/homo/hetero) không quá lớn trong kết quả hiện tại, cho thấy **bản thân việc đưa thông tin cấu trúc vào tabular model** đã chiếm phần lớn lợi ích.

### 12.4. GNN hetero5 + Softplus có tiềm năng rõ rệt ở lag 7

- Với heterogeneous graph và lag 7, GNN với output raw cho MAPE và sMAPE khá cao, phản ánh tác động của dự báo âm và tail nặng trong demand.
- Khi chuyển sang **Softplus output** (non-negative, trơn), MAPE và sMAPE giảm đáng kể, đồng thời MAE và RMSE trở nên cạnh tranh với XGBoost + graph features, cho thấy **ràng buộc non-negativity trong model** là rất quan trọng đối với đếm / demand zero-inflated.  

### 12.5. Homogeneous GNN và một số run lag 14 cần được tinh chỉnh thêm

- GNN homogeneous 5-type với cấu hình hiện tại chưa đạt kết quả tốt: các metric RMSE và sMAPE cao hơn so với hetero5 và XGBoost, có thể do thiết kế feature cho non-product nodes còn nghèo hoặc kiến trúc chưa tối ưu.  
- Một số run GNN với Softplus ở lag 14 cho kết quả “bão hòa” (metric gần như constant), cho thấy cần rerun hoặc tuning thêm; vì vậy, phân tích chính ở thời điểm hiện tại chủ yếu dựa trên lag 7 và các run ổn định.

### 12.6. Ý nghĩa đối với thiết kế benchmark

Từ các quan sát trên, có thể rút ra:

- Chuẩn baseline **XGBoost lag-based + categorical OHE** vẫn là mốc quan trọng cho mọi phương pháp mới trong supply-chain forecasting.
- **Graph information hữu ích** nhất khi được tích hợp cẩn thận: cả qua graph features cho XGBoost và qua GNN hetero5 với output Softplus trên unit demand zero-inflated. 
- Non-negativity không chỉ là chi tiết kỹ thuật; với demand forecasting, chọn đúng output parameterization (raw vs Softplus) tác động trực tiếp tới MAPE/sMAPE và interpretability của forecast. 
