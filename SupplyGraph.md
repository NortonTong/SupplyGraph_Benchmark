# Reproducible Supply‑Chain Forecasting Benchmark and Controlled Study of Graph/Subgraph Integration

## 1. Motivation and Research Gap

Supply chain planning là một bài toán tự nhiên cho **graph**: sản phẩm, nhà máy, kho và nhóm sản phẩm tạo thành một mạng lưới với cấu trúc rõ ràng. Tuy nhiên, nghiên cứu về graph‑based supply chain forecasting vẫn còn mới và thiếu một benchmark chuẩn, có thể reproduce đầy đủ.

Benchmark đáng chú ý hiện tại là **SupplyGraph**, một dataset từ ngành FMCG cho supply chain planning bằng GNNs. Tuy vậy, khi đem vào thực hành nghiên cứu, SupplyGraph có một số hạn chế:  

- Data và code không hoàn toàn mở, dẫn tới việc tái lập đầy đủ pipeline rất khó.
- Bản paper gốc không cung cấp đầy đủ chi tiết về preprocessing, temporal split, tuning và các baseline, khiến việc reproduce và mở rộng nghiên cứu bị hạn chế.  
- Các công trình sau sử dụng SupplyGraph chưa thống nhất về pipeline, tạo ra một “khoảng trống” về benchmark chuẩn cho GNN trong supply chain.

Ngoài ra, trong bối cảnh này, **cách xây dựng graph** (projected vs full homogeneous vs full heterogeneous) và **cách sử dụng graph** (graph features vs graph models) là hai trục quyết định quan trọng, nhưng chưa có một so sánh có kiểm soát và chi tiết.

***

## 2. Contributions

Chúng tôi xây dựng một benchmark **reproducible, open** cho demand forecasting (unit) trong supply chain dựa trên dataset kiểu SupplyGraph, với các đóng góp:

- Một pipeline preprocessing **minh bạch, có code đầy đủ**, từ raw CSV đến `.parquet` và `.pt` cho từng baseline/GNN, với schema rõ ràng và được mô tả chi tiết bằng markdown.
- Chuẩn hóa và so sánh **3 chiến lược build graph × 2 chiến lược use graph**:
  - 3 cách build graph:
    1. Projected product graphs (4 views).
    2. Full homogeneous graph (5 loại node).
    3. Full heterogeneous graph (5 loại node, nhiều loại edge).  
  - 2 cách use graph:
    1. Graph features: trích xuất thống kê/tính chất từ graph và đưa vào tabular model (XGBoost).  
    2. Graph models: dùng GNN trực tiếp trên graph với node‑level time series features.  
- Thiết kế bộ baseline chuẩn:
  - Naive time series (no graph) – last‑t0.  
  - Tabular time series (no graph) – XGBoost.  
  - Temporal sequence (no graph) – GRU với 3 cách tham số hóa output (raw, softplus, log1p‑head).  
  - Time series with graph features – XGBoost với projected/homo/hetero features.  
  - Time series with graph models – GNN (3 kiểu graph) với 3 cách tham số hóa output (raw, softplus, log1p).  
- Phân tích phân phối target `sales_order` (unit), thiết kế **output parameterizations** (raw vs softplus vs log1p) cho GRU và GNN phù hợp với phân phối **zero‑inflated + heavy‑tailed** của demand.

Trong scope hiện tại, benchmark tập trung vào **temporal_type = unit**; forecast theo weight được để lại cho công việc tương lai.

***

## 3. Data Overview

Dataset: **nodes = products**, edges = quan hệ giữa products; các series thời gian là các luồng quantity theo ngày. Ba thành phần chính: 
- **Nodes**: mỗi node là một sản phẩm.  
- **Edges**: quan hệ giữa sản phẩm:
  - Shared plant.  
  - Product group / sub‑group.  
  - Storage location.  
- **Temporal data**: cho mỗi cặp (product, day):
  - `sales_order` (demand, unit).  
  - `production`.  
  - `delivery` (to distributor).  
  - `factory_issue`.  

### 3.1. Raw files

- Node metadata:
  - `NodesIndex.csv` – mapping `Node` → `node_id`, `NodeIndex` → `node_index`.  
  - `Node Types (Product Group and Subgroup).csv` – `Node`, `Group`, `Sub‑Group`.  
  - `Nodes Type (Plant & Storage).csv` – `Node`, `Plant`, `Storage Location`.  
- Temporal data (Unit, wide, mỗi product là một cột):
  - `Unit/Sales Order.csv`, `Production.csv`, `Delivery To distributor.csv`, `Factory Issue.csv`.  
- Edge metadata:
  - `Edges (Plants).csv`.  
  - `Edges (Product Group).csv`.  
  - `Edge (Product Sub‑group).csv`.  
  - `Edges (Storage Location).csv`.  

***

## 4. Data Processing Pipeline

### 4.1. Node metadata

- Đọc:
  - `NodesIndex.csv` → chuẩn hóa thành `node_id`, `node_index`.  
  - `Node Types (Product Group and Subgroup).csv` → `node_id`, `group`, `sub_group`.  
  - `Nodes Type (Plant & Storage).csv` → `node_id`, `plant`, `storage_location`.  
- Merge 3 bảng trên `node_id` để có `df_meta` (một dòng / product) chứa toàn bộ thuộc tính tĩnh.

### 4.2. Temporal data (Unit)

#### 4.2.1. Load và chuẩn hóa wide → long

- `_load_temporal_wide_generic(subdir, filename, value_name)`:
  - Đọc file, parse `Date` → datetime.  
  - Giữ `Date` + các cột tên trùng với `node_id` hợp lệ (từ `df_meta`) để loại bỏ noise.  
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
  - `y_h7 = sales_order.shift(-7)` per `node_id` (forecast horizon H = 7 ngày).  
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
  - + `add_lag_features` cho 4 metrics với `LAG_WINDOW ∈ {7,14}` (lag 0 đến lag L−1).  
  - + `add_rolling_stats` với cùng windows (mean, std, max, min).  
  - + `filter_valid_samples`: drop toàn bộ dòng có NaN ở lag/rolling/label.  
  - Lưu: `base_full_unit.parquet`.

### 4.5. Histogram và phân phối Sales Order

- `data_distribution.py`:
  - Dùng `build_base_raw_for_inspect` để lấy `sales_order` theo (node_id, day).  
  - Vẽ histogram với 4 scaling:
    - linear–linear.  
    - log–log.  
    - linear–log.  
    - log–linear.  
  - Lưu: `data/processed/plots/sales_order_hist_4scales_unit.png`.  

Kết quả: phân phối **zero‑inflated**, right‑skewed rất mạnh (heavy tail), mean lớn nhưng median thấp, tail dài về phía giá trị lớn – đặc trưng điển hình của intermittent demand. 

***

## 5. Baselines (No Graph)

### 5.1. Baseline 0 – Naive last‑t0

- Input: `xgboost_h7_lag{L}_unit_full.csv` (full baseline cho XGBoost, chứa `sales_order`, `y_h7`, `split`, calendar và lag features).  
- Ý tưởng: `y_hat(t0+7) = sales_order(t0)` (giá trị unit gần nhất tại thời điểm t0).  
- Implementation:
  - Dùng cột `sales_order` làm `last_value_t0`.  
  - Trên test (`split == "test"`), tạo `y_pred_last_t0 = last_value_t0`.  
  - Fallback: nếu thiếu `sales_order` cho một điểm, thay bằng mean `y_h7` của node trong train, nếu vẫn NaN thì dùng global mean train.  
  - Tính MAE, RMSE, MAPE, sMAPE và lưu predictions theo `node_id`, `date`, `day`, `y_true`, `y_pred`.  

Naive last‑t0 là baseline tham chiếu đơn giản nhưng quan trọng để đánh giá mức cải thiện thực sự của các mô hình phức tạp.

### 5.2. Baseline 1 – Tabular time series – XGBoost

#### 5.2.1. Dataset construction

- Input: `base_full_unit.parquet`.  
- `build_xgboost_tabular()`:
  - Base columns: `["node_id", "node_index", "date", "day", "split"]`.  
  - Label `"y_h7"` → `"target"`.  
  - Feature columns:
    - Tất cả cột chứa `"lag"` hoặc `"roll"` (lag/rolling của 4 metrics).  
    - Categorical: `"group"`, `"sub_group"`, `"plant"`, `"storage_location"`.  
    - Calendar: `"day_of_week"`, `"is_weekend"`, `"month"`, `"day_of_month"`.  
  - Tạo `df_h7 = base_cols + feature_cols + ["target"]`.

#### 5.2.2. One‑hot encoding

- Categorical: `["group", "sub_group", "plant", "storage_location", "day_of_week", "is_weekend"]`.  
- `one_hot_encode_splits(df, CAT_COLS)`:
  - Tách `df_train`, `df_val`, `df_test` theo `split`.  
  - `pd.get_dummies` trên từng split.  
  - Align val/test theo schema train bằng `reindex(..., fill_value=0)`.  
  - Gộp lại `df_all_enc` → `xgboost_tabular_unit.parquet`.  

#### 5.2.3. XGBoost training

- Model: `XGBRegressor` (ví dụ 5000 trees, max_depth = 6, learning_rate = 0.05, tree_method = "hist"), early stopping trên validation. [nixtla](https://www.nixtla.io/blog/intermittent-demand)
- Train trên `split == "train"`, early stopping theo `val`, report metric trên `test`.  

### 5.3. Baseline 2 – Temporal sequence – GRU

#### 5.3.1. Dataset construction

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

  - One‑hot các cột categorical như XGBoost.  
  - Lưu: `gru_sequence_h7_L{window}_unit.parquet`, với window ∈ {7, 14}.  

#### 5.3.2. GRU model và output parameterizations

- Input vào GRU:
  - Chuỗi theo thời gian cho từng product:  
    - Dynamic: `sales_order`, `production`, `delivery`, `factory_issue`.  
    - Calendar: `day_of_week`, `is_weekend`, `month`, `day_of_month`.  
    - Categorical OHE (group, subgroup, plant, storage).  
  - Không dùng lag/rolling; toàn bộ tính “nhớ” là từ GRU.  
- `GRURegressor`:
  - GRU encoder (num_layers, hidden_size) → lấy hidden cuối `h_T`.  
  - MLP head `fc(h_T)` → scalar head.  
  - Ba chế độ output:
    - **raw**: `y_hat = head`.  
    - **softplus**: `y_hat = Softplus(head)` (non‑negative, smooth).  
    - **log1p‑head**: `y_hat = expm1(head).clamp_min(0)`; ở đây `head` là latent log1p‑space, nhưng loss vẫn tính trên scale gốc.  
- Training:
  - Loss = MSE(`y_pred`, `y_true`) trên **scale output** (tức scale gốc, vì softplus/log1p đã được expm1 bên trong model).  
  - Metric train/val/test: MAE, RMSE, MAPE, sMAPE trên cùng scale với output.  

***

## 6. Graph Construction Strategies

### 6.1. Projected product graphs (4 views)

4 homogeneous product graphs, mỗi graph biểu diễn một loại “similarity” giữa products:

- `same_plant`: cạnh giữa products cùng plant.  
- `same_group`: products cùng group.  
- `same_subgroup`: products cùng subgroup.  
- `same_storage`: products cùng storage_location.  

Mỗi view lưu riêng `edge_index` cho product graph.

### 6.2. Full homogeneous graph

- Node types: `product`, `plant`, `group`, `subgroup`, `storage_location`.  
- Edges: toàn bộ quan hệ product–(plant/group/subgroup/storage) nhưng **không phân biệt loại** (untyped).  
- Flatten:
  - Gán global index cho tất cả node types theo `node_type_order`.  
  - Gộp tất cả edges về một `edge_index` chung.  
  - Dùng type embeddings để phân biệt node type trong GNN.

### 6.3. Full heterogeneous graph

- Node types: như trên (5 loại).  
- Edge types, ví dụ:  
  - `product_plant`, `product_group`, `product_subgroup`, `product_storage`.  
- Dùng `edge_index_dict[(src_type, rel, dst_type)]` trong Heterogeneous GNN. [arxiv](https://arxiv.org/html/2401.15299v1)

Mục tiêu là so sánh chi tiết (projected vs homogeneous vs heterogeneous) × (graph features vs GNN).
***

## 7. Graph‑based Baselines – GNN

### 7.1. Node features cho GNN

Cho mỗi graph‑type (projected/homo/hetero), **node features cho product** tại thời điểm t bao gồm:

- 4 temporal metrics ở time t: `sales_order`, `production`, `delivery`, `factory_issue`.  
- Calendar tại t: `day_of_week`, `is_weekend`, `month`, `day_of_month`.  
- GNN nhìn snapshot feature tại từng day (bao gồm lag/rolling và các one-hot encoding features như group, subgroup, plant, storage_location).  

Các node‑type khác (plant/group/subgroup/storage):

- Trong homogeneous GNN: dùng zeros + type embeddings; chỉ output trên node `product`.  
- Trong heterogeneous GNN: mỗi node type có linear projection vào hidden, có thể là zeros nếu không có feature riêng.

### 7.2. Saved GNN packages

`build_gnn_dataset.py` tạo các file:

- Projected: `gnn_projected_h7_lag{L}_unit.pt`.  
- Homogeneous: `gnn_homo5_h7_lag{L}_unit.pt`.  
- Heterogeneous: `gnn_hetero5_h7_lag{L}_unit.pt`.  

Mỗi package chứa:

- `X_product`: tensor `[T, N_prod, F]` – features của product nodes theo thời gian.  
- `Y_product`: `[T, N_prod]` – target horizon H cho từng (day, product).  
- `days`: list ngày (hoặc index).  
- `split`: list `"train" / "val" / "test"` per day.  
- `edge_index_dict`: tùy loại graph:
  - Projected: `{view_name: edge_index}` với `view_name` ∈ {`same_group`, `same_subgroup`, `same_plant`, `same_storage`}.  
  - Homo/Hetero: `{(src_type, rel, dst_type): edge_index}`.  
- `num_nodes_dict`: `{node_type: N_type}` cho homogeneous/heterogeneous.

***

## 8. Output Parameterizations for GRU and GNN

Phân phối `sales_order` có đặc trưng:

- Rất nhiều zero (zero‑inflated).  
- Right‑skewed mạnh, tail dài, nhiều outliers lớn.
Trên phân phối này, việc chọn output head và scale loss là quan trọng.

### 8.1. Raw output

- Output: `y_pred = head`.  
- Cho phép dự báo âm.  
- Loss = MSE trên scale gốc.  
- Dùng cho tất cả model families (XGBoost, GRU, GNN) như baseline mặc định.

### 8.2. Softplus output

- Output: \(y_{\text{pred}} = \text{Softplus}(z) = \log(1 + \exp(z))\), non‑negative, smooth.  
- Tránh giá trị âm cho demand.  
- Áp dụng cho:
  - GRU: `GRURegressor(is_softplus=True)`.  
  - GNNs: `ProjectedGINRegressor/HomogeneousFiveTypeGINRegressor/HeterogeneousGINRegressor(is_softplus=True)`.  

### 8.3. Log1p‑based output

- Ý tưởng: head ~ log1p(y), output = expm1(head).  
- Trong GRU:
  - `is_log1p=True` → output `y_pred = expm1(head).clamp_min(0)`, loss vẫn là `MSE(y_pred, y_true)` trên scale gốc.  
- Trong GNN:
  - `is_log1p=True` → output `y_pred = expm1(clamp(head, max=max_log)).clamp_min(0)`.  
- Thực nghiệm cho thấy với GNN, biến thể này thường dẫn đến loss bất ổn (NaN hoặc hội tụ nghiệm constant), dù đã thử clamp log‑head và giảm LR, nên kết quả log1p của GNN được coi là “negative result” về loss/optimization chứ không phản ánh giới hạn kiến trúc.

***

## 9. GNN Training Pipeline

### 9.1. Early stopping và splits

- Sử dụng `split` trên trục **ngày**:
  - `train`: các ngày sớm nhất.  
  - `val`: các ngày ở giữa, dùng cho early stopping.  
  - `test`: các ngày cuối, chỉ dùng để đánh giá.  
- Early stopping:
  - Monitor `val_loss` (MSE trên target gốc).  
  - Dừng nếu `val_loss` không cải thiện sau một số epoch (patience, min_delta).  
  - Luôn lưu và restore **best model weights** dựa trên validation loss.

### 9.2. Projected GNN baseline

- Đối với mỗi lag window L ∈ {7, 14}:
  - Load `gnn_projected_h7_lag{L}_unit.pt`.  
  - Chọn một view product graph (`same_group`, `same_subgroup`, `same_plant`, `same_storage`).  
- Mô hình: `ProjectedGINRegressor` với 3 chế độ output (raw, softplus, log1p).  
- Training:
  - Mini‑batch theo **days**: mỗi batch là một block ngày với `X_product[t]` và `edge_index[view]`.  
  - Loss: MSE giữa `y_hat` (output đã qua raw/softplus/log1p head) và `Y_product` trên scale gốc.  
- Evaluation:
  - Tính MAE/RMSE/MAPE/sMAPE cho train/val/test cho từng biến thể output.  
  - Lưu CSV predictions per (day, product) và các plot time‑series per product cho test.

### 9.3. Homogeneous 5‑type GNN baseline

- Xây homogeneous graph:
  - Gán offsets cho từng node type theo `node_type_order`.  
  - Ghép edges từ `edge_index_dict` về một `edge_index` global (symmetrize bằng cách thêm reverse edges).  
- Mô hình: `HomogeneousFiveTypeGINRegressor`:
  - Input: concat feature + type embedding.  
  - Output: scalar head trên **toàn bộ nodes**, chỉ cắt phần `product` để tính loss.  
  - 3 chế độ output: raw, softplus, log1p (expM1 head_all).  
- Training/Eval:
  - Giống projected GNN, nhưng xử lý input như multi‑type graph flatten.

### 9.4. Heterogeneous 5‑type GNN baseline

- Sử dụng `edge_index_dict[(src_type, rel, dst_type)]`.  
- Mô hình: `HeterogeneousGINRegressor`:
  - Mỗi node type có projection riêng vào hidden.  
  - Mỗi layer là một `HeteroConv`/GIN over `edge_index_dict`.  
  - Output MLP head trên node type `product`.  
  - Ba chế độ output: raw, softplus, log1p.  
- Training/Eval:
  - Giống projected/homogeneous GNN, với loss trên product nodes.

***

## 10. Time Series Visualization per Product

Để hiểu rõ hơn pattern demand và hành vi mô hình, pipeline tự động **vẽ time series per product**:

- Với mỗi `(lag_window ∈ {7,14})` và mỗi graph‑type:
  - Sau khi train, dùng `Y_product` và `y_pred` trên test để vẽ `y_true` vs `y_pred` theo ngày cho nhiều products.  
  - Lưu các plot vào `data/processed/predictions/.../plots_*`.  

Mục đích:

- Visual kiểm tra seasonality, spikes, periods nhiều zero.  
- Hỗ trợ debug các sản phẩm khó forecast và so sánh qualitative giữa raw / softplus / log1p.

***

## 11. File Structure (Updated)

- `data/raw/NODE_DIR/...` – node metadata CSVs.  
- `data/raw/TEMPORAL_DIR/Unit/...` – temporal CSVs (unit).  
- `data/raw/edges/...` – edge metadata.  

- `data/processed/base_raw_unit.parquet` – raw sequences + calendar + labels + splits.  
- `data/processed/base_full_unit.parquet` – thêm lag/rolling.  

- `data/processed/baseline/xgboost_tabular_unit.parquet` – XGBoost tabular (OHE).  
- `data/processed/baseline/gru/gru_sequence_h7_L{window}_unit.parquet` – GRU base (OHE, windowed, no lag/rolling).  
- `data/processed/baseline/xgboost/xgboost_h7_lag{L}_unit_full.csv` – full baseline CSV cho naive/XGB.  

- `data/processed/gnn/gnn_projected_h7_lag{L}_unit.pt` – projected graph packages.  
- `data/processed/gnn/gnn_homo5_h7_lag{L}_unit.pt` – homogeneous graph packages.  
- `data/processed/gnn/gnn_hetero5_h7_lag{L}_unit.pt` – heterogeneous graph packages.  

- `data/processed/predictions_naive/...` – naive last‑t0 predictions + summary.  
- `data/processed/predictions/baseline_1/xgb_tabular/...` – XGBoost predictions.  
- `data/processed/predictions/baseline_2/gru/...` – GRU predictions (3 output modes).  
- `data/processed/predictions/baseline_4/gnn/...` – GNN predictions (3 graph types × 3 output modes).  


***

## 12. Key Empirical Findings (Unit Demand, H = 7)

Các thực nghiệm tập trung vào **unit demand**, horizon 7 ngày với hai lag window (7, 14), so sánh:  

- No-graph: naive last-t0, XGBoost tabular, GRU sequence.  
- Graph-features: XGBoost + projected/homogeneous/heterogeneous graph features.  
- Graph-models: GNN (projected/homo/hetero) với các output transform khác nhau (raw, softplus, log1p).  

Các metric được báo cáo trên test: MAE, RMSE, MAPE, sMAPE – phù hợp với thực hành forecast demand zero-inflated.
### 12.1. Naive last‑t0 đặt baseline tham chiếu

- Naive last‑t0 (dự báo `y_hat(t0+7) = sales_order(t0)`) trên `temporal_type = unit` đạt MAE ≈ 405 và RMSE ≈ 1117, với sMAPE khoảng 62 (trên cấu hình lag 7, H=7).  
- Điều này cho thấy chuỗi demand có mức tự tương quan đủ rõ để một chiến lược cực kỳ đơn giản vẫn đạt lỗi tương đối thấp theo sMAPE, và là baseline tham chiếu có ý nghĩa cho toàn bộ benchmark. 
### 12.2. XGBoost tabular (no graph) là baseline rất mạnh

- XGBoost tabular với lag 14 (`xgb_tabular`, no_graph) đạt MAE ≈ 375.4, RMSE ≈ 933.8, MAPE ≈ 192.3, sMAPE ≈ 126.2 trên test – nằm trong nhóm tốt nhất về MAE/sMAPE trong toàn bộ các mô hình thử nghiệm.  
- Với lag 7, XGBoost tabular vẫn giữ hiệu năng cạnh tranh (MAE ≈ 409.7, RMSE ≈ 1035.7, sMAPE ≈ 127.3), cho thấy **lag-based gradient boosting với OHE categorical** là baseline khó bị vượt qua trong bối cảnh demand zero‑inflated, heavy‑tailed. 
### 12.3. Graph features + XGBoost cho cải thiện nhẹ nhưng ổn định

Nhóm “xgb_graph” (XGBoost + graph features):

- Ở lag 14, ba biến thể graph features đều cho MAE quanh 365–368 và RMSE 890–932:  
  - `xgb_graph_hetero_raw`: MAE ≈ 365.32, RMSE ≈ 889.65, sMAPE ≈ 126.08.  
  - `xgb_graph_homo_raw`: MAE ≈ 365.93, RMSE ≈ 901.42, sMAPE ≈ 126.38.  
  - `xgb_graph_proj_raw`: MAE ≈ 368.18, RMSE ≈ 932.15, sMAPE ≈ 126.58.  
- So với XGBoost no‑graph, RMSE được cải thiện nhẹ, MAE và sMAPE tương đương hoặc nhỉnh hơn một chút; sự khác biệt giữa projected/homo/hetero features không quá lớn. Điều này gợi ý **việc đưa thông tin cấu trúc vào dạng features cho tabular model mang lại lợi ích ổn định nhưng vừa phải**, và bản thân XGBoost vẫn đóng vai trò chính. 

Ở lag 7, xgb_graph_homo/hetero/proj có MAE ≈ 401–422, RMSE ≈ 1004–1038, sMAPE ≈ 127–128, vẫn cạnh tranh tốt và nhất quán với lag 14.

### 12.4. GRU sequence không vượt được XGBoost trên dataset hiện tại

- GRU sequence no‑graph với các biến thể output (`raw`, `softplus`, `log1p-head`) cho MAE và RMSE cao hơn XGBoost tabular trong hầu hết cấu hình:  
  - H=7, w=7:  
    - GRU raw: MAE ≈ 506.4, RMSE ≈ 1263.5, sMAPE ≈ 148.8.  
    - GRU softplus: MAE ≈ 528.4, RMSE ≈ 1247.1, sMAPE ≈ 152.7.  
    - GRU log1p-head: MAE ≈ 518.2, RMSE ≈ 1228.8, sMAPE ≈ 96.0 (MAPE nhỏ nhưng RMSE lớn, phản ánh trade‑off không cân bằng trên tail).  
  - H=7, w=14:  
    - GRU raw: MAE ≈ 560.0, RMSE ≈ 1378.8, sMAPE ≈ 154.0.  
    - GRU softplus: MAE ≈ 627.4, RMSE ≈ 1356.3, sMAPE ≈ 153.3.  
    - GRU log1p-head: MAE ≈ 496.9, RMSE ≈ 1246.5, sMAPE ≈ 152.6.  
- Kết quả này phù hợp với kinh nghiệm thực tế: trên dữ liệu vừa và nhiều zero, **sequence model đơn giản (GRU) với MSE point loss không nhất thiết vượt được XGBoost lag‑based** nếu không có tuning sâu hoặc loss chuyên biệt cho intermittent demand. 

### 12.5. GNN với output raw cho kết quả kém hơn XGBoost và graph features

Với các GNN ở chế độ `target_transform = raw`:

- Heterogeneous GNN (`gnn_hetero5_h7_lag14_unit_raw`): MAE ≈ 402.3, RMSE ≈ 977.0, MAPE ≈ 296.9, sMAPE ≈ 135.6 – thua xgb_graph_hetero về cả MAE và sMAPE.  
- Homogeneous GNN (`gnn_homo5_h7_lag14_unit_raw`): MAE ≈ 776.9, RMSE ≈ 1518.3, MAPE ≈ 1338.7, sMAPE ≈ 149.2 – rõ ràng kém.  
- Projected GNN raw (same_subgroup/group/plant/storage) cũng cho RMSE cao (≈ 1731–1905) và sMAPE rất lớn (≈ 159–184), không cạnh tranh với xgb_graph.  

Những kết quả này cho thấy trong cấu hình hiện tại, **GNN với output linear + MSE trên demand zero‑inflated, heavy‑tailed không tận dụng được lợi thế graph** và thua mô hình tabular dùng graph features. Đây là tín hiệu nhất quán với các nghiên cứu nhấn mạnh rằng GNN cần loss/decoder chuyên biệt để xử lý zero‑inflated demand thay vì chỉ dùng MSE. 

### 12.6. Softplus và log1p trên GNN: vấn đề nằm ở loss/optimization hơn là kiến trúc

Với **Softplus output** cho GNN (projected, homo, hetero):

- Tất cả biến thể (lag 7 và 14, mọi graph type/view) hội tụ đến cùng một mức metric gần như identical:  
  - MAE ≈ 720.38, RMSE ≈ 1911.44, MAPE = 100, sMAPE ≈ 105.8–118.5.  
- Hành vi này cho thấy mô hình GNN bị kẹt ở nghiệm với dự báo gần như constant dương, nơi loss MSE không còn cải thiện thêm, bất chấp chúng tôi đã thử giảm learning rate, áp dụng gradient clipping và giới hạn miền log‑head trước `expm1` trong các biến thể có log1p nội bộ.  

Với **log1p mode** cho GNN:

- Nhiều run kết thúc với loss/metric NaN hoặc không tính được (các ô metric trống trong bảng), ngay cả khi clamp log‑head và dùng LR nhỏ.  
- Khi nới hoặc siết clamp để tránh NaN, mô hình lại rơi vào trạng thái tương tự Softplus – loss không giảm và dự báo constant – nên các kết quả log1p hiện tại **không được dùng để so sánh năng lực kiến trúc**.  

Trong paper, bạn có thể tóm tắt:

> Khi áp dụng Softplus hoặc log1p head cho GNN, chúng tôi quan sát loss thường xuyên rơi vào trạng thái bất ổn (NaN hoặc hội tụ đến nghiệm dự báo gần như constant) mặc dù đã thử giảm learning rate, áp dụng gradient clipping và giới hạn miền log-head. Điều này gợi ý rằng vấn đề nằm ở tương tác giữa phân phối demand zero-inflated, heavy-tailed và thiết kế loss MSE pointwise, hơn là do hạn chế vốn có của kiến trúc GNN. 
***

### 12.7. Ý nghĩa đối với thiết kế benchmark

Từ các kết quả trên, có thể rút ra ba thông điệp chính:

- **Baseline tabular rất mạnh**: XGBoost lag‑based với OHE categorical (no‑graph) đã đạt hiệu năng cao, và việc thêm graph features chỉ cải thiện vừa phải – đây là mốc chuẩn bắt buộc cho mọi phương pháp mới trong benchmark.  
- **Graph information hữu ích nhất khi kết hợp cẩn thận**: trong các thử nghiệm hiện tại, việc encode graph vào features cho XGBoost mang lại lợi ích rõ ràng hơn so với GNN end‑to‑end với MSE point loss, nhấn mạnh tầm quan trọng của hybrid design (graph + mạnh tabular).
- **Kiến trúc GNN cần đi kèm loss phù hợp với zero‑inflated demand**: các hiện tượng loss bất ổn với Softplus/log1p cho thấy rằng để khai thác graph triệt để, cần những thiết kế chuyên biệt (ví dụ hurdle/mixture likelihood, zero‑inflated count models) thay vì chỉ thay backbone bằng GNN với MSE cổ điển. 
