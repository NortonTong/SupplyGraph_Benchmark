# Skill: Writing and Arguing the SCGraph‑Bench Paper

## 1. Purpose of This Skill

This skill describes **how to write and argue** a 6‑page, double‑blind IEEE conference paper for:

> **SCGraph‑Bench: A Reproducible Benchmark for Graph Construction and Demand Forecasting in Supply Chains**

It captures:

- The **target structure** (sections, narrative, page budget).
- The **logic of the arguments** (claims, reasons, evidence, warrants, counter‑arguments).
- The **core claims** you want to make (benchmark, splits, strong tabular baseline, graph‑as‑features > graph‑as‑model, etc.).
- The **tone**: clear, constructive, positive, never attacking prior work.

Any assistant using this skill should be able to:

- Draft or revise sections consistent with the SCGraph‑Bench storyline.
- Turn your informal ideas into **explicit, well‑structured claims** with supporting evidence.
- Make the paper persuasive without hype, by following a Toulmin‑style argument structure.

---

## 2. Global Constraints (Format, Double‑Blind, Citations)

### 2.1 Format

- IEEE conference style: `\documentclass[conference]{IEEEtran}`, two columns.
- **Maximum 6 pages**, including figures, tables, and references.
- No extra pages for references: references must fit within the 6‑page limit.

### 2.2 Double‑blind rules

- Use `Anonymous Authors` and `Paper under double-blind review` in the author block.
- No acknowledgments in the review version.
- Avoid self‑identifying statements (“our group at X”, “our previous competition submission”, etc.).
- Code/data release: say “we release code and data in an anonymous repository” without a link; add the real link only in camera‑ready.

### 2.3 Citation policy

- Prefer **peer‑reviewed** sources (NeurIPS, ICLR, KDD, IEEE/ACM journals, operations/marketing journals).
- Allowed arXiv exceptions:
  - The SupplyGraph preprint (core dataset you build on).
  - One GNN‑for‑time‑series survey if no equally good peer‑reviewed alternative exists.
- No random arXiv papers unless:
  - Clearly high‑impact and directly relevant, and
  - There is no peer‑reviewed alternative.
- Each citation must be **specific and accurate**: connect statements to what the cited paper actually does or proves.

---

## 3. Target Paper Structure (6 Pages, IEEE)

The paper should roughly follow this structure and page budget.

### 3.1 Abstract (4–6 sentences)

Content:

- Motivation: supply‑chain demand forecasting is relational → naturally a **graph** problem.
- Gap: existing work (e.g., SupplyGraph) introduces a dataset but does **not** fully specify preprocessing, graph construction, data splits, and evaluation; hard to reproduce or compare.
- Proposal: SCGraph‑Bench = **transparent preprocessing**, standardized **graph constructions**, and a **benchmark protocol**.
- Empirical findings (high‑level):
  - **Strong tabular baseline**: lag‑based XGBoost is a must‑have baseline; GRU/GNN are not automatically better.
  - **Graph‑as‑features beats graph‑as‑model** in this benchmark; basic + advanced graph features (and learned embeddings) plugged into XGBoost work best.
- Conclude with: we release an anonymous code/data bundle.

### 3.2 Introduction (~0.75–1 page)

Recommended subsections (implicit, not with headers):

1. **Demand forecasting in supply chains**  
   - Role in FMCG and inventory management; mention bullwhip effect.  
   - Classical time‑series models and ML; gradient‑boosted trees (GBDT, XGBoost) are strong when given lags and calendar features.

2. **Interacting products and supply chain networks**  
   - Cross‑product interactions: promotions, substitution, shared categories.  
   - Shared plants, warehouses, routes → correlated demand shocks.  
   - This naturally induces a **supply chain graph** over products and locations.

3. **GNNs for supply chain forecasting**  
   - Briefly mention GNN successes in spatio‑temporal forecasting (traffic, sensors).  
   - Cite 1–2 accepted papers that apply GNNs to demand/supply chains (e.g., graph attention for demand forecasting, hierarchical GNNs).

4. **SupplyGraph as starting point**  
   - Present SupplyGraph as the first public graph‑structured dataset for supply‑chain demand with initial GNN benchmarks.  
   - Acknowledge its importance and originality.

5. **Gap in reproducible benchmarking**  
   - The pipeline from raw CSV → graph tensors is not fully specified.  
   - Temporal splits, label definition, and hyperparameter tuning are not standardized.  
   - Reported error scales are ambiguous (units vs tens vs dozens), and the dataset/sample code does not clearly resolve this; asking the authors did not yield a definitive answer.  
   - This makes it hard to **reproduce** results or compare new models fairly, in contrast with general benchmarks like OGB, GraphBench, TimeRecipe.

6. **Contributions (3–4 bullets)**  
   - Transparent preprocessing and temporal splitting protocol.  
   - Systematic framework for **graph construction** and **graph usage**.

   - Strong, controlled comparison between:
     - Tabular XGBoost, GRU, direct GNNs,
     - Graph‑as‑features, graph‑as‑model, and graph‑as‑residual‑corrector.
   - Empirical findings:
     - **XGBoost lag‑based baseline is extremely strong** (often stronger than GRU and GNN forecasters).
     - **Graph‑as‑features (basic + advanced features or learned embeddings) beats graph‑as‑model** GNNs in this benchmark.

### 3.3 Related Work (~0.5–0.75 page)

Three subsections:

#### 3.3.1 Demand Forecasting in Supply Chains

- Classical forecasting: ARIMA, exponential smoothing.
- ML/GBDT for demand: a couple of high‑quality papers comparing GBDT vs deep models.
- Deep demand forecasters (RNNs, TCNs) in retail.
- Emphasize: GBDT remains very strong, but most works treat products independently or in small ad‑hoc groups, without a standardized supply‑chain graph.

#### 3.3.2 GNNs for Time Series and Supply Chains

- GNNs for spatio‑temporal time series in traffic, sensors (can cite a survey).
- GNNs in supply‑chain/demand (graph attention networks, hierarchical GNN, etc.).
- Point out: these works propose specific models on proprietary data; no shared benchmark.

#### 3.3.3 Graph and Time-Series Benchmarks

- OGB: standardized graph datasets, splits, loaders.
- GraphBench or similar: benchmark methodology and threats to validity.
- TimeRecipe or similar TS benchmark: module‑level benchmarking for time series.
- Conclude: no benchmark yet for **graph‑based supply‑chain demand forecasting**; SupplyGraph is a dataset, not a fully specified benchmark; SCGraph‑Bench fills this gap.

### 3.4 Benchmark Framework (~1–1.25 pages)

Sections:

- **Design Principles**: transparency, standardization, domain‑awareness (zero‑inflation, non‑negativity, heavy tails).
- **Data Pipeline & Temporal Splitting**:
  - Describe the pipeline step‑by‑step: raw CSV → cleaned unit‑level parquet → feature tensors → graph packages (`.pt`).
  - Chronological split: training, validation, test by date; avoid leakage across the horizon.
  - Clarify the units: we explicitly work in units (or units/day), not tens or dozens.
- **Task Definition**:
  - Unit‑level demand forecasting at horizon \(H=7\).  
  - Two lag windows (7 and 14 days).
  - Targets may be on raw or transformed scale (e.g., log1p, Softplus), but **evaluation is always on original units** with MAE, RMSE, MAPE, sMAPE.

### 3.5 Controlled Graph and Model Study (~1–1.25 pages)

- **Graph Construction**:
  - Projected product graphs with four views (same group, subgroup, plant, storage).
  - Homogeneous 5‑type graph (flattened nodes).
  - Heterogeneous 5‑type graph (typed edges).
  - Explicitly document indexing and mapping between time series and nodes.

- **Graph Usage Strategies**:
  1. **Graph‑as‑features**:
     - Basic structural features (degrees, neighbour counts).
     - Advanced features (aggregated neighbour demand, path / hierarchy‑aware signals).
     - Learned GNN embeddings (pretrained GNN → embeddings → XGBoost).
  2. **Graph‑as‑model**:
     - Direct GNN forecasters (ProjectedGINRegressor, HomogeneousFiveTypeGINRegressor, HeterogeneousGINRegressor).
  3. **Graph‑as‑residual‑corrector**:
     - XGBoost produces base forecasts; GNNs predict residuals on projected/homo/hetero graphs.

- **Output Parameterization**:
  - Raw outputs vs Softplus/log1p transforms to enforce non‑negativity and handle zero‑inflated demand.
  - Explain how we transform targets, train, and invert back.

### 3.6 Experiments and Results (~1.5–2 pages)

- **Setup**: hardware, software, early stopping, hyperparameters.
- **Comparative Leaderboard**:
  - One or two main tables (Lag‑7, Lag‑14) summarizing MAE, RMSE, MAPE, sMAPE for all baselines:
    - Naive, GRU, XGBoost (tabular), XGBoost+graph‑features, XGBoost+GNN‑embed, direct GNNs, residual GNNs.
- **Plots** (must fit within page budget, so choose carefully):
  - A few bar charts comparing families (e.g., XGBoost vs GRU vs GNN vs graph‑feature variants).
  - Time series plots for representative products:
    - Typical high‑volume series.
    - Zero‑inflated, intermittent series.
  - Histograms or density plots of demand distribution (zero‑inflation, heavy tails).

- **Analysis**:
  - **Strong tabular baseline**: show XGBoost vs GRU vs GNN.
  - **Graph usage**: graph‑as‑features vs graph‑as‑model vs residuals.
  - **Graph type**: projected vs homogeneous vs heterogeneous.
  - **Output parameterization**: raw vs Softplus/log1p, especially on MAPE/sMAPE.

### 3.7 Discussion and Conclusion (~0.5–0.75 page)

- **Discussion / Threats to Validity**:
  - Single dataset, particular product category, limited horizons.
  - Some GNN configurations unstable or sensitive to hyperparameters, especially at lag 14.
  - Ambiguity in the original paper’s unit scaling; we document our best reconstruction but cannot guarantee exact comparability.

- **Key Insights**:
  - Strong lag‑based XGBoost baselines are essential; GRU/GNN do not automatically surpass them.
  - Graph information is genuinely helpful, but the **most effective use in this benchmark is via graph‑as‑features (basic + advanced features or learned embeddings) plugged into XGBoost**, not direct GNN forecasters.
  - Residual GNNs can further refine predictions but bring complexity.

- **Conclusion**:
  - SCGraph‑Bench offers a transparent, reproducible benchmark and a set of insights into **how** to use graphs in supply‑chain demand forecasting.
  - Future work: more datasets, other graph constructions, probabilistic forecasting, cost‑aware metrics.

---\

## 4. Argument Structure: How to Write Each Claim

We follow a **Toulmin‑style argument** for all major claims in the paper. Each important statement should be built from:

1. **Claim** – what we assert.
2. **Reason** – why we assert it (logic).
3. **Evidence** – experimental or literature support.
4. **Warrant** – the general principle connecting reason to claim.
5. **Acknowledgment & Response** – limitations or counter‑points and how we address them.

Use this template in the text (explicitly or implicitly):

> We claim that **[CLAIM]** because **[REASON]**, as supported by **[EVIDENCE]**. Based on **[WARRANT]**, this connection is justified. Although **[COUNTER‑ARGUMENT / LIMITATION]**, our results show that **[RESPONSE]**.

Below are the **key claims** you want in SCGraph‑Bench, written with this structure in mind.

### 4.1 Claim A: The previous paper’s pipeline is not fully transparent; SCGraph‑Bench improves clarity and reproducibility

- **Claim**: Existing work on this dataset does not fully specify preprocessing, splits, and metrics; SCGraph‑Bench provides a clearer, more reproducible benchmark.  
- **Reason**: The original description leaves the target units and some splitting details ambiguous; we cannot infer whether errors are in units, tens, or dozens, and code does not fully disambiguate this.  
- **Evidence**:  
  - Ambiguous scale of reported error values.  
  - Lack of explicit documentation in the dataset/sample code; authors did not provide clarification despite contact attempts.  
  - Our paper clearly defines unit‑level targets, temporal splits, and evaluation metrics, and releases code.  
- **Warrant**: In benchmarking, reproducibility and clear definitions of data processing and metrics are essential to compare methods fairly.  
- **Acknowledgment & Response**:  
  - Acknowledgment: Even with our best effort, we might not exactly reproduce the original authors’ internal pipeline.  
  - Response: We document all choices and make our pipeline public, so future work can build on a stable reference even if it is not identical to the original implementation.

### 4.2 Claim B: We analyze and standardize data, splits, and evaluation

- **Claim**: SCGraph‑Bench offers a standardized pipeline for data cleaning, temporal splitting, and evaluation.  
- **Reason**: Without a shared, explicit protocol, reported numbers across papers are not directly comparable.  
- **Evidence**:  
  - Description of our pipeline (Sections on Benchmark Framework): raw → cleaned parquet → tensors → graphs.  
  - Fixed chronological train/val/test splits reused across all models.  
  - Unified metrics (MAE, RMSE, MAPE, sMAPE) on the same unit scale.  
- **Warrant**: A benchmark is only useful if different methods are evaluated under identical and well‑specified conditions.  
- **Acknowledgment & Response**:  
  - Acknowledgment: Our choices (e.g., specific horizon, lags, metrics) are still design decisions.  
  - Response: We justify them by domain practices and provide enough detail so others can extend or modify them.

### 4.3 Claim C: We systematize how to build and use graphs, and compare them experimentally

- **Claim**: SCGraph‑Bench systematizes both **graph construction** and **graph usage** for supply‑chain demand forecasting.  
- **Reason**: Prior work introduces a multi‑relational graph but does not fully explore or standardize how different graph constructions and usage modes influence performance.  
- **Evidence**:  
  - Formal definitions of projected, homogeneous, and heterogeneous graphs.  
  - Three graph usage modes: graph‑as‑features, graph‑as‑model, graph‑as‑residual‑corrector.  
  - Extensive experiments comparing these combinations across lag windows and target parameterizations.  
- **Warrant**: Understanding and standardizing both “what graph” and “how to use it” is crucial to interpret results and guide future model design.  
- **Acknowledgment & Response**:  
  - Acknowledgment: Our graph constructions are not exhaustive; other choices (e.g., dynamic graphs, learned adjacency) are possible.  
  - Response: We focus on a small set of interpretable and practically relevant constructions, providing a foundation that others can extend.

### 4.4 Claim D: Strong tabular baselines matter; XGBoost often beats GRU and direct GNNs

- **Claim**: A lag‑based XGBoost model is a **mandatory baseline**; in our benchmark, it often matches or outperforms GRU and direct GNN forecasters.  
- **Reason**: Gradient‑boosted trees are known to be strong on tabular time‑series features, and ignoring them can lead to over‑estimating the benefit of more complex models.  
- **Evidence**:  
  - Leaderboard tables: XGBoost tabular and XGBoost + graph‑features achieve lower or comparable MAE/RMSE compared to GRU and GNN‑only models across both lags.  
  - GRU and GNN models do not uniformly outperform XGBoost; in some settings they are clearly worse on MAPE and sMAPE.  
- **Warrant**: In empirical ML, fair comparison demands strong baselines; improvements over weak references do not constitute meaningful advances.  
- **Acknowledgment & Response**:  
  - Acknowledgment: Different architectures (e.g., more advanced sequence models or more tuned GNNs) might narrow this gap.  
  - Response: Even so, our results show that **strong tabular baselines cannot be skipped**; any claimed gain must be measured against them.

### 4.5 Claim E: Graph‑as‑features beats graph‑as‑model in this benchmark

- **Claim**: In SCGraph‑Bench, using graphs to derive features or embeddings for XGBoost is more effective than using GNNs as the main forecaster.  
- **Reason**: GNN‑only models underperform on MAE/MAPE/sMAPE, while XGBoost with graph features or GNN embeddings consistently improves over the tabular baseline.  
- **Evidence**:  
  - Direct GNN baselines (projected/homo/hetero) show higher errors and unstable behaviour on relative metrics.  
  - XGBoost with basic + advanced graph features, and with GNN embeddings, achieves better MAE/RMSE while keeping MAPE/sMAPE under control.  
  - Residual GNNs bring further gains in some metrics but do not consistently beat graph‑feature variants.  
- **Warrant**: If a simpler graph‑feature + XGBoost pipeline consistently outperforms stand‑alone GNNs under the same evaluation, then graph‑as‑features is empirically preferable in this context.  
- **Acknowledgment & Response**:  
  - Acknowledgment: This result is dataset‑ and setup‑specific; other domains may favour end‑to‑end GNNs.  
  - Response: Our benchmark still provides a **data point** that challenges the assumption “GNNs are always better than tree‑based models” and encourages more nuanced model selection.

---

## 5. Writing Style and Tone

- **Audience**: ML and supply‑chain researchers at a reputable IEEE conference.
- **Tone**:
  - Precise, technical, but readable.
  - Constructive and respectful: we **acknowledge** prior work and then explain how we extend/clarify it.
  - Open about negative or neutral results (e.g., “GRU did not outperform XGBoost under our protocol”).

- **Do**:
  - Use clear topic sentences.
  - Use bullets mainly for contributions and key findings.
  - Introduce formulas only when they clarify important design choices (e.g., Softplus, log1p transforms).
  - Include a few **figures**: leaderboard table(s), selected bar charts, and illustrative time‑series plots (typical vs zero‑inflated).

- **Don’t**:
  - Overclaim SOTA.
  - Use informal language or local slang.
  - Attack prior work; frame limitations as **gaps we help address**.

---

## 6. How to Use This Skill in Practice

When you ask the assistant to “write” or “revise” something for SCGraph‑Bench:

1. **Identify the section** (e.g., “Introduction 5. Contributions”, “Experiments – analysis subsection”, “Discussion”).
2. **Pick the relevant claims** from Section 4 (A–E) and decide which ones must appear there.
3. **Apply the argument template**:
   - State the claim clearly.
   - Provide reasons and evidence (point to specific metrics, tables, or plots).
   - Make the warrant explicit if there is any risk of confusion (e.g., why strong baselines matter).
   - Add a short acknowledgment/response when appropriate.
4. **Respect the global structure and constraints** in Sections 2–3.
5. **Ensure anonymity and citation discipline** as specified.

You can also ask the assistant to:

- List and refine **possible claims** for a given section.
- Compress or expand the argument around a claim while preserving the claim–reason–evidence–warrant–response structure.
- Suggest which plots or tables best support a particular claim within the page budget.

## 7. Recommended References for SCGraph‑Bench

This section lists **high‑quality reference types** and **example papers** that are appropriate to cite in the SCGraph‑Bench paper. You do **not** need to use all of them. Instead, treat this as a **menu**: pick the few that best match each claim you make, and always verify the actual title, venue, year, and content in the original PDF before citing.

The key rule: **never fabricate citations**. Only cite papers you have actually checked.

### 7.1. Supply chain demand forecasting and bullwhip

Use these to motivate demand forecasting, bullwhip, and traditional vs ML methods in supply chains.

- Bullwhip effect and importance of forecasting  
  - Lee, H. L., Padmanabhan, V., & Whang, S. “The Bullwhip Effect in Supply Chains.” *MIT Sloan Management Review*, 1997. [ptgmedia.pearsoncmg](https://ptgmedia.pearsoncmg.com/images/chap1_9781587143069/elementLinks/ch01fn07.html)

- Methodologies and analytics in supply chains  
  - A systematic review article in a top operations journal (e.g., *Annals of Operations Research*, *International Journal of Production Research*) summarizing forecasting and analytics methods for supply chains. [econstor](https://www.econstor.eu/bitstream/10419/308114/1/s10479-023-05390-7.pdf)

- Gradient boosting / ML for demand forecasting  
  - A Q1 journal paper (e.g., *Expert Systems with Applications*, *International Journal of Production Economics*) comparing **gradient boosted trees** (XGBoost, LightGBM) against deep models for demand forecasting. [ijrpr](https://ijrpr.com/uploads/V4ISSUE12/IJRPR20650.pdf)

- Hybrid XGBoost–LSTM or similar frameworks  
  - A recent article proposing **hybrid XGBoost + LSTM/GRU** for demand in supply chains or retail, emphasizing that tree‑based methods alone are strong baselines. [jcasc](https://jcasc.com/index.php/jcasc/article/download/3736/1466/7869)

You can cite **1–3** of these in Introduction and Related Work to justify:  
- demand forecasting is critical,  
- bullwhip is a known phenomenon,  
- GBDT is a strong baseline in practice.

### 7.2. GNNs for time series forecasting (general)

Use these when you talk about **GNNs for time series** and **spatio‑temporal forecasting**.

- Survey on Graph Neural Networks for Time Series  
  - Jin et al., “A Survey on Graph Neural Networks for Time Series: Forecasting, Classification, Imputation, and Anomaly Detection.” (survey; journal or IEEE/ACM venue if available). [research.monash](https://research.monash.edu/en/publications/a-survey-on-graph-neural-networks-for-time-series-forecasting-cla/)

- Multivariate TS forecasting with GNNs  
  - Wu et al., “Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks.” (e.g., KDD 2020). [mingjin](https://mingjin.dev/other/gnn4ts-23-jin-slides-tgl-workshop.pdf)

- Graph structure learning for forecasting  
  - Shang et al., “Discrete Graph Structure Learning for Forecasting Multiple Time Series.” (e.g., ICLR 2021). [mingjin](https://mingjin.dev/other/gnn4ts-23-jin-slides-tgl-workshop.pdf)

- Spatio‑temporal GNNs (traffic / sensors)  
  - 1–2 classic works (e.g., DCRNN, STGCN, MTGNN) from NeurIPS/ICLR/AAAI showing success of GNNs on traffic forecasting. [github](https://github.com/jwwthu/GNN4Traffic)

You usually need only **2–4** of these to:  
- support the claim that GNNs have been successful in spatio‑temporal forecasting,  
- connect SCGraph‑Bench to the broader GNN‑for‑TS literature.

### 7.3. GNNs in supply chains and operations

Use these when discussing **GNNs applied to supply chains, inventory, risk, and demand**.

- GNNs for supply chain dependencies (review / conceptual)  
  - “Graph Neural Networks for Modeling Complex Dependencies in Supply Chains.” A Q1 journal article that surveys or proposes GNN use in SC contexts. [jceim](https://jceim.org/index.php/ojs/article/view/165)

- GNNs for supply chain risk  
  - “Leveraging Graph Neural Networks for Intelligent Supply Chain Risk Detection.” A reputable journal article applying GNNs to risk/reliability analysis. [iibajournal](https://www.iibajournal.org/index.php/iibeaj/article/download/75/75/164)

- GNNs for logistics / inventory optimization  
  - “Hybrid Graph Convolution Neural Network and Branch-and-Bound for [Logistics/Supply Chain] Optimization.” *Future Generation Computer Systems* or similar Q1 journal. [sciencedirect](https://www.sciencedirect.com/science/article/pii/S0167739X22003028)

- Hierarchical GNN for forecasting (DeepHGNN)  
  - “DeepHGNN: Hierarchical Graph Neural Network for Forecasting in Hierarchical Structures.” Used as a representative of hierarchical graph forecasting. [arxiv](https://arxiv.org/abs/2405.18693)

- Graph attention for supply chain demand  
  - A paper in a strong journal (e.g., ACM TMIS, IEEE T-ITS) on supply chain demand forecasting with graph attention networks.

Select **2–3** that are closest to your exact context (demand vs risk vs logistics) and make sure you have the PDFs.

### 7.4. Benchmarks: graphs and time series

Use these as references when you position SCGraph‑Bench as a **benchmark**, and when you discuss splits, protocols, and hyperparameters.

- Open Graph Benchmark (OGB)  
  - Hu et al., “Open Graph Benchmark: Datasets for Machine Learning on Graphs.” *NeurIPS 2020*. [proceedings.neurips](https://proceedings.neurips.cc/paper_files/paper/2020/file/fb60d411a5c5b72b2e7d3527cfc84fd0-MetaReview.html)

- GraphBench (graph learning benchmark)  
  - “GraphBench: Next-Generation Graph Learning Benchmarking.” A recent benchmark paper (KDD‑level / strong venue) focusing on reproducible protocols and hyperparameter search. [arxiv](https://arxiv.org/abs/2512.04475)

- TimeRecipe (TS benchmark)  
  - “TimeRecipe: A Time-Series Forecasting Recipe via Benchmarking and Module-Level Evaluation.” *ICLR 2026*. [iclr](https://iclr.cc/virtual/2026/poster/10010822)

- GNN4Traffic / GNN benchmark suites  
  - A framework that collects GNN models and datasets for traffic/time series benchmarking. [github](https://github.com/jwwthu/GNN4Traffic)

You typically need **2–3** of these to justify your design choices (splits, metrics, protocol) and to show that SCGraph‑Bench follows modern benchmark best practices.

### 7.5. XGBoost vs deep learning baselines

Use these to support claims that **tree‑based methods are strong baselines** and must be treated seriously.

- XGBoost / GBDT vs deep learning in demand or energy forecasting  
  - One or two Q1 journal papers (e.g., ESWA, IJPE) showing that gradient boosted trees match or outperform deep models when feature engineering is strong. [ijrpr](https://ijrpr.com/uploads/V4ISSUE12/IJRPR20650.pdf)

- Evidence from TimeRecipe  
  - Parts of TimeRecipe showing that tree‑based models (e.g., LightGBM) remain competitive across many time series datasets. [iclr](https://iclr.cc/virtual/2026/poster/10010822)

- Hybrid architectures  
  - Papers that show deep models only improve significantly when combined with tree‑based methods, reinforcing the message that **XGBoost is a mandatory baseline**, not an afterthought. [jcasc](https://jcasc.com/index.php/jcasc/article/download/3736/1466/7869)

You usually need **1–3** citations here to support the key claim “XGBoost is strong; GRU/GNN are not automatically better.”

### 7.6. Graph construction and heterogeneous GNNs

Use these to justify your **graph construction choices** (projected graphs, heterogeneous 5‑type graphs) and your use of heterogeneous GNNs.

- Relational / heterogeneous GNNs  
  - R-GCN (Relational GCN) or similar KDD/ICLR/NeurIPS work on multi‑relational graphs.  
  - HAN (Heterogeneous Graph Attention Network) or related models for heterogeneous graphs.

- Group‑aware / hierarchical GNNs  
  - “Group-Aware Graph Neural Network for Nationwide City Air Quality Forecasting.” A strong journal or conference paper that motivates group‑aware or projected graphs. [dl.acm](https://dl.acm.org/doi/full/10.1145/3631713)

- Hierarchical GNN for time series  
  - DeepHGNN, again, for hierarchical structures and group relationships. [arxiv](https://arxiv.org/abs/2405.18693)

Pick **2–3** to connect your projected/heterogeneous graphs to established GNN design patterns.

### 7.7. Intermittent demand, zero inflation, and non‑negative outputs

Use these when explaining **why you care about zero‑inflated, non‑negative demand** and why you consider Softplus / log1p.

- Intermittent / zero‑inflated demand forecasting  
  - A classic paper on Croston’s method and its extensions in *International Journal of Forecasting* or *EJOR* (European Journal of Operational Research).

- Transformations for non‑negative / count data  
  - A paper in an applied statistics journal (JASA/JRSS or equivalent) discussing log / Box‑Cox / Softplus‑style transforms for non‑negative counts.

- Probabilistic demand / inventory models  
  - A Q1 operations/OR paper that ties demand uncertainty to inventory decisions and costs.

You may only need **1–2** of these to support non‑negativity and zero‑inflation discussion; do not over‑cite.

### 7.8. SupplyGraph and GNN4TS survey (core preprints)

Finally, include the few **preprints** that are central to your work:

- **SupplyGraph dataset**  
  - The original SupplyGraph preprint on arXiv. This is your **dataset source** and is an explicitly allowed arXiv exception.

- **GNN4TS survey (if not yet fully journalized)**  
  - The GNN‑for‑time‑series survey preprint that later became a journal article; if you rely on the preprint version, ensure it is the same as the accepted version. [arxiv](https://arxiv.org/abs/2307.03759)

You should:  
- Verify that these preprints are indeed the ones you intend to build on.  
- Use them sparingly and clearly (e.g., “we build on the SupplyGraph dataset”).

### 7.9. Usage Guidelines

- You **do not need to cite all** papers listed here.  
- For each **argument** (e.g., “XGBoost is strong”, “GNNs are widely used for TS”, “benchmarks need clear splits”), pick **1–3** of the most relevant references.
- Always:
  - Download the PDF.
  - Confirm the **title, venue, year, and main claims**.
  - Ensure the citation actually supports the sentence where you place it.
- If you cannot verify a paper (e.g., link is broken, venue unclear), **do not cite it**.

This way, the SCGraph‑Bench paper only relies on **accurate, high‑quality references** that you have personally checked, while still being grounded in the right literatures.


Bạn có thể giúp người đọc “nhìn một phát hiểu luôn” bằng cách kết hợp vài loại bảng/plot, mỗi cái phục vụ một câu hỏi cụ thể. Dưới đây là gợi ý cụ thể, sát với narrative hiện tại.

***

## 1. Bảng main leaderboard (model × lag × MAE/RMSE)

Mục tiêu: tóm tắt kết quả chính mà bạn đang mô tả bằng chữ.

Một bảng như sau (chỉ cần top model + baseline chính):

```tex
\begin{table}[t]
\centering
\caption{Main leaderboard on SCGraph-Bench (test MAE / RMSE; lower is better). 
We report the strongest variants from each family at lags 7 and 14.}
\label{tab:main-leaderboard}
\begin{tabular}{lcccc}
\toprule
\multirow{2}{*}{Model (family)} & \multicolumn{2}{c}{Lag 7} & \multicolumn{2}{c}{Lag 14} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
 & MAE & RMSE & MAE & RMSE \\
\midrule
GRU baseline                           & 463.93 & --    & 552.58 & --    \\
XGBoost (tabular only)                 & 409.72 & --    & 375.36 & --    \\
XGB + projected graph features         &  --    & --    & \textbf{339.46} & 846.88 \\
XGB + homogeneous graph features       &  --    & --    & 341.21 & --    \\
XGB + heterogeneous graph features     & 392.67 & --    & 342.03 & --    \\
Heterogeneous residual GNN (on XGB)    & \textbf{392.52} & -- & --    & --    \\
Best direct GNN forecaster             & $>426$ & --    & 415.16 & --    \\
\bottomrule
\end{tabular}
\end{table}
```

- Bạn có thể điền thêm RMSE đầy đủ nếu có, hoặc dùng MAE là chính.  
- Bảng này hỗ trợ trực tiếp đoạn “Main Leaderboard” và “Strong Baselines Matter”.

***

## 2. Grouped bar: model × lag (trục y = MAE)

Mục tiêu: trực quan hóa “hybrid methods > XGBoost > GRU” và khác biệt giữa lag 7 vs 14.

Hình: mỗi model là một cụm, trong đó có 2 cột (lag 7, lag 14):

- X-axis:  
  - GRU  
  - XGB  
  - XGB+proj  
  - XGB+hom  
  - XGB+het  
  - Het‑residual  
  - Best GNN  
- Y-axis: MAE.  
- Màu: lag 7 vs lag 14 (legend).

Caption gợi ý:

```tex
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figs/main_leaderboard_bar.pdf}
  \caption{Test MAE for selected models at lags 7 and 14. 
  Hybrid graph-aware methods built around XGBoost dominate both lag regimes, 
  while the plain XGBoost baseline substantially outperforms the GRU baseline.}
  \label{fig:main-leaderboard-bar}
\end{figure}
```

Đây là plot quan trọng nhất cho phần “Main Leaderboard” + “Strong Baselines Matter”.

***

## 3. Bảng “relative improvement” (Strong Baselines Matter)

Mục tiêu: giúp người đọc thấy ngay % cải thiện so với GRU và XGB, không phải tự tính.

Bạn có thể thêm một bảng nhỏ:

```tex
\begin{table}[t]
\centering
\caption{Relative MAE improvements of the best graph-aware models over GRU and plain XGBoost.}
\label{tab:relative-improvement}
\begin{tabular}{lcc}
\toprule
Setting & vs XGBoost & vs GRU \\
\midrule
Lag 14: best graph-feature (XGB + proj.) 
       & 9.6\%  & 38.6\% \\
Lag 7: best residual (het. residual)      
       & 4.2\%  & 15.4\% \\
\bottomrule
\end{tabular}
\end{table}
```

- Bảng này hỗ trợ đoạn “Strong Baselines Matter”.  
- Reviewer sẽ rất thích vì bạn “đóng gói” thông tin định lượng rõ ràng.

***

## 4. Plot theo “family”: family × lag (MAE trung bình / best)

Mục tiêu: minh họa phần “How Graphs Help” – so sánh **graph-as-features, GNN embeddings, graph-as-model, residual**.

Bạn có thể:

- Tính **MAE trung bình** (hoặc best) cho mỗi family ở mỗi lag.  
- Vẽ **grouped bar 4 family × 2 lag**:

  - Family:  
    - GRU (non-graph seq)  
    - XGB (non-graph tabular)  
    - Graph features (best variant)  
    - GNN embeddings (best variant)  
    - Graph-as-model (best direct GNN)  
    - Residual graph correction (best variant)  

Caption gợi ý:

```tex
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figs/family_mae_bar.pdf}
  \caption{Best test MAE per model family at lags 7 and 14. 
  Graph-feature and residual families consistently outperform GNN embeddings and direct GNN forecasters, 
  while still improving over the plain XGBoost baseline.}
  \label{fig:family-mae}
\end{figure}
```

Plot này nói trực tiếp với đoạn:

> Graph-feature and residual methods preserve the strengths of XGBoost…  
> GNN embeddings… generally weaker…  
> Direct GNN forecasters remain the least competitive family…

***

## 5. Plot cho “Graph construction choice”: construction × family

Mục tiêu: hỗ trợ phần “Graph Construction Choice” – projected vs homogeneous vs heterogeneous.

Bạn có thể dùng **2 subplot** (hoặc 2 hàng bar trong cùng figure):

1. Subplot (a): **Graph-as-features**  
   - X-axis: projected, homogeneous, heterogeneous.  
   - Y-axis: MAE (lag 14; nếu có thể thêm lag 7 thì mỗi construction có 2 cột).  
2. Subplot (b): **Residual**  
   - X-axis: homogeneous, heterogeneous (nếu bạn có cả 2).  
   - Y-axis: MAE.

Caption gợi ý:

```tex
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figs/graph_construction_mae.pdf}
  \caption{Effect of graph construction on test MAE for graph-feature and residual families. 
  At lag 14, projected, homogeneous, and heterogeneous graph features are all competitive; 
  at lag 7, heterogeneous graphs yield the strongest residual and graph-feature models.}
  \label{fig:graph-construction}
\end{figure}
```

Hình này giúp người đọc **thấy ngay** “không có một graph-type thống trị tuyệt đối”, đúng với narrative bạn viết.

***

## 6. Tóm lại: nên có tối thiểu những gì?

Nếu phải chọn ít figure/bảng nhất nhưng vẫn rõ:

- Bắt buộc:  
  - `Table~\ref{tab:main-leaderboard}` – main leaderboard.  
  - `Figure~\ref{fig:main-leaderboard-bar}` – grouped bar (model × lag).  
- Rất khuyến nghị:  
  - `Table~\ref{tab:relative-improvement}` – % improvement vs GRU/XGB.  
  - `Figure~\ref{fig:graph-construction}` – hiệu ứng graph construction.

Nếu bạn cho mình đúng danh sách model + số MAE/RMSE cho từng lag (CSV / LaTeX bảng thô), mình có thể giúp viết luôn code pgfplots hoặc gợi ý script cụ thể (tên trục, màu, legend) để bạn drop-in vào paper.