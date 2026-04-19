# Skill: Writing SCGraph-Bench Paper (6-page IEEE, Double-Blind)

## 1. Scope of This Skill

This skill describes how to write a **6-page IEEE conference paper** (double-blind) for the project:

> SCGraph-Bench: A Reproducible Benchmark for Graph Construction and Demand Forecasting in Supply Chains

It encodes:
- Target structure (sections, length per section).
- Tone and level of detail.
- How to integrate **graph construction**, **demand forecasting**, **GNNs**, and **benchmarking**.
- Citation practices (only strong, accepted venues except for the few preprints we *explicitly* build on, like SupplyGraph).

Any agent using this skill should be able to:
- Draft or revise sections of the paper.
- Keep writing consistent with the agreed storyline and constraints.
- Help shorten or expand text while preserving the core message.

---

## 2. Global Constraints

1. **Format**
   - IEEE conference format (`\documentclass[conference]{IEEEtran}`), two columns.
   - Max **6 pages total**, including figures, tables, and references.
   - Double-blind: no real author names, affiliations, or identifiable acknowledgments in the submission version.

2. **Double-blind rules**
   - Use `Anonymous Authors` and `Paper under double-blind review` in `\author`.
   - No acknowledgments section in the review version.
   - No self-identifying text: avoid “our lab at UIT”, “our industry partner”, etc.
   - Code/data release: mention “we release code and data in an anonymous repository” without linking; add actual link only at camera-ready.

3. **Citation policy**
   - Prefer **accepted** conference/journal papers (NeurIPS, ICLR, KDD, IEEE/ACM journals, top operations/marketing journals).
   - Exceptions:
     - Survey GNN4TS preprint (if needed) and **SupplyGraph** preprint, because benchmark is built around it.
   - No random arXiv unless:
     - It’s SupplyGraph itself, or
     - It’s a major survey where no peer-reviewed alternative exists.
   - Citations should be **specific and accurate**: statements tied to the actual content of the cited paper, not generic claims.

4. **Narrative priorities**
   - Paper is **not** “new model SOTA”; it is:
     - A **reproducible benchmark** (data + pipeline + protocol).
     - A carefully controlled empirical study of **graph construction** + **graph usage** + **output parameterization**.
   - Always foreground:
     - Transparency of pipeline.
     - Reproducibility of splits and evaluation.
     - Strong baselines (XGBoost) as serious references.
     - Domain specifics: zero-inflated, heavy-tailed demand; non-negativity constraints.

---

## 3. Target Paper Structure (6 Pages)

The paper should roughly follow this structure:

1. **Abstract**
   - 4–6 sentences.
   - Mention:
     - Supply chain demand forecasting is relational → graphs.
     - Existing work (SupplyGraph) lacks fully transparent, reproducible pipeline and protocol.
     - SCGraph-Bench: transparent preprocessing + constrained graph constructions + benchmark protocol.
     - Empirical findings: XGBoost strong, but heterogeneous GNN + Softplus brings gains in certain regimes.

2. **Introduction (~0.75–1 page)**
   - Flow:
     1. **Demand forecasting in supply chains**
        - Role in FMCG, bullwhip effect (Lee et al.).
        - Traditional models and ML; GBDT as strong baselines.
     2. **Interacting products and supply chain network**
        - Cross-effects between products (cross-elasticity, promotions).
        - Shared plants, warehouses, and logistics → correlated shocks.
        - This naturally defines a supply chain graph.
     3. **GNNs in supply chain**
        - GNNs for spatio-temporal forecasting.
        - Specific accepted works: graph attention for supply chain demand, DeepHGNN for hierarchical time series.
     4. **SupplyGraph**
        - First public graph-structured dataset for supply chain planning with GNNs.
        - Introduces multi-relational graph and initial GNN benchmarks.
     5. **Gap**
        - Pipeline from raw CSV → graph tensors not fully specified.
        - Temporal splits, labels, and hyperparameter tuning not standardized.
        - Hard to reproduce results or compare models fairly.
        - Contrast with OGB, GraphBench, TimeRecipe.
     6. **Contributions**
        - 3 bullets:
          - Transparent preprocessing and graph construction pipeline.
          - Reproducible benchmark protocol (splits, baselines, metrics).
          - Empirical study: graph-as-features vs graph-as-models + raw vs Softplus outputs on projected/homo/hetero graphs.

3. **Related Work (~0.5–0.75 page)**  
   Split into 3 subsections:

   ### 3.1 Demand Forecasting in Supply Chains
   - Briefly survey:
     - Traditional forecasting (ARIMA, exponential smoothing).
     - GBDT + hybrid TS models (cite 1–2 good 2024–2025 journal papers).
     - Deep models for demand: RNNs/TCN/etc. (1 strong journal).
   - Key points:
     - GBDT (e.g., XGBoost) remains very strong when combined with lag/calendar features.
     - Most works treat products/locations independently or in small groups; they do not fully exploit the supply chain graph.

   ### 3.2 GNNs for Time Series and Supply Chains
   - Two paragraphs:
     - GNNs for spatio-temporal TS (traffic, sensors); possibly use GNN4TS survey (ACM Computing Surveys / arXiv).
     - GNNs for supply chain / demand:
       - Graph attention-based forecasting (ACM TMIS or similar).
       - DeepHGNN in Expert Systems with Applications.
       - Other GNN applications in production planning, risk.
   - Key gap:
     - These papers propose specific models on proprietary datasets.
     - Lack of shared, standardized benchmark and baselines.

   ### 3.3 Graph and Time-Series Benchmarks
   - Describe:
     - OGB (NeurIPS 2020) → datasets + standardized splits + loaders.
     - GraphBench (KDD-level) → protocol design, hyperparameter search, threats to validity.
     - TimeRecipe (ICLR 2026) → module-level benchmarking for time-series.
   - Conclude:
     - No analogous benchmark yet for **graph-based supply chain demand forecasting**.
     - SupplyGraph is a dataset, but not a fully specified benchmark.
     - SCGraph-Bench is meant to fill this gap.

4. **The Proposed Benchmark Framework (~1–1.25 pages)**
   - **Design Principles**: transparency, standardization, domain-awareness.
   - **Data Pipeline & Temporal Splitting**:
     - From raw CSV (nodes/edges/temporal) → `base_raw_unit.parquet` → `base_full_unit.parquet` → GNN `.pt` packages.
     - Strict chronological splits; avoid leakage.
   - **Task Definition**:
     - Unit demand forecasting, horizon 7 days (H=7).
     - Lags 7 and 14.
     - Metrics: MAE, RMSE, MAPE, sMAPE.

5. **Controlled Study: Graph Construction and Modeling (~1–1.25 pages)**
   - **Graph Construction**:
     - Projected product graphs (4 views).
     - Homogeneous 5-type.
     - Heterogeneous 5-type.
   - **Graph-as-Features vs Graph-as-Models**:
     - Graph features → XGBoost.
     - End-to-end GNNs for each graph type.
   - **Output Constraints**:
     - Raw vs Softplus outputs for non-negativity on zero-inflated demand.

6. **Experiments and Results (~1.5–2 pages)**
   - **Setup**: hardware, software, hyperparameter search, splits.
   - **Leaderboard**: one main table (Lag-7, maybe Lag-14).
   - **Analysis**:
     - XGBoost vs GRU vs GNN.
     - Graph features vs GNN.
     - Hetero vs Homo vs Projected.
     - Raw vs Softplus: effect on MAPE/sMAPE.

7. **Discussion and Conclusion (~0.5–0.75 page)**
   - **Discussion / Threats to Validity**: single dataset, specific domain, some GNN configurations unstable at lag 14, etc.
   - **Limitations**: models/datasets not covered.
   - **Conclusion**: benchmark + insights; future directions.

---

## 4. Writing Style and Tone

- Audience: ML + supply-chain researchers at a reputable IEEE conference.
- Tone:
  - Precise, technical but readable.
  - Avoid hype; focus on clarity and rigor.
  - Highlight negative/neutral results honestly (e.g., “GRU did not outperform XGBoost under our setup”).

- Do:
  - Use clear topic sentences per paragraph.
  - Use bullet lists sparingly, mainly for contributions and key findings.
  - Keep math simple; only introduce formulas where they clarify (e.g., Softplus).

- Don’t:
  - Overclaim SOTA.
  - Use informal language or domain-specific slang.
  - Leak author identity or institution.

---

## 5. Citation Mapping (Canonical References)

When the paper needs to:

- **Motivate supply chain forecasting and bullwhip**:
  - Cite Lee et al., MIT Sloan Mgmt Rev 1997.

- **Discuss cross-product demand interactions**:
  - Use classic retail / promotion papers in Journal of Marketing Research.

- **Show GBDT strength for demand**:
  - Use 1–2 recent Expert Systems with Applications / IJ-type papers comparing GBDT vs DL for demand.

- **Show GNN success for supply chain demand**:
  - `Supply Chain Demand Forecasting via Graph Attention Networks` (ACM TMIS or equivalent).
  - `DeepHGNN` (Expert Systems with Applications, 2025).

- **Refer to SupplyGraph**:
  - Use its arXiv entry as the canonical dataset reference.

- **Connect to benchmarks**:
  - OGB (NeurIPS 2020).
  - GraphBench (KDD-level).
  - TimeRecipe (ICLR 2026).

The skill assumes these references are already gathered and available in a `.bib` file or `thebibliography` environment.

---

## 6. How to Use This Skill

When asked to “write” or “revise” a section of the SCGraph-Bench paper:

1. Identify which section (e.g., Introduction, Related Work, Experiments).
2. Use the structure and narrative described above to:
   - Decide what content must be present.
   - Respect length constraints.
3. Use only the agreed reference set (or add new ones if they are:
   - Accepted in strong venues,
   - And directly relevant).
4. Ensure the text remains anonymous and consistent with the global story:
   - Transparent benchmark,
   - Graph construction for supply chains,
   - GNN vs XGBoost,
   - Non-negativity and zero-inflation.

This skill is specific to the SCGraph-Bench paper and should not be applied blindly to unrelated topics.