# SupplyGraph Experiment

This project provides data preprocessing, feature engineering, and model training scripts for supply chain forecasting using XGBoost, GNN, and TGCN models.

## Project Structure

- `src/` — Source code for preprocessing and training
- `data/` — Data folder (raw and processed)
- `src/config/config.py` — Path and parameter configuration

## Setup

1. **Clone the repository**
2. **Install dependencies**
   - Python 3.8+
   - Recommended: Create a virtual environment
   - Install required packages:
     ```bash
     pip install pandas numpy scikit-learn xgboost networkx
     ```
     (Add other packages as needed for GNN/TGCN)
3. **Prepare data**
   - Place raw data files in the `data/raw/` subfolders as structured in the repo.

## Preprocessing

Run the preprocessing script to generate processed datasets:

```bash
python src/preprocess_baseline.py
```

This will create processed files in `data/processed/`.

## Training XGBoost

Train XGBoost models using:

```bash
python src/train_xgboost.py
```

## Training GNN/TGCN

- For GNN: `python src/train_gnn.py`
- For TGCN: `python src/train_tgcn.py`

## Configuration

Edit `src/config/config.py` to adjust paths or parameters as needed.

## Output

- Processed data and model predictions are saved in `data/processed/`.

## Notes

- Ensure all required data files are present in the correct folders.
- For advanced usage (e.g., custom models), see the source code in `src/`.
