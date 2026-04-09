import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROC_DIR = DATA_DIR / "processed"
df = pd.read_csv(PROC_DIR / "predictions" / "arma_rolling" / "arma_rolling_h1_test_predictions.csv")
mae = mean_absolute_error(df["y_true"], df["y_pred"])
rmse = root_mean_squared_error(df["y_true"], df["y_pred"])
print("H=1 MAE:", mae, "RMSE:", rmse)

df7 = pd.read_csv(PROC_DIR / "predictions" / "arma_rolling" / "arma_rolling_h7_test_predictions.csv")
mae7 = mean_absolute_error(df7["y_true"], df7["y_pred"])
rmse7 = root_mean_squared_error(df7["y_true"], df7["y_pred"])
print("H=7 MAE:", mae7, "RMSE:", rmse7)