# train_gru.py
import pandas as pd
from config.config import PROC_DIR, LAG_WINDOWS
from gru_baseline import train_one_gru, rolling_origin_evaluation_gru

def main():
    summaries = []

    for lag_w in LAG_WINDOWS:
        for h in (1, 7):
            # Train/val/test chuẩn
            _, info = train_one_gru(
                horizon=h,
                lag_window=lag_w,
                window=lag_w,
                hidden_size=128,
                num_layers=2,
                dropout=0.2,
                batch_size=256,
                n_epochs=200,
                lr=1e-3,
                tag=f"gru_lag{lag_w}",
                patience=20,
                min_delta=0.0,
            )
            summaries.append(info)

        # Rolling-origin cho H=7 (giống XGBoost)
        rolling_origin_evaluation_gru(
            horizon=7,
            lag_window=lag_w,
            window=lag_w,
            origins=[100, 150, 190],
            hidden_size=128,
            num_layers=2,
            dropout=0.2,
            batch_size=256,
            n_epochs=50,      # thường không cần quá nhiều epoch cho từng origin
            lr=1e-3,
            tag_prefix=f"gru_lag{lag_w}",
            patience=10,
            min_delta=0.0,
        )

    df_sum = pd.DataFrame(summaries)
    out_dir = PROC_DIR / "predictions" / "gru"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary_gru_baseline_lags.csv"
    df_sum.to_csv(out_path, index=False)
    print("\n=== GRU baseline summary ===")
    print(df_sum)
    print(f"\nSaved GRU summary to {out_path}")

if __name__ == "__main__":
    main()