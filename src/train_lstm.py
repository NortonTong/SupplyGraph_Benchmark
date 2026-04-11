import pandas as pd
from config.config import PROC_DIR, LAG_WINDOWS
from lstm_baseline import train_one_lstm

def main():
    summaries = []
    for lag_w in LAG_WINDOWS:
        for h in (1, 7):
            _, info = train_one_lstm(
                horizon=h,
                lag_window=lag_w,
                window=lag_w,
                hidden_size=128,
                num_layers=2,
                dropout=0.2,
                batch_size=256,
                n_epochs=200,
                lr=1e-3,
                tag=f"lstm_lag{lag_w}",
                patience=20,
                min_delta=0.0,
            )
            summaries.append(info)

    df_sum = pd.DataFrame(summaries)
    out_dir = PROC_DIR / "predictions" / "lstm"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary_lstm_baseline_lags.csv"
    df_sum.to_csv(out_path, index=False)
    print("\n=== LSTM baseline summary ===")
    print(df_sum)
    print(f"\nSaved LSTM summary to {out_path}")

if __name__ == "__main__":
    main()