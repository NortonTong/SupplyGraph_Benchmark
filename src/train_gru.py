import pandas as pd
from config.config import PROC_DIR, TEMPORAL_TYPE
from gru_baseline import (
    train_one_gru,
    rolling_origin_evaluation_gru,
    TARGET_TYPES,
)

def main():
    summaries = []
    horizon = 7
    windows = [7, 14]           # 2 lookback windows
    origins = [100, 150, 190]   # dùng cho rolling-origin

    for temporal_type in TEMPORAL_TYPE:  # ví dụ ["unit", "weight"]
        for window in windows:
            for target_type in TARGET_TYPES:  # ["raw", "log1p"]
                tag = f"gru_baseline2_h{horizon}_w{window}_{target_type}_{temporal_type}"
                print(
                    f"\n=== Fixed-split GRU baseline 2: "
                    f"H={horizon}, window={window}, target_type={target_type}, "
                    f"temporal_type={temporal_type} ==="
                )

                _, info = train_one_gru(
                    horizon=horizon,
                    window=window,
                    hidden_size=128,
                    num_layers=2,
                    dropout=0.2,
                    batch_size=256,
                    n_epochs=200,
                    lr=1e-3,
                    tag=tag,
                    patience=20,
                    min_delta=0.0,
                    temporal_type=temporal_type,
                    target_type=target_type,
                )
                info["temporal_type"] = temporal_type
                summaries.append(info)

            # Rolling-origin baseline (ở đây mình cho rolling với target_type="raw" để đơn giản;
            # nếu muốn so log1p thì gọi thêm lần nữa với target_type="log1p")
            print(
                f"\n=== Rolling-origin GRU baseline 2 (raw target): "
                f"H={horizon}, window={window}, temporal_type={temporal_type} ==="
            )
            rolling_origin_evaluation_gru(
                horizon=horizon,
                window=window,
                origins=origins,
                hidden_size=128,
                num_layers=2,
                dropout=0.2,
                batch_size=256,
                n_epochs=50,   # ít epoch hơn cho rolling để tiết kiệm
                lr=1e-3,
                tag_prefix=f"gru_baseline2_rolling_h{horizon}_w{window}",
                patience=10,
                min_delta=0.0,
                temporal_type=temporal_type,
                target_type="raw",   # hoặc "log1p" nếu muốn
            )

    # Tổng hợp kết quả fixed-split
    if summaries:
        df_sum = pd.DataFrame(summaries)
        df_sum = df_sum.sort_values(
            ["temporal_type", "window", "horizon", "target_type", "tag"]
        )

        print("\n=== GRU Baseline 2 (fixed split) summary ===")
        print(
            df_sum[
                [
                    "temporal_type",
                    "window",
                    "horizon",
                    "target_type",
                    "tag",
                    "n_features",
                    "seq_len",
                    "train_rmse",
                    "val_rmse",
                    "test_rmse",
                    "train_mae",
                    "val_mae",
                    "test_mae",
                ]
            ]
        )

        out_dir = PROC_DIR / "predictions" / "baseline_2" / "gru"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "summary_gru_baseline2_h7_windows7_14_targets_raw_log1p.csv"
        df_sum.to_csv(out_path, index=False)
        print(f"\nSaved GRU baseline 2 summary to {out_path}")

if __name__ == "__main__":
    main()