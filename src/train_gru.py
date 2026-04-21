import pandas as pd

from config.config import PROC_DIR, DEFAULT_EXPERIMENTS
from gru_baseline import train_one_gru


def main():
    summaries = []

    experiments = DEFAULT_EXPERIMENTS

    for exp in experiments:
        temporal_type = exp.temporal_type
        horizons = list(exp.horizons)
        lag_windows = list(exp.gru_seq_lengths)  # dùng gru_seq_lengths làm window
        is_softplus = exp.is_softplus
        is_log1p = exp.is_log1p

        horizon = horizons[0]

        transform_tag = "raw"
        if is_softplus:
            transform_tag = "softplus"
        elif is_log1p:
            transform_tag = "log1p-head"

        for window in lag_windows:
            tag = f"gru_baseline2_h{horizon}_w{window}_{transform_tag}_{temporal_type}"
            print(
                f"\n=== Fixed-split GRU baseline 2: "
                f"H={horizon}, window={window}, output={transform_tag}, "
                f"temporal_type={temporal_type} ==="
            )

            _, info = train_one_gru(
                horizon=horizon,
                window=window,
                hidden_size=128,
                num_layers=2,
                dropout=0.2,
                batch_size=256,
                n_epochs=400,
                lr=1e-3,
                tag=tag,
                patience=40,
                min_delta=0.0,
                temporal_type=temporal_type,
                is_softplus=is_softplus,
                is_log1p=is_log1p,
            )
            info["temporal_type"] = temporal_type
            summaries.append(info)

    # Tổng hợp kết quả fixed-split
    if summaries:
        df_sum = pd.DataFrame(summaries)
        df_sum = df_sum.sort_values(
            ["temporal_type", "window", "horizon", "output_transform", "tag"]
        )

        print("\n=== GRU Baseline 2 (fixed split) summary ===")
        print(
            df_sum[
                [
                    "temporal_type",
                    "window",
                    "horizon",
                    "output_transform",
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
        window_str = "_".join(str(w) for w in sorted({s["window"] for s in summaries}))
        transform_set = sorted({s["output_transform"] for s in summaries})
        transform_str = "_".join(transform_set)
        out_path = out_dir / f"summary_gru_baseline2_h{horizon}_windows_{window_str}_outputs_{transform_str}.csv"
        df_sum.to_csv(out_path, index=False)
        print(f"\nSaved GRU baseline 2 summary to {out_path}")


if __name__ == "__main__":
    main()