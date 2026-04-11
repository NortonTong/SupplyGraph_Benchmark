import torch
import torch.nn as nn

class LSTMRegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, (h_n, c_n) = self.lstm(x)   # h_n: (num_layers, batch, hidden)
        last_hidden = h_n[-1]            # (batch, hidden)
        y_hat = self.fc(last_hidden).squeeze(-1)
        return y_hat
    
from math import sqrt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from config.config import PROC_DIR, LAG_WINDOWS
from gru_baseline import (
    GRUDataset, EarlyStopping,  # reuse dataset + early stopping
    load_baseline, build_sequences, make_dataloaders,
)

def train_one_lstm(
    horizon: int,
    lag_window: int,
    window: int,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    batch_size: int = 256,
    n_epochs: int = 200,
    lr: float = 1e-3,
    device: str | None = None,
    tag: str = "LSTM",
    patience: int = 20,
    min_delta: float = 0.0,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    df = load_baseline(horizon=horizon, lag_window=lag_window)

    (
        train_loader,
        val_loader,
        test_loader,
        feature_cols,
        (seq_len, input_size),
    ) = make_dataloaders(
        df=df,
        window=window,
        batch_size=batch_size,
    )

    model = LSTMRegressor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_rmse = float("inf")
    best_state = None
    early_stopper = EarlyStopping(patience=patience, min_delta=min_delta)

    print(
        f"\n=== Training LSTM H{horizon} lag{lag_window} ({tag}) "
        f"seq_len={seq_len}, n_features={input_size} ==="
    )

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_losses = []
        for Xb, yb in train_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            y_pred = model(Xb)
            loss = criterion(y_pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.to(device)
                yb = yb.to(device)
                y_pred = model(Xb)
                val_preds.append(y_pred.cpu().numpy())
                val_trues.append(yb.cpu().numpy())

        val_preds = np.concatenate(val_preds)
        val_trues = np.concatenate(val_trues)
        mask = np.isfinite(val_preds) & np.isfinite(val_trues)
        val_preds = val_preds[mask]
        val_trues = val_trues[mask]

        if len(val_trues) == 0:
            print("Warning: no valid validation samples after filtering NaNs.")
            return model, {
                "horizon": horizon,
                "lag_window": lag_window,
                "tag": tag,
                "val_rmse": np.nan,
                "test_rmse": np.nan,
                "test_mae": np.nan,
                "n_features": input_size,
                "seq_len": seq_len,
            }

        val_rmse = sqrt(mean_squared_error(val_trues, val_preds))
        val_mae = mean_absolute_error(val_trues, val_preds)

        print(
            f"Epoch {epoch:03d} | "
            f"TrainLoss={np.mean(train_losses):.4f} | "
            f"ValRMSE={val_rmse:.4f} | ValMAE={val_mae:.4f}"
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = model.state_dict()

        if early_stopper.step(val_rmse):
            print(f"Early stopping triggered at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test
    model.eval()
    test_preds, test_trues = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.to(device)
            y_pred = model(Xb)
            test_preds.append(y_pred.cpu().numpy())
            test_trues.append(yb.numpy())

    test_preds = np.concatenate(test_preds)
    test_trues = np.concatenate(test_trues)
    mask = np.isfinite(test_preds) & np.isfinite(test_trues)
    test_preds = test_preds[mask]
    test_trues = test_trues[mask]

    if len(test_trues) == 0:
        print("Warning: no valid test samples after filtering NaNs.")
        test_rmse = np.nan
        test_mae = np.nan
    else:
        test_rmse = sqrt(mean_squared_error(test_trues, test_preds))
        test_mae = mean_absolute_error(test_trues, test_preds)

    print(
        f"\n[LSTM][H{horizon}][lag{lag_window}][{tag}] "
        f"Test RMSE={test_rmse:.4f}, MAE={test_mae:.4f}"
    )

    out_dir = PROC_DIR / "models" / "lstm"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"lstm_h{horizon}_lag{lag_window}_{tag}.pth"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "horizon": horizon,
            "lag_window": lag_window,
            "window": window,
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "feature_cols": feature_cols,
        },
        model_path,
    )
    print(f"Saved LSTM model to {model_path}")

    info = {
        "horizon": horizon,
        "lag_window": lag_window,
        "tag": tag,
        "val_rmse": best_val_rmse,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "n_features": input_size,
        "seq_len": seq_len,
    }
    return model, info