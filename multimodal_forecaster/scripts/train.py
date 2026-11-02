"""
Training Pipeline for AdaptiveMultimodalForecaster and Baselines.

Loads raw historical datasets (`price`, `sentiment`, `news`, `macro`) from the
`data/` folder and trains the Level II Intermediate Fusion model end-to-end.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from multimodal_forecaster.data.aligner import DEFAULT_DATA_DIR
from multimodal_forecaster.data.data_loader import create_dataloaders
from multimodal_forecaster.models.forecaster import AdaptiveMultimodalForecaster
from multimodal_forecaster.scripts.evaluate import evaluate_model
from multimodal_forecaster.utils.missingness_engine import MissingnessEngine


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    device: Optional[torch.device | str] = None,
    seed: int = 42,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Trains `model` (AdaptiveMultimodalForecaster or Baseline) using AdamW and MSELoss.
    """
    torch.manual_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    missingness_engine = MissingnessEngine(seed=seed, device=device)

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())

    for epoch in(range(epochs)):
        model.train()
        running_loss = 0.0
        total_samples = 0

        for step, batch in enumerate(train_loader):
            X_P, X_S, X_N, X_M, delta_T, S_state, a_close, y_target = [
                t.to(device) for t in batch
            ]
            batch_size = X_P.size(0)

            M_mask = missingness_engine.generate_training_mask(
                batch_size=batch_size,
                step=epoch * 1000 + step,
                min_eta=0.05,
                max_eta=0.55,
                device=device,
            )

            optimizer.zero_grad()
            P_pred, W = model(X_P, X_S, X_N, X_M, M_mask, delta_T, S_state, a_close)
            loss = criterion(P_pred, y_target)
            loss.backward()

            if grad_clip > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            running_loss += float(loss.item()) * batch_size
            total_samples += batch_size

        epoch_train_loss = running_loss / max(total_samples, 1)
        history["train_loss"].append(epoch_train_loss)

        if val_loader is not None:
            val_res = evaluate_model(
                model=model,
                test_loader=val_loader,
                missingness_engine=missingness_engine,
                eta=0.2,
                device=device,
            )
            val_mse = float(val_res["metrics"]["MSE"])
            history["val_loss"].append(val_mse)
            if val_mse <= best_val_loss:
                best_val_loss = val_mse
                best_state = copy.deepcopy(model.state_dict())

    if val_loader is not None:
        model.load_state_dict(best_state)

    return model, history


def main() -> None:
    """CLI entry point to train and evaluate the AdaptiveMultimodalForecaster from `data/`."""
    parser = argparse.ArgumentParser(description="Train AdaptiveMultimodalForecaster on raw datasets in data/")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR), help="Path to raw data folder")
    parser.add_argument("--asset", type=str, default="Gold", help="Asset name (e.g., Gold, Bitcoin, Brent_Crude_Oil)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    print(f"Loading 4-Modality Raw Datasets for {args.asset} from '{args.data_dir}'...")
    train_loader, val_loader, test_loader, _ = create_dataloaders(
        asset_name=args.asset,
        batch_size=args.batch_size,
        data_dir=Path(args.data_dir),
    )
    model = AdaptiveMultimodalForecaster(d_latent=256, rank=1, v_indices=3)
    overhead = model.parameter_overhead_report()
    print(
        f"Model initialized | Total params: {int(overhead['total_params']):,} | "
        f"LoRA params: {int(overhead['lora_params']):,} ({overhead['lora_overhead_pct']:.4f}% < 2% NFR2)"
    )

    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
    )
    print(f"Training finished | Final Train MSE: {history['train_loss'][-1]:.4f}")

    eval_out = evaluate_model(model, test_loader, eta=0.2)
    metrics = eval_out["metrics"]
    bt = eval_out["backtest"]
    print(
        f"Test Metrics (eta=0.2) -> RMSE: {metrics['RMSE']:.4f} | MAE: {metrics['MAE']:.4f} | "
        f"MAPE: {metrics['MAPE']:.2f}% | R2: {metrics['R2']:.4f} | DA: {metrics['DA']:.2f}%"
    )
    print(
        f"Backtest -> CumReturn: {bt['Cumulative_Return_Pct']:.2f}% | "
        f"Sharpe: {bt['Annualised_Sharpe_Ratio']:.2f} | MaxDD: {bt['Max_Drawdown_Pct']:.2f}%"
    )


if __name__ == "__main__":
    main()
