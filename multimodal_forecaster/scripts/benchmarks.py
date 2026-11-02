"""
Phase 6: Benchmark Specifications & Audit Tracking Loops (RQ1-RQ4, NFR2, NFR3).

Implements `run_audited_robustness_sweep` from Phase 6 of the blueprint and
full comparative benchmarking across:
- AdaptiveMultimodalForecaster (Proposed Level II Intermediate Fusion + LoRA)
- StaticEarlyFusionBaseline (Level I Early Fusion Concatenation with Zero-Imputation)
- SingleModalityPriceLSTM (Unimodal Price LSTM Control)
over progressive missingness rates eta in [0.1, 0.3, 0.5, 0.7, 0.9] using raw datasets
stored in the `data/` folder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from multimodal_forecaster.data.aligner import DEFAULT_DATA_DIR
from multimodal_forecaster.data.data_loader import create_dataloaders
from multimodal_forecaster.models.forecaster import (
    AdaptiveMultimodalForecaster,
    SingleModalityPriceLSTM,
    StaticEarlyFusionBaseline,
)
from multimodal_forecaster.scripts.evaluate import evaluate_model
from multimodal_forecaster.scripts.train import train_model
from multimodal_forecaster.utils.missingness_engine import MissingnessEngine


def run_audited_robustness_sweep(
    model: nn.Module,
    test_loader: DataLoader,
    missingness_engine: MissingnessEngine,
    eta_range: Iterable[float],
) -> Dict[float, np.ndarray]:
    """
    Executes robustness evaluation loops while strictly logging weight channels (NFR3).
    Matches Phase 6 specification in readme.md.
    """
    model.eval()
    device = next(model.parameters()).device

    audit_logs: Dict[float, List[np.ndarray] | np.ndarray] = {eta: [] for eta in eta_range}

    with torch.no_grad():
        for eta in eta_range:
            for step, batch in enumerate(test_loader):
                X_P, X_S, X_N, X_M, delta_T, S_state, a_close, _ = [
                    t.to(device) for t in batch
                ]

                M_mask = missingness_engine.generate_eval_mask(
                    X_P.size(0), step, eta=eta, device=device
                )

                P_pred, w_weights = model(
                    X_P, X_S, X_N, X_M, M_mask, delta_T, S_state, a_close
                )

                audit_logs[eta].append(w_weights.cpu().numpy())

            audit_logs[eta] = np.concatenate(audit_logs[eta], axis=0)

            assert np.allclose(
                np.sum(audit_logs[eta], axis=-1), 1.0, atol=1e-5
            ), "NFR3 Violation: Weights must sum to 1"
            assert np.all(
                audit_logs[eta] >= 0.0
            ), "NFR3 Violation: Negative weights detected"

    print("NFR3 Verification Step: Audit pass completed with full convex parameter compliance.")
    return audit_logs  # type: ignore[return-value]


def run_comparative_benchmarks(
    assets: Optional[List[str]] = None,
    eta_range: Optional[List[float]] = None,
    num_days: Optional[int] = None,
    epochs: int = 4,
    batch_size: int = 32,
    seed: int = 42,
    data_dir: Optional[Union[str, Path]] = DEFAULT_DATA_DIR,
) -> Dict[str, Dict[str, Dict[float, Dict[str, float]]]]:
    """
    Runs comparative benchmark backtests (RQ1, RQ2, RQ3, RQ4) across assets and
    missingness rates comparing:
    1. AdaptiveMultimodalForecaster (Proposed)
    2. StaticEarlyFusionBaseline (Level I Early Fusion)
    3. SingleModalityPriceLSTM (Unimodal Price LSTM)
    """
    if assets is None:
        assets = ["Gold", "Bitcoin", "Brent_Crude_Oil"]
    if eta_range is None:
        eta_range = [0.1, 0.3, 0.5, 0.7, 0.9]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    missingness_engine = MissingnessEngine(seed=seed, device=device)
    results: Dict[str, Dict[str, Dict[float, Dict[str, float]]]] = {}

    for asset in assets:
        train_loader, val_loader, test_loader, _ = create_dataloaders(
            asset_name=asset,
            batch_size=batch_size,
            num_days=num_days,
            data_dir=data_dir if data_dir is not None else DEFAULT_DATA_DIR,
        )

        models_to_compare: Dict[str, nn.Module] = {
            "Adaptive_Level_II": AdaptiveMultimodalForecaster(d_latent=256, rank=1, v_indices=3),
            "Static_Early_Fusion": StaticEarlyFusionBaseline(d_latent=256),
            "Single_Modality_LSTM": SingleModalityPriceLSTM(input_dim=8),
        }

        asset_results: Dict[str, Dict[float, Dict[str, float]]] = {}
        for model_name, model in models_to_compare.items():
            trained_model, _ = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs,
                lr=1e-3,
                device=device,
                seed=seed,
            )

            _ = run_audited_robustness_sweep(
                trained_model, test_loader, missingness_engine, eta_range
            )

            eta_summary: Dict[float, Dict[str, float]] = {}
            for eta in eta_range:
                eval_res = evaluate_model(
                    model=trained_model,
                    test_loader=test_loader,
                    missingness_engine=missingness_engine,
                    eta=eta,
                    device=device,
                )
                combined = {}
                combined.update(eval_res["metrics"])
                combined.update(eval_res["backtest"])
                eta_summary[eta] = combined

            asset_results[model_name] = eta_summary
        results[asset] = asset_results

    return results


if __name__ == "__main__":
    summary = run_comparative_benchmarks(
        assets=["Gold", "Bitcoin", "Brent_Crude_Oil"],
        eta_range=[0.1, 0.5, 0.9],
        epochs=3,
    )
    for asset_name, model_map in summary.items():
        print(f"\n=== Asset: {asset_name} ===")
        for m_name, eta_map in model_map.items():
            for eta_val, metrics_dict in eta_map.items():
                print(
                    f"  {m_name:22s} | eta={eta_val:.1f} | "
                    f"RMSE={metrics_dict['RMSE']:8.3f} | MAE={metrics_dict['MAE']:8.3f} | "
                    f"DA={metrics_dict['DA']:5.1f}% | Sharpe={metrics_dict['Annualised_Sharpe_Ratio']:6.2f}"
                )
