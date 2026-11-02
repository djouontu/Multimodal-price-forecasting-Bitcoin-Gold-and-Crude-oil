"""
Dynamic Multi-Asset 3-Model Comparative Benchmark & CSV Exporter.

Automatically discovers ANY asset placed in the `data/` folder (or runs on user-specified
`--assets`) from 2019-01-01 to 2023-12-31, trains and evaluates the 3 benchmark architectures
(`AdaptiveMultimodalForecaster`, `StaticEarlyFusionBaseline`, `SingleModalityPriceLSTM`),
and saves ONLY the resulting CSV files into the `results/` directory:
- `results/{slug}_benchmark_metrics.csv`
- `results/{slug}_test_predictions.csv`
- `results/{slug}_gating_weights_W.csv`
- `results/all_assets_benchmark_metrics.csv`
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import torch

from multimodal_forecaster.data.aligner import DEFAULT_DATA_DIR, MultimodalDataAligner
from multimodal_forecaster.data.data_loader import create_dataloaders
from multimodal_forecaster.models.forecaster import (
    AdaptiveMultimodalForecaster,
    SingleModalityPriceLSTM,
    StaticEarlyFusionBaseline,
)
from multimodal_forecaster.scripts.benchmarks import run_audited_robustness_sweep
from multimodal_forecaster.scripts.evaluate import evaluate_model
from multimodal_forecaster.scripts.train import train_model
from multimodal_forecaster.utils.missingness_engine import MissingnessEngine


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = WORKSPACE_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _normalize_asset_slug(asset_name: str) -> str:
    """Converts any arbitrary asset name into a clean lowercase file slug."""
    slug = asset_name.strip().lower().replace(" ", "_").replace("-", "_")
    if slug in ("brent_crude_oil", "crude_oil"):
        return "oil"
    return slug


def _format_display_name(asset_name: str) -> str:
    """Formats an arbitrary asset identifier into a human-readable title."""
    return asset_name.strip().replace("_", " ").title()


def run_single_asset_benchmark(
    asset_name: str,
    data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
    output_dir: Union[str, Path] = RESULTS_DIR,
    epochs: int = 5,
    batch_size: int = 32,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Runs the 3-model comparative benchmark for ANY `asset_name` present in `data_dir`
    and writes ONLY CSV files into `output_dir`:
    - `{slug}_benchmark_metrics.csv`
    - `{slug}_test_predictions.csv`
    - `{slug}_gating_weights_W.csv`
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    slug = _normalize_asset_slug(asset_name)
    display_name = _format_display_name(asset_name)

    metrics_csv_path = out_path / f"{slug}_benchmark_metrics.csv"
    pred_csv_path = out_path / f"{slug}_test_predictions.csv"
    w_csv_path = out_path / f"{slug}_gating_weights_W.csv"

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n[{display_name}] 1. Loading chronological dataset from '{data_dir}'...")
    train_loader, val_loader, test_loader, datasets = create_dataloaders(
        asset_name=asset_name,
        batch_size=batch_size,
        data_dir=data_dir,
    )
    test_ds = datasets["test"]
    test_dates = pd.to_datetime(test_ds.timestamps)
    true_prices = test_ds.y_target.cpu().numpy().reshape(-1)
    current_prices = test_ds.P_current.cpu().numpy().reshape(-1)

    models_spec = {
        "Adaptive_Level_II (Proposed)": AdaptiveMultimodalForecaster(d_latent=256, rank=1, v_indices=3),
        "Static_Early_Fusion (Level I)": StaticEarlyFusionBaseline(d_latent=256),
        "Single_Modality_LSTM (Price Only)": SingleModalityPriceLSTM(input_dim=8),
    }

    eta_range = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    missingness_engine = MissingnessEngine(seed=seed, device=device)

    benchmark_rows: List[Dict[str, Any]] = []
    predictions_eta02: Dict[str, np.ndarray] = {}
    predictions_by_eta: Dict[str, Dict[float, np.ndarray]] = {}
    gating_weights_eta02: np.ndarray | None = None

    print(f"[{display_name}] 2. Training and evaluating 3 models ({epochs} epochs each)...")
    for model_label, model_instance in models_spec.items():
        print(f"   -> Training {model_label} on {display_name}...")
        trained_m, _ = train_model(
            model=model_instance,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=1e-3,
            device=device,
            seed=seed,
        )
        predictions_by_eta[model_label] = {}

        _ = run_audited_robustness_sweep(
            trained_m, test_loader, missingness_engine, [0.1, 0.5, 0.9]
        )

        for eta in eta_range:
            eval_res = evaluate_model(
                model=trained_m,
                test_loader=test_loader,
                missingness_engine=missingness_engine,
                eta=eta,
                device=device,
            )
            m_dict = eval_res["metrics"]
            b_dict = eval_res["backtest"]
            preds = eval_res["predictions"].reshape(-1)
            predictions_by_eta[model_label][eta] = preds

            if np.isclose(eta, 0.2):
                predictions_eta02[model_label] = preds
                if "Adaptive_Level_II" in model_label:
                    gating_weights_eta02 = eval_res["weights"]

            mean_w = eval_res["mean_weights"]
            benchmark_rows.append(
                {
                    "Asset": display_name,
                    "Model": model_label,
                    "Eta": eta,
                    "W_Price": float(mean_w[0]),
                    "W_Sentiment": float(mean_w[1]),
                    "W_News": float(mean_w[2]),
                    "W_Macro": float(mean_w[3]),
                    "RMSE": float(m_dict["RMSE"]),
                    "MAE": float(m_dict["MAE"]),
                    "MAPE_Pct": float(m_dict["MAPE"]),
                    "R2": float(m_dict["R2"]),
                    "DA_Pct": float(m_dict["DA"]),
                    "CumReturn_Pct": float(b_dict["Cumulative_Return_Pct"]),
                    "Sharpe": float(b_dict["Annualised_Sharpe_Ratio"]),
                    "MaxDD_Pct": float(b_dict["Max_Drawdown_Pct"]),
                    "Active_Trade_Ratio_Pct": float(b_dict["Active_Trade_Ratio_Pct"]),
                }
            )

    metrics_df = pd.DataFrame(benchmark_rows)
    metrics_df.to_csv(metrics_csv_path, index=False)

    assert gating_weights_eta02 is not None
    w_df = pd.DataFrame(
        {
            "Date": test_dates,
            "W_Price": gating_weights_eta02[:, 0],
            "W_Sentiment": gating_weights_eta02[:, 1],
            "W_News": gating_weights_eta02[:, 2],
            "W_Macro": gating_weights_eta02[:, 3],
            "W_Sum": gating_weights_eta02.sum(axis=-1),
        }
    )
    w_df.to_csv(w_csv_path, index=False)

    pred_df = pd.DataFrame(
        {
            "Date": test_dates,
            "Current_Close_P_t": current_prices,
            "True_Next_Close_P_t1": true_prices,
            "Pred_Adaptive_Level_II_eta02": predictions_eta02["Adaptive_Level_II (Proposed)"],
            "Pred_Static_Early_Fusion_eta02": predictions_eta02["Static_Early_Fusion (Level I)"],
            "Pred_Single_Modality_LSTM_eta02": predictions_eta02["Single_Modality_LSTM (Price Only)"],
            "W_Price_eta02": gating_weights_eta02[:, 0],
            "W_Sentiment_eta02": gating_weights_eta02[:, 1],
            "W_News_eta02": gating_weights_eta02[:, 2],
            "W_Macro_eta02": gating_weights_eta02[:, 3],
            "Pred_Adaptive_Level_II_eta01": predictions_by_eta["Adaptive_Level_II (Proposed)"][0.1],
            "Pred_Adaptive_Level_II_eta05": predictions_by_eta["Adaptive_Level_II (Proposed)"][0.5],
            "Pred_Adaptive_Level_II_eta09": predictions_by_eta["Adaptive_Level_II (Proposed)"][0.9],
        }
    )
    pred_df.to_csv(pred_csv_path, index=False)

    print(f"[{display_name}] Saved CSV files:")
    print(f"   - {metrics_csv_path}")
    print(f"   - {pred_csv_path}")
    print(f"   - {w_csv_path}")

    return metrics_df


def run_all_assets_csv_export(
    assets: Optional[List[str]] = None,
    data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
    output_dir: Union[str, Path] = RESULTS_DIR,
    epochs: int = 5,
    batch_size: int = 32,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Automatically discovers all assets present in `data_dir` (when `assets=None`)
    or iterates over the specified `assets` list, runs the 3-model comparative
    benchmark for each asset, and exports only CSV files into `output_dir`.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not assets:
        discovered = MultimodalDataAligner.discover_assets_in_folder(data_dir)
        if not discovered:
            raise FileNotFoundError(
                f"No asset price dataset files (*_price_data.csv, *_price.csv, *_ohlcv.csv, or price.csv) "
                f"were found in '{data_dir}'."
            )
        assets = discovered
        print(f"Auto-discovered {len(assets)} asset(s) in '{data_dir}': {assets}")

    dfs: List[pd.DataFrame] = []
    for asset_name in assets:
        df_asset = run_single_asset_benchmark(
            asset_name=asset_name,
            data_dir=data_dir,
            output_dir=out_path,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
        )
        dfs.append(df_asset)

    all_metrics_df = pd.concat(dfs, ignore_index=True)
    all_metrics_csv = out_path / "all_assets_benchmark_metrics.csv"
    all_metrics_df.to_csv(all_metrics_csv, index=False)

    print(f"\nSaved combined multi-asset metrics CSV: {all_metrics_csv}")
    print(all_metrics_df.to_string(index=False))
    return all_metrics_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-discover any assets in data/ and export 3-model benchmark CSV results."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="Path to the data/ folder containing raw CSV/Parquet files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(RESULTS_DIR),
        help="Directory where output CSV files are saved (default: results/).",
    )
    parser.add_argument(
        "--assets",
        nargs="*",
        default=None,
        help="Optional list of asset names to run (if omitted, auto-discovers all assets in --data-dir).",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs per model.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    run_all_assets_csv_export(
        assets=args.assets,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
