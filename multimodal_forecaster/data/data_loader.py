"""
PyTorch Dataset and DataLoader construction for the Multimodal Financial Forecaster.

Loads raw historical datasets (`price`, `sentiment`, `news`, `macro`) from the
`data/` folder, performs chronological out-of-sample splitting (Train / Validation / Test),
and yields 8-tuples matching the benchmark specification:
(X_P, X_S, X_N, X_M, delta_T, S_state, a_close, y_target)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from multimodal_forecaster.data.aligner import (
    DEFAULT_DATA_DIR,
    AlignedDatasetBundle,
    MultimodalDataAligner,
)


class MultimodalFinancialDataset(Dataset):
    """
    PyTorch Dataset yielding aligned 4-modality samples:
    - X_P: [h, 8] Price OHLCV + technicals tensor
    - X_S: [h, 3] FinBERT Sentiment probability tensor
    - X_N: [d, 256] DistilBERT Expert News tensor
    - X_M: [k] Macro Inflation indicator tensor
    - delta_T: [4] Temporal latency vector
    - S_state: [V] Market state volatility vector
    - a_close: [1] Rolling lag-based closing average baseline
    - y_target: [1] Next-day closing price target P^{(t+1)}
    """

    def __init__(
        self,
        bundle: AlignedDatasetBundle,
        indices: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        if indices is None:
            indices = np.arange(len(bundle.y_target))

        self.X_P = torch.as_tensor(bundle.X_OHLCV[indices], dtype=torch.float32)
        self.X_S = torch.as_tensor(bundle.X_sentiment[indices], dtype=torch.float32)
        self.X_N = torch.as_tensor(bundle.X_Expert[indices], dtype=torch.float32)
        self.X_M = torch.as_tensor(bundle.X_macro[indices], dtype=torch.float32)
        self.delta_T = torch.as_tensor(bundle.delta_T[indices], dtype=torch.float32)
        self.S_state = torch.as_tensor(bundle.S_state[indices], dtype=torch.float32)
        self.a_close = torch.as_tensor(bundle.a_close[indices], dtype=torch.float32)
        self.y_target = torch.as_tensor(bundle.y_target[indices], dtype=torch.float32)
        self.P_current = torch.as_tensor(bundle.P_current[indices], dtype=torch.float32)
        self.timestamps = bundle.timestamps[indices]
        self.asset_name = bundle.asset_name

    def __len__(self) -> int:
        return self.y_target.size(0)

    def __getitem__(
        self, idx: int
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        return (
            self.X_P[idx],
            self.X_S[idx],
            self.X_N[idx],
            self.X_M[idx],
            self.delta_T[idx],
            self.S_state[idx],
            self.a_close[idx],
            self.y_target[idx],
        )


def create_dataloaders(
    asset_name: str = "Gold",
    batch_size: int = 32,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    num_days: Optional[int] = None,
    aligner: Optional[MultimodalDataAligner] = None,
    data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
    price_file: Optional[Union[str, Path]] = None,
    sentiment_file: Optional[Union[str, Path]] = None,
    news_file: Optional[Union[str, Path]] = None,
    macro_file: Optional[Union[str, Path]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, MultimodalFinancialDataset]]:
    """
    Creates chronological (non-shuffled in time split) Train, Validation, and Test
    DataLoaders strictly from the raw datasets in `data_dir` (default: `<project_root>/data`).
    """
    if aligner is None:
        aligner = MultimodalDataAligner()

    bundle = aligner.build_aligned_dataset(
        asset_name=asset_name,
        num_days=num_days,
        data_dir=data_dir,
        price_file=price_file,
        sentiment_file=sentiment_file,
        news_file=news_file,
        macro_file=macro_file,
    )
    total_samples = len(bundle.y_target)
    train_end = int(total_samples * train_ratio)
    val_end = int(total_samples * (train_ratio + val_ratio))

    train_idx = np.arange(0, train_end)
    val_idx = np.arange(train_end, val_end)
    test_idx = np.arange(val_end, total_samples)

    train_ds = MultimodalFinancialDataset(bundle, train_idx)
    val_ds = MultimodalFinancialDataset(bundle, val_idx)
    test_ds = MultimodalFinancialDataset(bundle, test_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    datasets = {"train": train_ds, "val": val_ds, "test": test_ds}
    return train_loader, val_loader, test_loader, datasets
