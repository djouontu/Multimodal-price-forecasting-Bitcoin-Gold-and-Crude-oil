"""
Phase 3: Periodic Low-Frequency Parametric Missingness Engine (FR6).

Models the structural reporting intervals of Macro Inflation (Index 3),
rendering it available strictly once every 30 rows when unmasked, while
overlaying independent sample-level Bernoulli trials controlled by rate eta
(0.1 to 0.9).

Guarantees that all 4 modalities are never simultaneously blind for any sample:
at least one modality is always available (sum(M(t)) >= 1.0).

Indices:
    0 = Price (X_OHLCV)
    1 = Sentiment (X_sentiment)
    2 = Expert News (X_Expert)
    3 = Macro Inflation (X_inflation)
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import torch


class MissingnessEngine:
    """
    Generates binary availability mask M(t) in {0, 1}^{Batch x 4} for the 4 modalities.
    Supports:
    - Periodic monthly schedule for Macro Inflation (Index 3: available every 30 rows)
    - Independent Bernoulli dropout controlled by rate parameter eta in [0.0, 0.9]
    - Sparse textual commentary droughts (contiguous blocks)
    - Multi-modality simultaneous system outages
    - Strict non-blindness guarantee: every sample has at least 1 active modality.
    """

    def __init__(
        self,
        macro_period: int = 30,
        seed: int = 42,
        device: Optional[torch.device | str] = None,
    ) -> None:
        self.macro_period = macro_period
        self.seed = seed
        self.device = device
        self.rng = np.random.default_rng(seed)

    def reset_seed(self, seed: Optional[int] = None) -> None:
        """Resets internal random generator for reproducible evaluation sweeps."""
        self.rng = np.random.default_rng(self.seed if seed is None else seed)

    @staticmethod
    def _enforce_at_least_one_active(
        mask: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Ensures no sample row is completely blind across all 4 modalities.
        If a row has sum == 0, activates one randomly selected high-frequency modality
        (or modality 0..2) so sum(M[b]) >= 1.0.
        """
        row_sums = mask.sum(axis=-1)
        blind_rows = np.where(row_sums == 0.0)[0]
        for r_idx in blind_rows:
            # Prefer activating one of Price (0), Sentiment (1), or News (2)
            fallback_channel = int(rng.integers(0, 3))
            mask[r_idx, fallback_channel] = 1.0
        return mask

    def generate_eval_mask(
        self,
        batch_size: int,
        step: int = 0,
        eta: float = 0.0,
        regime: str = "periodic_bernoulli",
        device: Optional[torch.device | str] = None,
    ) -> torch.Tensor:
        """
        Generates availability mask M(t) in {0.0, 1.0}^{batch_size x 4}.

        Parameters
        ----------
        batch_size : int
            Number of samples in the current evaluation batch.
        step : int
            Current batch step index (used to track global row index for the
            30-day periodic Macro Inflation availability schedule).
        eta : float
            Dropout rate parameter in [0.0, 0.95]. Probability of a modality
            dropping out on a given sample is eta (keep probability = 1 - eta).
        regime : str
            One of:
            - "periodic_bernoulli" (default): Macro available every 30 rows + Bernoulli(1 - eta)
            - "complete": All 4 modalities active (except structural macro schedule if eta > 0)
            - "text_drought": Contiguous multi-day dropout of Sentiment (1) and News (2)
            - "multi_outage": Simultaneous outage of 2 or 3 modalities
        device : torch.device or str, optional
            Target device for the returned tensor.
        """
        eta_clipped = float(np.clip(eta, 0.0, 0.99))
        keep_prob = 1.0 - eta_clipped

        # Deterministic per-(step, eta) RNG stream so sweeps are reproducible
        step_seed = int(self.seed + (step + 1) * 1009 + round(eta_clipped * 1000) * 9176)
        local_rng = np.random.default_rng(step_seed)

        # Independent Bernoulli trials across all 4 modalities
        bernoulli_keep = (local_rng.random((batch_size, 4)) < keep_prob).astype(np.float32)

        # Structural reporting interval of Macro Inflation (Index 3):
        # Strictly available once every 30 rows when unmasked
        global_row_indices = np.arange(step * batch_size, (step + 1) * batch_size)
        macro_structural_mask = ((global_row_indices % self.macro_period) == 0).astype(np.float32)
        bernoulli_keep[:, 3] = bernoulli_keep[:, 3] * macro_structural_mask

        if regime == "complete" and eta_clipped == 0.0:
            mask = np.ones((batch_size, 4), dtype=np.float32)
            mask[:, 3] = macro_structural_mask
        elif regime == "text_drought":
            mask = bernoulli_keep.copy()
            # Contiguous blocks where both Sentiment (1) and Expert News (2) drop out
            drought_length = max(1, int(round(batch_size * max(eta_clipped, 0.5))))
            mask[:drought_length, 1] = 0.0
            mask[:drought_length, 2] = 0.0
        elif regime == "multi_outage":
            mask = bernoulli_keep.copy()
            # Drop 2 or 3 channels simultaneously in each sample
            for b in range(batch_size):
                num_drop = 3 if eta_clipped >= 0.5 else 2
                drop_indices = local_rng.choice(4, size=num_drop, replace=False)
                mask[b, drop_indices] = 0.0
        else:
            mask = bernoulli_keep

        # Guarantee that all modalities are never blind at the same time
        mask = self._enforce_at_least_one_active(mask, local_rng)

        target_device = device if device is not None else self.device
        return torch.as_tensor(mask, dtype=torch.float32, device=target_device)

    def generate_training_mask(
        self,
        batch_size: int,
        step: int = 0,
        min_eta: float = 0.0,
        max_eta: float = 0.5,
        device: Optional[torch.device | str] = None,
    ) -> torch.Tensor:
        """
        Generates stochastic availability masks during training so that both the
        DynamicGatingReliabilityLayer and Low-Rank Adaptation (LoRA) parameters
        experience diverse missing-modality patterns (sum(M) < 4.0) and complete
        patterns (sum(M) == 4.0).
        """
        eta_sample = float(self.rng.uniform(min_eta, max_eta))
        mask = self.generate_eval_mask(
            batch_size=batch_size,
            step=step,
            eta=eta_sample,
            device=device,
        )
        # Ensure ~25% of training samples have all 4 modalities present so complete-data
        # and missing-data branches are both well calibrated
        if batch_size >= 4:
            complete_rows = max(1, batch_size // 4)
            mask[:complete_rows, :] = 1.0
        return mask

