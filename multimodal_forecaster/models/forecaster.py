"""
Phase 5: Complete Model & Shifting Price Regression Head (FR4, FR5, NFR2)
and Benchmark Baseline Models (Static Early Fusion & Single-Modality Price LSTM).

Modality Encoders (FR2):
- Modality 0 (Price): PriceProbabilisticEncoder (LSTM)
- Modality 1 (Social Sentiment): SentimentProbabilisticEncoder (FinBERT)
- Modality 2 (Expert Financial News): NewsProbabilisticEncoder (DistilBERT)
- Modality 3 (Macro Inflation): MacroProbabilisticEncoder (MLP)
"""

from __future__ import annotations

from typing import Dict, Tuple
import torch
import torch.nn as nn

from multimodal_forecaster.models.layers import (
    DAINLayer,
    DistilBERTNewsEncoder,
    DynamicGatingReliabilityLayer,
    FinBERTSentimentEncoder,
    MacroProbabilisticEncoder,
    NewsProbabilisticEncoder,
    PriceProbabilisticEncoder,
    SentimentProbabilisticEncoder,
)


class AdaptiveMultimodalForecaster(nn.Module):
    """
    Level II Intermediate Fusion Forecasting Framework with:
    - Step 1: DAINLayer pre-processing per modality
    - Step 2: Decoupled Probabilistic Modality Encoders:
        * Price: LSTM Probabilistic Encoder
        * Social Sentiment: FinBERT Probabilistic Encoder
        * Expert News: DistilBERT Probabilistic Encoder
        * Macro Indicators: MLP Probabilistic Encoder
    - Steps 3-6: DynamicGatingReliabilityLayer (Evidential Reliability + Hard Masking)
    - Low-Rank Adaptation (LoRA, rank=1) active under missingness (M.sum() < 4.0)
    - Shifting Price Regression Head predicting offset O(t) around a_close
    """

    def __init__(
        self,
        d_latent: int = 256,
        rank: int = 1,
        v_indices: int = 3,
        d_var: int = 16,
        macro_dim: int = 1,
        load_pretrained_nlp: bool = False,
    ) -> None:
        super(AdaptiveMultimodalForecaster, self).__init__()
        self.d_latent = d_latent
        self.rank = rank
        self.v_indices = v_indices
        self.d_var = d_var
        self.macro_dim = macro_dim

        # Step 1: DAIN Layers (before modality encoders)
        self.dain_price = DAINLayer(input_dim=8)
        self.dain_sentiment = DAINLayer(input_dim=3)
        self.dain_news = DAINLayer(input_dim=d_latent)
        self.dain_macro = DAINLayer(input_dim=macro_dim)

        # Step 2: Probabilistic Modality Encoders (LSTM, FinBERT, DistilBERT, MLP)
        self.encoder_price = PriceProbabilisticEncoder(
            d_out=d_latent, d_var=d_var, input_dim=8
        )
        self.encoder_sentiment = FinBERTSentimentEncoder(
            d_out=d_latent,
            d_var=d_var,
            input_dim=3,
            load_pretrained_weights=load_pretrained_nlp,
        )
        self.encoder_news = DistilBERTNewsEncoder(
            d_out=d_latent,
            d_var=d_var,
            input_dim=d_latent,
            load_pretrained_weights=load_pretrained_nlp,
        )
        self.encoder_macro = MacroProbabilisticEncoder(
            d_out=d_latent, d_var=d_var, input_dim=macro_dim
        )

        # Steps 3-6: Reliability Attenuation & Dynamic Gating Layer
        self.gating_reliability_pipeline = DynamicGatingReliabilityLayer(
            d_latent=d_latent,
            d_var=d_var,
            v_indices=v_indices,
        )

        # Low-Rank Adaptation Recovery Loop (LoRA: rank=1)
        self.lora_A = nn.Parameter(torch.randn(d_latent, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, d_latent))

        # Shifting Price Regression Head
        self.regression_head = nn.Sequential(
            nn.Linear(d_latent, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        X_OHLCV: torch.Tensor,
        X_sentiment: torch.Tensor,
        X_Expert: torch.Tensor,
        X_macro: torch.Tensor,
        M: torch.Tensor,
        delta_T: torch.Tensor,
        S: torch.Tensor,
        a_close: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # STEP 1 & 2: Forward Ingestion Pass (DAIN -> Probabilistic Encoders)
        X_price_norm = self.dain_price(X_OHLCV)
        X_sent_norm = self.dain_sentiment(X_sentiment)
        X_news_norm = self.dain_news(X_Expert)
        X_macro_norm = self.dain_macro(X_macro)

        e1, sigma1, h1 = self.encoder_price(X_price_norm)
        e2, sigma2, h2 = self.encoder_sentiment(X_sent_norm)
        e3, sigma3, h3 = self.encoder_news(X_news_norm)
        e4, sigma4, h4 = self.encoder_macro(X_macro_norm)

        E_stack = torch.stack([e1, e2, e3, e4], dim=1)
        Sigma_stack = torch.stack([sigma1, sigma2, sigma3, sigma4], dim=1)
        H_scores = torch.stack([h1, h2, h3, h4], dim=-1)

        # STEPS 3-6: Pipeline Pass
        Z_fused, W = self.gating_reliability_pipeline(
            E_stack, Sigma_stack, H_scores, M, delta_T, S
        )

        # Low-Rank Adaptation Recovery Loop (Active when M.sum() < 4.0)
        is_missing = (M.sum(dim=-1, keepdim=True) < 4.0).float()
        delta_z = torch.matmul(torch.matmul(Z_fused, self.lora_A), self.lora_B)
        Z_final = Z_fused + (is_missing * delta_z)

        # Shifting Price Regression Head & Baseline Denormalisation
        offset = self.regression_head(Z_final)
        P_predict = offset + a_close

        # Returns both target predictions and the Interpretability Tensor logs (NFR3)
        return P_predict, W

    def parameter_overhead_report(self) -> Dict[str, float]:
        """
        Computes parameter counts and verifies NFR2 compliance:
        LoRA adaptation parameter overhead must be < 2% of backbone parameters.
        """
        total_params = sum(p.numel() for p in self.parameters())
        lora_params = self.lora_A.numel() + self.lora_B.numel()
        backbone_params = total_params - lora_params
        overhead_pct = (lora_params / max(backbone_params, 1)) * 100.0
        return {
            "total_params": float(total_params),
            "backbone_params": float(backbone_params),
            "lora_params": float(lora_params),
            "lora_overhead_pct": float(overhead_pct),
            "nfr2_compliant": bool(overhead_pct < 2.0),
        }


class StaticEarlyFusionBaseline(nn.Module):
    """
    Baseline 1 (Dissertation Section 3.3.2): Static Level I Early Fusion Concatenation.
    Directly concatenates encoded modality vectors into X_concat = [e1 || e2 || e3 || e4]
    using naive zero-imputation when a modality is missing (M_i = 0), and passes
    the concatenated representation through a dense MLP regression head without
    dynamic gating or low-rank adaptation.
    """

    def __init__(
        self,
        d_latent: int = 256,
        d_var: int = 16,
        macro_dim: int = 1,
    ) -> None:
        super(StaticEarlyFusionBaseline, self).__init__()
        self.d_latent = d_latent
        self.encoder_price = PriceProbabilisticEncoder(d_out=d_latent, d_var=d_var, input_dim=8)
        self.encoder_sentiment = FinBERTSentimentEncoder(d_out=d_latent, d_var=d_var, input_dim=3)
        self.encoder_news = DistilBERTNewsEncoder(d_out=d_latent, d_var=d_var, input_dim=d_latent)
        self.encoder_macro = MacroProbabilisticEncoder(d_out=d_latent, d_var=d_var, input_dim=macro_dim)

        self.mlp_head = nn.Sequential(
            nn.Linear(4 * d_latent, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        X_OHLCV: torch.Tensor,
        X_sentiment: torch.Tensor,
        X_Expert: torch.Tensor,
        X_macro: torch.Tensor,
        M: torch.Tensor,
        delta_T: torch.Tensor,
        S: torch.Tensor,
        a_close: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        e1, _, _ = self.encoder_price(X_OHLCV)
        e2, _, _ = self.encoder_sentiment(X_sentiment)
        e3, _, _ = self.encoder_news(X_Expert)
        e4, _, _ = self.encoder_macro(X_macro)

        # Naive zero-imputation for missing channels
        e1_imputed = e1 * M[:, 0:1]
        e2_imputed = e2 * M[:, 1:2]
        e3_imputed = e3 * M[:, 2:3]
        e4_imputed = e4 * M[:, 3:4]

        X_concat = torch.cat([e1_imputed, e2_imputed, e3_imputed, e4_imputed], dim=-1)
        offset = self.mlp_head(X_concat)
        P_predict = offset + a_close

        # Static uniform weights across active channels for logging compatibility
        active_sum = torch.clamp(M.sum(dim=-1, keepdim=True), min=1.0)
        static_weights = M / active_sum
        return P_predict, static_weights


class SingleModalityPriceLSTM(nn.Module):
    """
    Baseline 2 (Dissertation Section 3.3.2): Single-Modality Price LSTM.
    Unimodal baseline operating exclusively on historical market price technicals (X_OHLCV)
    using a standard two-layer LSTM network.
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 128,
        num_layers: int = 2,
    ) -> None:
        super(SingleModalityPriceLSTM, self).__init__()
        self.dain_price = DAINLayer(input_dim=input_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        X_OHLCV: torch.Tensor,
        X_sentiment: torch.Tensor,
        X_Expert: torch.Tensor,
        X_macro: torch.Tensor,
        M: torch.Tensor,
        delta_T: torch.Tensor,
        S: torch.Tensor,
        a_close: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_norm = self.dain_price(X_OHLCV)
        # If price modality (index 0) is dropped out, zero-impute price input
        x_masked = x_norm * M[:, 0].view(-1, 1, 1)
        lstm_out, _ = self.lstm(x_masked)
        offset = self.regression_head(lstm_out[:, -1, :])
        P_predict = offset + a_close

        # Unimodal weight vector (100% on Price when active, else uniform fallback)
        w = torch.zeros_like(M)
        w[:, 0] = 1.0
        return P_predict, w
