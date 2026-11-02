"""
Phase 4: Chronological Forward Pass Evaluation Engine (Steps 1-6).

Implements:
- Step 1: DAINLayer (Deep Adaptive Input Normalisation)
- Step 2: Probabilistic Modality Encoders extracting (e_i, sigma_i, H(e_i)):
    * PriceProbabilisticEncoder (LSTM over h=30 x 8 OHLCV+Technicals)
    * SentimentProbabilisticEncoder (FinBERT Backbone over h=30 x 3 FinBERT polarities or token IDs)
    * NewsProbabilisticEncoder (DistilBERT Backbone over d=5 x 512 news embeddings or token IDs)
    * MacroProbabilisticEncoder (MLP over Macro CPI inflation indicators)
- Steps 3-6: DynamicGatingReliabilityLayer:
    * Step 3: Evidential Reliability Estimation (Attenuation Block)
    * Step 4: Joint Context Assembly C(t) = [M(t) || R(t) || S(t)]
    * Step 5: Attention Gating & Hard-Boundary Penalization (-1e9) + Softmax
    * Step 6: Convex Cross-Modal Fusion Z_fused = sum_i W_i * e_i
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BertModel,
    DistilBertConfig,
    DistilBertModel,
)


def compute_shannon_entropy(embedding: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Computes normalized Shannon information entropy H(e_i^{(t)}) in R^{Batch}
    over the latent embedding distribution.
    """
    probs = F.softmax(embedding, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + eps), dim=-1)
    max_entropy = torch.log(torch.tensor(float(embedding.size(-1)), device=embedding.device))
    return entropy / (max_entropy + eps)


class DAINLayer(nn.Module):
    """Deep Adaptive Input Normalisation Layer."""

    def __init__(self, input_dim: int) -> None:
        super(DAINLayer, self).__init__()
        self.input_dim = input_dim
        self.alpha = nn.Parameter(torch.zeros(1, input_dim, 1))
        self.beta = nn.Parameter(torch.ones(1, input_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x_trans = x.transpose(1, 2)
            mean = torch.mean(x_trans, dim=-1, keepdim=True)
            std = torch.std(x_trans, dim=-1, keepdim=True, unbiased=False) + 1e-8
            x_norm = (x_trans - mean) / std
            x_adaptive = (x_norm - self.alpha) * self.beta
            return x_adaptive.transpose(1, 2)
        else:
            mean = torch.mean(x, dim=0, keepdim=True)
            std = torch.std(x, dim=0, keepdim=True, unbiased=False) + 1e-8
            return ((x - mean) / std - self.alpha.squeeze(-1)) * self.beta.squeeze(-1)


class PriceProbabilisticEncoder(nn.Module):
    """
    Step 2 Price LSTM Probabilistic Encoder (Modality 0).
    Maps normalized X_OHLCV in R^{Batch x h x 8} to:
    - e_0 in R^{Batch x d_out}
    - sigma_0 in R^{Batch x d_var}
    - h_0 in R^{Batch}
    """

    def __init__(
        self,
        d_out: int = 256,
        d_var: int = 16,
        input_dim: int = 8,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super(PriceProbabilisticEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.mu_head = nn.Sequential(
            nn.Linear(hidden_dim, d_out),
            nn.LayerNorm(d_out),
            nn.GELU(),
        )
        self.var_head = nn.Linear(hidden_dim, d_var)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        e = self.mu_head(last_hidden)
        sigma = F.softplus(self.var_head(last_hidden)) + 1e-6
        h = compute_shannon_entropy(e)
        return e, sigma, h


class SentimentProbabilisticEncoder(nn.Module):
    """
    Step 2 Social Sentiment FinBERT Probabilistic Encoder (Modality 1, FR2).
    Uses a Hugging Face FinBERT (BERT-based) transformer architecture to encode
    social sentiment over the lookback window h=30:
    - Accepts DAIN-normalized FinBERT polarity sequences X_sentiment in R^{Batch x h x 3}
      ([Prob_pos, Prob_neg, Prob_neut]) and routes them as inputs_embeds through
      the FinBERT encoder backbone (`transformers.BertModel`).
    - Also supports direct raw token `input_ids` & `attention_mask` or raw text strings
      via `extract_polarities_from_token_ids` using a 3-class FinBERT classification head
      (`BertForSequenceClassification`, num_labels=3: positive, negative, neutral).
    Outputs:
    - e_1 in R^{Batch x d_out} (d_out=256)
    - sigma_1 in R^{Batch x d_var} (d_var=16)
    - h_1 in R^{Batch} (Shannon information entropy)
    """

    FINBERT_MODEL_NAME = "ProsusAI/finbert"

    def __init__(
        self,
        d_out: int = 256,
        d_var: int = 16,
        input_dim: int = 3,
        finbert_hidden_dim: int = 128,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 4,
        pretrained_model_name: Optional[str] = None,
        load_pretrained_weights: bool = False,
    ) -> None:
        super(SentimentProbabilisticEncoder, self).__init__()
        self.d_out = d_out
        self.d_var = d_var
        self.input_dim = input_dim
        self.model_name = pretrained_model_name or self.FINBERT_MODEL_NAME

        # Project 3D FinBERT polarity coordinates [Prob_pos, Prob_neg, Prob_neut]
        # into the FinBERT transformer embedding dimension
        if load_pretrained_weights:
            self.finbert_classifier = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, num_labels=3
            )
            self.finbert_backbone = self.finbert_classifier.bert
            hidden_size = self.finbert_backbone.config.hidden_size
        else:
            finbert_config = BertConfig(
                vocab_size=30522,
                hidden_size=finbert_hidden_dim,
                num_hidden_layers=num_hidden_layers,
                num_attention_heads=num_attention_heads,
                intermediate_size=finbert_hidden_dim * 2,
                max_position_embeddings=256,
                num_labels=3,  # [positive, negative, neutral]
            )
            self.finbert_classifier = BertForSequenceClassification(finbert_config)
            self.finbert_backbone = self.finbert_classifier.bert
            hidden_size = finbert_hidden_dim

        self.polarity_projector = nn.Linear(input_dim, hidden_size)
        self.mu_head = nn.Sequential(
            nn.Linear(hidden_size, d_out),
            nn.LayerNorm(d_out),
            nn.GELU(),
        )
        self.var_head = nn.Linear(hidden_size, d_var)

    def extract_polarities_from_token_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Runs token IDs through the FinBERT 3-class sequence classifier to produce
        [Prob_pos, Prob_neg, Prob_neut] in R^{Batch x 3}.
        """
        logits = self.finbert_classifier(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits
        return F.softmax(logits, dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for Modality 1 (Social Sentiment):
        - If `x` is a floating-point polarity tensor in R^{Batch x h x 3}, projects into
          FinBERT hidden space and encodes via `self.finbert_backbone(inputs_embeds=...)`.
        - If `x` is integer `input_ids` in Z^{Batch x seq_len}, encodes directly via
          `self.finbert_backbone(input_ids=x, attention_mask=attention_mask)`.
        """
        if x.dtype in (torch.long, torch.int, torch.int32, torch.int64):
            outputs = self.finbert_backbone(input_ids=x, attention_mask=attention_mask)
            pooled = outputs.last_hidden_state[:, 0, :]
        else:
            if x.dim() == 2:
                x = x.unsqueeze(1)
            inputs_embeds = self.polarity_projector(x)
            outputs = self.finbert_backbone(inputs_embeds=inputs_embeds)
            # Pool across the h=30 lookback window
            pooled = torch.mean(outputs.last_hidden_state, dim=1)

        e = self.mu_head(pooled)
        sigma = F.softplus(self.var_head(pooled)) + 1e-6
        h = compute_shannon_entropy(e)
        return e, sigma, h


class NewsProbabilisticEncoder(nn.Module):
    """
    Step 2 Expert Financial News DistilBERT Probabilistic Encoder (Modality 2, FR2).
    Uses Hugging Face `DistilBertModel` (`distilbert-base-uncased` architecture)
    with tokenization sequence parameter `max_length=256` 
    to encode expert news headlines/commentary over the sliding delay window d=5:
    - Accepts DAIN-normalized X_Expert in R^{Batch x d x 512} and passes it as
      `inputs_embeds` through `DistilBertModel`.
    - Also supports direct token `input_ids` & `attention_mask` in Z^{Batch x seq_len}
      (truncated to `max_length=256`).
    Outputs:
    - e_2 in R^{Batch x d_out} (d_out=512)
    - sigma_2 in R^{Batch x d_var} (d_var=16)
    - h_2 in R^{Batch} (Shannon information entropy)
    """

    DISTILBERT_MODEL_NAME = "distilbert-base-uncased"
    DISTILBERT_MAX_LENGTH = 256

    def __init__(
        self,
        d_out: int = 256,
        d_var: int = 16,
        input_dim: int = 256,
        max_length: int = 256,
        n_layers: int = 2,
        n_heads: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        pretrained_model_name: Optional[str] = None,
        load_pretrained_weights: bool = False,
    ) -> None:
        super(NewsProbabilisticEncoder, self).__init__()
        self.d_out = d_out
        self.d_var = d_var
        self.input_dim = input_dim
        self.max_length = max_length
        self.model_name = pretrained_model_name or self.DISTILBERT_MODEL_NAME

        if load_pretrained_weights:
            self.distilbert = AutoModel.from_pretrained(self.model_name)
            distil_dim = self.distilbert.config.dim
        else:
            distil_config = DistilBertConfig(
                vocab_size=30522,
                dim=input_dim,
                n_layers=n_layers,
                n_heads=n_heads,
                hidden_dim=dim_feedforward,
                dropout=dropout,
                attention_dropout=dropout,
                max_position_embeddings=max_length,
            )
            self.distilbert = DistilBertModel(distil_config)
            distil_dim = input_dim

        self.input_proj = (
            nn.Identity() if input_dim == distil_dim else nn.Linear(input_dim, distil_dim)
        )
        self.mu_head = nn.Sequential(
            nn.Linear(distil_dim, d_out),
            nn.LayerNorm(d_out),
            nn.GELU(),
        )
        self.var_head = nn.Linear(distil_dim, d_var)

    def tokenize_headlines(
        self,
        headlines: List[str],
        tokenizer: Optional[AutoTokenizer] = None,
        device: Optional[torch.device | str] = None,
    ) -> dict:
        """
        Tokenizes news headlines for DistilBERT with `max_length=256`.
        """
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        encoded = tokenizer(
            headlines,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        if device is not None:
            encoded = {k: v.to(device) for k, v in encoded.items()}
        return encoded

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for Modality 2 (Expert News via DistilBERT):
        - If `x` is a floating-point tensor in R^{Batch x d x 256}, passes `inputs_embeds`
          through `self.distilbert`.
        - If `x` is integer `input_ids` in Z^{Batch x seq_len}, truncates to
          `max_length=256` and encodes token IDs through `self.distilbert`.
        """
        if x.dtype in (torch.long, torch.int, torch.int32, torch.int64):
            if x.size(1) > self.max_length:
                x = x[:, : self.max_length]
                if attention_mask is not None:
                    attention_mask = attention_mask[:, : self.max_length]
            outputs = self.distilbert(input_ids=x, attention_mask=attention_mask)
            pooled = outputs.last_hidden_state[:, 0, :]
        else:
            if x.dim() == 2:
                x = x.unsqueeze(1)
            if x.size(1) > self.max_length:
                x = x[:, : self.max_length, :]
            inputs_embeds = self.input_proj(x)
            outputs = self.distilbert(inputs_embeds=inputs_embeds)
            # Mean-pool over the sliding delay window d=5
            pooled = torch.mean(outputs.last_hidden_state, dim=1)

        e = self.mu_head(pooled)
        sigma = F.softplus(self.var_head(pooled)) + 1e-6
        h = compute_shannon_entropy(e)
        return e, sigma, h


class MacroProbabilisticEncoder(nn.Module):
    """
    Step 2 Macroeconomic Inflation MLP Probabilistic Encoder (Modality 3).
    Maps normalized X_macro in R^{Batch x k} to:
    - e_3 in R^{Batch x d_out}
    - sigma_3 in R^{Batch x d_var}
    - h_3 in R^{Batch}
    """

    def __init__(
        self,
        d_out: int = 256,
        d_var: int = 16,
        input_dim: int = 1,
        hidden_dim: int = 64,
    ) -> None:
        super(MacroProbabilisticEncoder, self).__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.mu_head = nn.Sequential(
            nn.Linear(hidden_dim, d_out),
            nn.LayerNorm(d_out),
            nn.GELU(),
        )
        self.var_head = nn.Linear(hidden_dim, d_var)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.dim() == 3:
            x = x[:, -1, :]
        hidden = self.backbone(x)
        e = self.mu_head(hidden)
        sigma = F.softplus(self.var_head(hidden)) + 1e-6
        h = compute_shannon_entropy(e)
        return e, sigma, h


# Explicit aliases for FinBERT Sentiment and DistilBERT News Encoders
PriceLSTMEncoder = PriceProbabilisticEncoder
FinBERTSentimentEncoder = SentimentProbabilisticEncoder
SentimentLSTMEncoder = SentimentProbabilisticEncoder
DistilBERTNewsEncoder = NewsProbabilisticEncoder
NewsTransformerEncoder = NewsProbabilisticEncoder
MacroMLPEncoder = MacroProbabilisticEncoder


class DynamicGatingReliabilityLayer(nn.Module):
    """
    Executes Steps 3 to 6 of the chronological forward pass sequence:
    - Step 3: Reliability Estimation (The Attenuation Block)
    - Step 4: Joint Context Assembly C(t) = [M(t) || R(t) || S(t)]
    - Step 5: Attention Gating & Hard-Boundary Penalization (-1e9)
    - Step 6: Convex Cross-Modal Fusion
    """

    def __init__(
        self,
        d_latent: int = 256,
        d_var: int = 16,
        v_indices: int = 3,
    ) -> None:
        super(DynamicGatingReliabilityLayer, self).__init__()
        self.d_latent = d_latent
        self.d_var = d_var
        self.v_indices = v_indices
        self.num_modalities = 4

        # Step 3 Parameters: Reliability Attenuation Layer Components
        self.phi_projections = nn.ModuleList([nn.Linear(d_var, 1) for _ in range(4)])
        self.w_coef = nn.Linear(d_latent + v_indices, 2)
        self.lambdas = nn.Parameter(torch.rand(4) * 0.1 + 0.01)

        # Step 5 Parameters: Gating Weight Tensor G in R^{4 x (8+V)}
        self.G = nn.Parameter(torch.randn(4, 4 + 4 + v_indices) * 0.1)

        # Stored for interpretability inspection (RQ2 / RQ4)
        self.last_reliability: torch.Tensor | None = None

    def forward(
        self,
        E_stack: torch.Tensor,
        Sigma_stack: torch.Tensor,
        H_scores: torch.Tensor,
        M: torch.Tensor,
        delta_T: torch.Tensor,
        S: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Executes Steps 3 to 6 of the chronological forward pass sequence."""
        # Safety guard: ensure at least one modality is active per batch row
        row_sums = M.sum(dim=-1, keepdim=True)
        if torch.any(row_sums == 0.0):
            M = M.clone()
            zero_rows = (row_sums.squeeze(-1) == 0.0)
            M[zero_rows, 0] = 1.0

        R_scores = []

        # ====================================================
        # STEP 3: RELIABILITY ESTIMATION (THE ATTENUATION BLOCK)
        # ====================================================
        for i in range(self.num_modalities):
            phi_sigma = F.softplus(self.phi_projections[i](Sigma_stack[:, i, :]).squeeze(-1))
            encoder_context = torch.cat([E_stack[:, i, :], S], dim=-1)
            coefficients = F.softplus(self.w_coef(encoder_context))
            delta_i = coefficients[:, 0]
            tau_i = coefficients[:, 1]

            h_i = H_scores[:, i]
            dt_i = delta_T[:, i]
            decay_sum = phi_sigma + (delta_i * h_i) + (tau_i * dt_i)
            r_i = torch.exp(-torch.abs(self.lambdas[i]) * decay_sum)

            r_i = torch.where(M[:, i] == 1.0, r_i, torch.zeros_like(r_i))
            R_scores.append(r_i)

        R = torch.stack(R_scores, dim=-1)
        self.last_reliability = R.detach()

        # ====================================================
        # STEP 4: JOINT CONTEXT ASSEMBLY
        # ====================================================
        C = torch.cat([M, R, S], dim=-1)

        # ====================================================
        # STEP 5: ATTENTION GATING & HARD-BOUNDARY PENALIZATION
        # ====================================================
        raw_logits = torch.tanh(F.linear(C, self.G))
        p = torch.where(M == 1.0, torch.zeros_like(M), torch.fill_(torch.clone(M), -1e9))
        penalised_logits = raw_logits + p

        # Softmax allocation naturally bounds weights (sum_wi = 1.0, wi >= 0) matching NFR3
        W = F.softmax(penalised_logits, dim=-1)

        # ====================================================
        # STEP 6: CONVEX CROSS-MODAL FUSION
        # ====================================================
        W_expanded = W.unsqueeze(-1)
        Z_fused = torch.sum(W_expanded * E_stack, dim=1)

        return Z_fused, W
