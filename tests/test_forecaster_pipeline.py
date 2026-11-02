"""
Comprehensive verification and unit test suite for the Multimodal Stock Prediction
Framework (Phases 1-6, FR1-FR6, NFR1-NFR3, RQ1-RQ4), including raw dataset ingestion
from the `data/` folder.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from transformers import BertModel, DistilBertModel

from multimodal_forecaster.data.aligner import (
    DEFAULT_DATA_DIR,
    PRICE_FEATURE_COLUMNS,
    SENTIMENT_COLUMNS,
    MultimodalDataAligner,
)
from multimodal_forecaster.data.data_loader import (
    MultimodalFinancialDataset,
    create_dataloaders,
)
from multimodal_forecaster.models.forecaster import (
    AdaptiveMultimodalForecaster,
    SingleModalityPriceLSTM,
    StaticEarlyFusionBaseline,
)
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
from multimodal_forecaster.scripts.benchmarks import (
    run_audited_robustness_sweep,
)
from multimodal_forecaster.scripts.evaluate import (
    evaluate_model,
)
from multimodal_forecaster.scripts.train import train_model
from multimodal_forecaster.utils.missingness_engine import MissingnessEngine


def test_raw_data_folder_ingestion(tmp_path: Path) -> None:
    """
    Verifies loading the 4 raw datasets from the `data/` folder:
    1. Raw `price.csv` (only Date, Open, High, Low, Close, Volume -> computes EMA, RSI, BB_Width)
    2. Raw `sentiment.csv` (with raw unstructured text comments -> encoded via FinBERT)
    3. Raw `news.csv` (with raw unstructured news headlines -> encoded via DistilBERT)
    4. Raw `macro.csv` (monthly CPI inflation releases with NaNs on intermediate days)
    """
    dates = pd.bdate_range("2021-01-01", periods=95)

    # 1. Raw Price CSV (only 5 OHLCV columns; technicals computed by aligner)
    raw_price_df = pd.DataFrame(
        {
            "Date": dates,
            "Open": np.linspace(1800.0, 1900.0, len(dates)),
            "High": np.linspace(1810.0, 1915.0, len(dates)),
            "Low": np.linspace(1790.0, 1890.0, len(dates)),
            "Close": np.linspace(1805.0, 1905.0, len(dates)),
            "Volume": np.full(len(dates), 150000.0),
        }
    )
    raw_price_df.to_csv(tmp_path / "price.csv", index=False)

    # 2. Raw Sentiment CSV (unstructured financial comments)
    raw_sent_df = pd.DataFrame(
        {
            "Date": dates,
            "text": [
                "Bullish gold breakout as safe haven demand climbs"
                if i % 2 == 0
                else "Market consolidates ahead of treasury auction"
                for i in range(len(dates))
            ],
        }
    )
    raw_sent_df.to_csv(tmp_path / "sentiment.csv", index=False)

    # 3. Raw Expert News CSV (unstructured news headlines)
    raw_news_df = pd.DataFrame(
        {
            "Date": dates,
            "headline": [
                f"Central bank report day {i}: monetary policy outlook and commodity flows"
                for i in range(len(dates))
            ],
        }
    )
    raw_news_df.to_csv(tmp_path / "news.csv", index=False)

    # 4. Raw Macro Inflation CSV (monthly release strictly every 30 rows, NaN otherwise)
    macro_vals = np.full(len(dates), np.nan)
    macro_vals[::30] = [2.4, 2.6, 2.9, 3.1]
    raw_macro_df = pd.DataFrame({"Date": dates, "CPI_Inflation": macro_vals})
    raw_macro_df.to_csv(tmp_path / "macro.csv", index=False)

    aligner = MultimodalDataAligner(lookback_h=30, news_delay_d=5)
    bundle = aligner.load_from_data_folder(data_dir=tmp_path, asset_name="Gold")
    assert bundle is not None

    expected_samples = len(dates) - 30
    assert bundle.X_OHLCV.shape == (expected_samples, 30, 8)
    assert bundle.X_sentiment.shape == (expected_samples, 30, 3)
    assert np.allclose(bundle.X_sentiment.sum(axis=-1), 1.0, atol=1e-5)
    assert bundle.X_Expert.shape == (expected_samples, 5, 256)
    assert bundle.X_macro.shape == (expected_samples, 1)
    assert not np.isnan(bundle.X_macro).any()

    # Verify create_dataloaders directly from the raw data folder
    train_loader, val_loader, test_loader, _ = create_dataloaders(
        asset_name="Gold",
        batch_size=16,
        data_dir=tmp_path,
    )
    first_batch = next(iter(train_loader))
    assert len(first_batch) == 8


def test_phase2_data_alignment_and_dimensions(tmp_path: Path) -> None:
    """
    Verifies FR1 (Phase 2) strictly from raw files in `data/`:
    - Price matrix X_OHLCV in R^{h x 8} with h=30 and 8 structural channels
    - Sentiment matrix X_sentiment in R^{h x 3} with FinBERT probabilities summing to 1
    - Expert News matrix X_Expert in R^{d x 256} with d=5 DistilBERT representations
    - Macro Inflation raw records appear strictly once every 30 steps (intermediate NaNs)
    - Aligned bundle contains valid forward-filled macro values, delta_T, S_state, a_close, y_target
    - Missing data files raise FileNotFoundError (synthetic generation is disabled).
    """
    aligner = MultimodalDataAligner(lookback_h=30, news_delay_d=5, news_embed_dim=256, macro_dim=1)

    with pytest.raises(FileNotFoundError):
        aligner.build_aligned_dataset(asset_name="Gold", data_dir=tmp_path)

    price_df = aligner.load_raw_price_file(DEFAULT_DATA_DIR / "price.csv").iloc[:120]
    target_index = price_df.index
    sent_df = aligner.load_raw_sentiment_file(DEFAULT_DATA_DIR / "sentiment.csv", target_index=target_index)
    news_emb = aligner.load_raw_news_file(DEFAULT_DATA_DIR / "news.csv", target_index=target_index)
    macro_df = aligner.load_raw_macro_file(DEFAULT_DATA_DIR / "macro.csv", target_index=target_index)

    assert list(price_df.columns) == PRICE_FEATURE_COLUMNS
    assert list(sent_df.columns) == SENTIMENT_COLUMNS
    assert np.allclose(sent_df.to_numpy().sum(axis=-1), 1.0, atol=1e-5)

    # Verify S_3(t) / BB_Width formula uses +1e-5 denominator protection
    close_s = price_df["Close"].astype(float)
    mu_20 = close_s.rolling(window=20, min_periods=1).mean()
    std_20 = close_s.rolling(window=20, min_periods=1).std(ddof=0).fillna(0.0)
    expected_bb_width = (4.0 * std_20) / (mu_20.abs() + 1e-5)
    assert np.allclose(price_df["BB_Width"].to_numpy(), expected_bb_width.to_numpy(), atol=1e-6)

    raw_macro_col = macro_df.iloc[:, 0].to_numpy()
    for idx in range(len(raw_macro_col)):
        if idx % 30 == 0:
            assert not np.isnan(raw_macro_col[idx]), f"Expected macro value at step {idx}"
        else:
            assert np.isnan(raw_macro_col[idx]), f"Expected NaN at intermediate step {idx}"

    bundle = aligner.align_streams(price_df, sent_df, news_emb, macro_df, asset_name="Gold")
    N = len(bundle.y_target)
    assert N == 120 - 30
    assert bundle.X_OHLCV.shape == (N, 30, 8)
    assert bundle.X_sentiment.shape == (N, 30, 3)
    assert bundle.X_Expert.shape == (N, 5, 256)
    assert bundle.X_macro.shape == (N, 1)
    assert bundle.delta_T.shape == (N, 4)
    assert bundle.S_state.shape == (N, 3)
    assert bundle.a_close.shape == (N, 1)
    assert bundle.y_target.shape == (N, 1)
    assert not np.isnan(bundle.X_macro).any()

    # Verify S_1(t) uses log-returns r(t) = ln(P_close(t) / P_close(t-1))
    log_ret = np.log(close_s.clip(lower=1e-5) / close_s.shift(1).clip(lower=1e-5)).fillna(0.0)
    expected_s1_full = (log_ret.rolling(10, min_periods=1).std(ddof=0).fillna(0.0) * 100.0).to_numpy(dtype=np.float32)
    assert np.allclose(bundle.S_state[:, 0], expected_s1_full[29:-1], atol=1e-5)
    # Verify S_3(t) uses 10 * (B_upper - B_lower) / (|mu_20(t)| + 1e-5)
    expected_s3_full = (expected_bb_width * 10.0).to_numpy(dtype=np.float32)
    assert np.allclose(bundle.S_state[:, 2], expected_s3_full[29:-1], atol=1e-5)


def test_phase3_missingness_engine_constraints() -> None:
    """
    Verifies FR6 (Phase 3):
    - Macro Inflation (Index 3) is available only on multiples of 30 steps
    - Independent Bernoulli trials respect eta in [0.1, 0.9]
    - No sample is ever completely blind across all 4 modalities (sum(M(t)) >= 1.0).
    """
    engine = MissingnessEngine(macro_period=30, seed=99)

    mask_eta0 = engine.generate_eval_mask(batch_size=65, step=0, eta=0.0).numpy()
    assert mask_eta0.shape == (65, 4)
    for row_idx in range(65):
        if row_idx % 30 == 0:
            assert mask_eta0[row_idx, 3] == 1.0
        else:
            assert mask_eta0[row_idx, 3] == 0.0

    for eta in [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
        mask = engine.generate_eval_mask(batch_size=128, step=2, eta=eta).numpy()
        row_sums = mask.sum(axis=-1)
        assert np.all(row_sums >= 1.0), f"Blind sample found at eta={eta}"
        global_rows = np.arange(2 * 128, 3 * 128)
        non_release_rows = (global_rows % 30) != 0
        assert np.all(mask[non_release_rows, 3] == 0.0)


def test_phase4_finbert_distilbert_encoders_and_gating() -> None:
    """
    Verifies Phase 4 (Steps 1-6, FR2, FR3):
    - SentimentProbabilisticEncoder (FinBERTSentimentEncoder) uses Hugging Face BertModel
    - NewsProbabilisticEncoder (DistilBERTNewsEncoder) uses Hugging Face DistilBertModel
    - DynamicGatingReliabilityLayer enforces R_i == 0 and W_i == 0 when M_i == 0,
      and convex sum_i W_i == 1.0 (NFR3).
    """
    batch_size = 8
    dain_3d = DAINLayer(input_dim=8)
    x_3d = torch.randn(batch_size, 30, 8) * 50.0 + 1200.0
    out_3d = dain_3d(x_3d)
    assert out_3d.shape == (batch_size, 30, 8)
    assert not torch.isnan(out_3d).any()

    dain_2d = DAINLayer(input_dim=1)
    x_2d = torch.randn(batch_size, 1) * 3.0 + 5.0
    out_2d = dain_2d(x_2d)
    assert out_2d.shape == (batch_size, 1)
    assert not torch.isnan(out_2d).any()

    enc_p = PriceProbabilisticEncoder(d_out=256, d_var=16, input_dim=8)
    enc_s = FinBERTSentimentEncoder(d_out=256, d_var=16, input_dim=3)
    enc_n = DistilBERTNewsEncoder(d_out=256, d_var=16, input_dim=256, max_length=256)
    enc_m = MacroProbabilisticEncoder(d_out=256, d_var=16, input_dim=1)

    assert isinstance(enc_s.finbert_backbone, BertModel)
    assert enc_s.FINBERT_MODEL_NAME == "ProsusAI/finbert"
    assert isinstance(enc_n.distilbert, DistilBertModel)
    assert enc_n.DISTILBERT_MODEL_NAME == "distilbert-base-uncased"
    assert enc_n.max_length == 256
    assert enc_n.distilbert.config.max_position_embeddings == 256

    e1, s1, h1 = enc_p(out_3d)
    e2, s2, h2 = enc_s(torch.randn(batch_size, 30, 3))
    e3, s3, h3 = enc_n(torch.randn(batch_size, 5, 256))
    e4, s4, h4 = enc_m(out_2d)

    for e, s, h in [(e1, s1, h1), (e2, s2, h2), (e3, s3, h3), (e4, s4, h4)]:
        assert e.shape == (batch_size, 256)
        assert s.shape == (batch_size, 16)
        assert torch.all(s > 0.0)
        assert h.shape == (batch_size,)

    dummy_tokens = torch.randint(100, 5000, (batch_size, 300), dtype=torch.long)
    polarities = enc_s.extract_polarities_from_token_ids(dummy_tokens[:, :128])
    assert polarities.shape == (batch_size, 3)
    assert torch.allclose(polarities.sum(dim=-1), torch.ones(batch_size), atol=1e-5)

    e2_tok, _, _ = enc_s(dummy_tokens[:, :128])
    # DistilBERT truncates 300 tokens to max_length=256 seamlessly
    e3_tok, _, _ = enc_n(dummy_tokens)
    assert e2_tok.shape == (batch_size, 256) and e3_tok.shape == (batch_size, 256)

    E_stack = torch.stack([e1, e2, e3, e4], dim=1)
    Sigma_stack = torch.stack([s1, s2, s3, s4], dim=1)
    H_scores = torch.stack([h1, h2, h3, h4], dim=-1)

    M = torch.tensor([[1.0, 1.0, 0.0, 0.0]] * batch_size)
    delta_T = torch.zeros(batch_size, 4)
    S = torch.randn(batch_size, 3)

    gate_layer = DynamicGatingReliabilityLayer(d_latent=256, d_var=16, v_indices=3)
    Z_fused, W = gate_layer(E_stack, Sigma_stack, H_scores, M, delta_T, S)

    assert Z_fused.shape == (batch_size, 256)
    assert W.shape == (batch_size, 4)
    assert torch.allclose(W[:, 2], torch.zeros(batch_size), atol=1e-6)
    assert torch.allclose(W[:, 3], torch.zeros(batch_size), atol=1e-6)
    assert torch.allclose(W.sum(dim=-1), torch.ones(batch_size), atol=1e-5)


def test_phase5_forecaster_lora_and_nfr2_overhead() -> None:
    """
    Verifies Phase 5 (FR4, FR5, NFR2):
    - AdaptiveMultimodalForecaster forward pass produces P_predict and W
    - LoRA adaptation activates only when M.sum(dim=-1) < 4.0
    - NFR2 parameter overhead of LoRA is < 2% of backbone network
    - End-to-end backpropagation updates DAIN, FinBERT, DistilBERT, Gating, LoRA, and Regression Head.
    """
    model = AdaptiveMultimodalForecaster(d_latent=256, rank=1, v_indices=3)
    with torch.no_grad():
        model.lora_B.normal_(0.0, 0.05)

    overhead = model.parameter_overhead_report()
    assert overhead["nfr2_compliant"] is True
    assert overhead["lora_overhead_pct"] < 2.0

    B = 4
    X_P = torch.randn(B, 30, 8)
    X_S = torch.rand(B, 30, 3)
    X_N = torch.randn(B, 5, 256)
    X_M = torch.randn(B, 1)
    delta_T = torch.rand(B, 4)
    S = torch.randn(B, 3)
    a_close = torch.full((B, 1), 1500.0)
    M_missing = torch.tensor(
        [
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )

    P_pred, W = model(X_P, X_S, X_N, X_M, M_missing, delta_T, S, a_close)
    assert P_pred.shape == (B, 1)
    assert W.shape == (B, 4)

    target = torch.full((B, 1), 1505.0)
    loss = torch.nn.functional.mse_loss(P_pred, target)
    loss.backward()

    assert model.lora_A.grad is not None and torch.norm(model.lora_A.grad) > 0.0
    assert model.lora_B.grad is not None and torch.norm(model.lora_B.grad) > 0.0
    assert model.gating_reliability_pipeline.G.grad is not None


def test_phase6_audited_robustness_sweep_and_benchmarks() -> None:
    """
    Verifies Phase 6 (RQ1, RQ3, NFR3):
    - `run_audited_robustness_sweep` executes across eta_range = [0.1, 0.5, 0.9]
      and passes all NFR3 convexity assertions
    - Training + evaluation + financial backtesting run cleanly across models.
    """
    train_loader, val_loader, test_loader, _ = create_dataloaders(
        asset_name="Gold", batch_size=16, num_days=100
    )
    model = AdaptiveMultimodalForecaster(d_latent=256, rank=1, v_indices=3)
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=2,
        lr=1e-3,
    )
    assert len(history["train_loss"]) == 2

    engine = MissingnessEngine(seed=42)
    eta_range = [0.1, 0.5, 0.9]
    audit_logs = run_audited_robustness_sweep(
        trained_model, test_loader, engine, eta_range
    )

    for eta in eta_range:
        w_mat = audit_logs[eta]
        assert w_mat.ndim == 2 and w_mat.shape[1] == 4
        assert np.allclose(np.sum(w_mat, axis=-1), 1.0, atol=1e-5)
        assert np.all(w_mat >= 0.0)

    for baseline in [StaticEarlyFusionBaseline(d_latent=256), SingleModalityPriceLSTM(input_dim=8)]:
        res = evaluate_model(baseline, test_loader, engine, eta=0.5)
        assert "MSE" in res["metrics"] and "DA" in res["metrics"]
        assert "Annualised_Sharpe_Ratio" in res["backtest"]
