"""
Evaluation metrics and Financial Backtesting Protocol (Section 3.3.4 & 3.3.5).

Computes:
- MSE, RMSE, MAE, MAPE (%), R^2, and Directional Accuracy (DA %)
- Daily trading strategy backtest with threshold delta = 0.2% (0.002):
  Cumulative Equity Return, Annualised Sharpe Ratio, and Maximum Drawdown (MDD).
"""

from __future__ import annotations

from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from multimodal_forecaster.utils.missingness_engine import MissingnessEngine


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    p_current: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Computes unscaled dollar-denominated evaluation metrics (Section 3.3.4):
    - MSE: Mean Squared Error
    - RMSE: Root Mean Squared Error
    - MAE: Mean Absolute Error
    - MAPE: Mean Absolute Percentage Error (%)
    - R2: Coefficient of Determination
    - DA: Directional Accuracy (%)
    """
    y_t = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_p = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    mse = float(np.mean((y_t - y_p) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_t - y_p)))
    mape = float(np.mean(np.abs((y_t - y_p) / (np.abs(y_t) + 1e-8))) * 100.0)

    ss_res = float(np.sum((y_t - y_p) ** 2))
    ss_tot = float(np.sum((y_t - np.mean(y_t)) ** 2))
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-8)))

    if p_current is not None:
        p_c = np.asarray(p_current, dtype=np.float64).reshape(-1)
        actual_dir = np.sign(y_t - p_c)
        pred_dir = np.sign(y_p - p_c)
        da = float(np.mean(actual_dir == pred_dir) * 100.0)
    else:
        if len(y_t) > 1:
            actual_dir = np.sign(np.diff(y_t))
            pred_dir = np.sign(y_p[1:] - y_t[:-1])
            da = float(np.mean(actual_dir == pred_dir) * 100.0)
        else:
            da = 100.0

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "R2": r2,
        "DA": da,
    }


def run_financial_backtest(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    p_current: np.ndarray,
    delta_threshold: float = 0.002,
    trading_days_per_year: int = 252,
) -> Dict[str, float]:
    """
    Executes the Financial Backtesting Protocol (Section 3.3.5):
    - Predicted return signal: r^{(t+1)} = (P_predict^{(t+1)} - P^{(t)}) / P^{(t)}
    - Long (+1) if r^{(t+1)} > +delta (delta = 0.2% = 0.002)
    - Short (-1) if r^{(t+1)} < -delta
    - Neutral (0) if |r^{(t+1)}| <= delta
    Returns Cumulative Equity Return (%), Annualised Sharpe Ratio, and Max Drawdown (%).
    """
    y_t = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_p = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    p_c = np.asarray(p_current, dtype=np.float64).reshape(-1)

    pred_return = (y_p - p_c) / (np.abs(p_c) + 1e-8)
    actual_return = (y_t - p_c) / (np.abs(p_c) + 1e-8)

    positions = np.zeros_like(pred_return)
    positions[pred_return > delta_threshold] = 1.0
    positions[pred_return < -delta_threshold] = -1.0

    # Deduct transaction cost when position is non-zero
    Position_Changes = np.abs(np.diff(np.concatenate([[0.0], positions])))
    strategy_daily_returns = (positions * actual_return) - (Position_Changes * (delta_threshold * 0.5))

    equity_curve = np.cumprod(1.0 + strategy_daily_returns)
    cumulative_return_pct = float((equity_curve[-1] - 1.0) * 100.0) if len(equity_curve) > 0 else 0.0

    mean_ret = float(np.mean(strategy_daily_returns))
    std_ret = float(np.std(strategy_daily_returns, ddof=0))
    if std_ret > 1e-10:
        sharpe_ratio = float((mean_ret / std_ret) * np.sqrt(trading_days_per_year))
    else:
        sharpe_ratio = 0.0

    running_peak = np.maximum.accumulate(equity_curve)
    drawdowns = (running_peak - equity_curve) / (running_peak + 1e-8)
    max_drawdown_pct = float(np.max(drawdowns) * 100.0) if len(drawdowns) > 0 else 0.0

    return {
        "Cumulative_Return_Pct": cumulative_return_pct,
        "Annualised_Sharpe_Ratio": sharpe_ratio,
        "Max_Drawdown_Pct": max_drawdown_pct,
        "Active_Trade_Ratio_Pct": float(np.mean(positions != 0.0) * 100.0),
    }


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    missingness_engine: Optional[MissingnessEngine] = None,
    eta: float = 0.0,
    regime: str = "periodic_bernoulli",
    device: Optional[torch.device | str] = None,
) -> Dict[str, object]:
    """
    Evaluates a trained model on `test_loader` under specified dropout rate `eta`
    and missingness `regime`, returning regression metrics, backtest metrics,
    predictions, and attention weights W(t).
    """
    if device is None:
        device = next(model.parameters()).device
    if missingness_engine is None:
        missingness_engine = MissingnessEngine(device=device)

    model.eval()
    preds_list = []
    targets_list = []
    p_curr_list = []
    weights_list = []

    with torch.no_grad():
        for step, batch in enumerate(test_loader):
            X_P, X_S, X_N, X_M, delta_T, S_state, a_close, y_target = [
                t.to(device) for t in batch
            ]
            M_mask = missingness_engine.generate_eval_mask(
                X_P.size(0), step=step, eta=eta, regime=regime, device=device
            )
            P_pred, W = model(X_P, X_S, X_N, X_M, M_mask, delta_T, S_state, a_close)

            preds_list.append(P_pred.cpu().numpy())
            targets_list.append(y_target.cpu().numpy())
            # Current close price is the last time step (index -1) of Channel 3 (Close) in X_OHLCV
            p_curr_list.append(X_P[:, -1, 3:4].cpu().numpy())
            weights_list.append(W.cpu().numpy())

    y_pred = np.concatenate(preds_list, axis=0)
    y_true = np.concatenate(targets_list, axis=0)
    p_curr = np.concatenate(p_curr_list, axis=0)
    weights = np.concatenate(weights_list, axis=0)

    # Verify NFR3 convex bounds
    assert np.allclose(np.sum(weights, axis=-1), 1.0, atol=1e-5), (
        "NFR3 Violation: Weights must sum to 1.0"
    )
    assert np.all(weights >= 0.0), "NFR3 Violation: Negative weights detected"

    reg_metrics = compute_regression_metrics(y_true, y_pred, p_curr)
    bt_metrics = run_financial_backtest(y_true, y_pred, p_curr)

    return {
        "eta": eta,
        "regime": regime,
        "metrics": reg_metrics,
        "backtest": bt_metrics,
        "mean_weights": np.mean(weights, axis=0).tolist(),
        "weights": weights,
        "predictions": y_pred,
        "targets": y_true,
    }

