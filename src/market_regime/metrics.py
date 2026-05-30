import numpy as np
import pandas as pd


def annualized_sharpe(returns, periods_per_year=252):
    """Compute annualized Sharpe ratio from periodic returns."""
    returns = pd.Series(returns).dropna()
    if returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))


def max_drawdown(returns):
    """Compute maximum drawdown from periodic returns."""
    returns = pd.Series(returns).fillna(0)
    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1
    return float(drawdown.min())


def strategy_returns(actual_returns, probabilities, threshold=0.45):
    """Long/cash strategy return from positive-direction probabilities."""
    signal = (np.asarray(probabilities) >= threshold).astype(float)
    return signal * np.asarray(actual_returns)
