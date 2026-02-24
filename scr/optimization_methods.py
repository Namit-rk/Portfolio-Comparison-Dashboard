import numpy as np
from scipy.optimize import minimize

# ============================================================
# 1) EQUAL WEIGHT
# ============================================================
def equal_weight(market:object):
    """
    Returns equal allocated weights (1/N)
    """
    n = len(market.tickers)
    return np.ones(n) / n


# ============================================================
# 2) MAX SHARPE (MEAN-VARIANCE / MARKOWITZ)
# ============================================================
def max_sharpe(market:object, rf:float):
    """
    Returns weights which Maximizes Sharpe ratio:
        (w·mu - rf) / sqrt(wᵀΣw)
    """
    mu = market.mu
    cov = market.cov
    n = len(mu)

    def neg_sharpe(w):
        ret = np.dot(w, mu)
        vol = np.sqrt(w.T @ cov @ w)
        if vol == 0:
            return 1e6
        return -(ret - rf) / vol

    constraints = ({
        'type': 'eq',
        'fun': lambda w: np.sum(w) - 1
    })
    bounds = tuple((0.05, 1.0) for _ in range(n))
    init = np.ones(n) / n

    result = minimize(
        neg_sharpe,
        init,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    if not result.success:
        raise RuntimeError(f"Max Sharpe optimization failed: {result.message}")

    return result.x


# ============================================================
# 3) MINIMUM VARIANCE PORTFOLIO
# ============================================================
def min_variance(market:object):
    """
    Returns weights which Minimizes portfolio volatility:
        sqrt(wᵀΣw)
    """
    cov = market.cov
    n = len(cov)

    def portfolio_vol(w):
        return np.sqrt(w.T @ cov @ w)

    constraints = ({
        'type': 'eq',
        'fun': lambda w: np.sum(w) - 1
    })
    bounds = tuple((0.05, 1.0) for _ in range(n))
    init = np.ones(n) / n
    result = minimize(
        portfolio_vol,
        init,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    if not result.success:
        raise RuntimeError(f"Min Variance optimization failed: {result.message}")

    return result.x


# ============================================================
# 4) RISK PARITY
# ============================================================
def risk_parity(market:object):
    """
    Returns weights whichvEqualizes each asset's 
    contribution to portfolio volatility.
    """
    cov = market.cov
    n = len(cov)

    def risk_contribution(w):
        portfolio_var = w.T @ cov @ w
        marginal = cov @ w
        rc = w * marginal / np.sqrt(portfolio_var)
        return rc

    def objective(w):
        rc = risk_contribution(w)
        return np.sum((rc - np.mean(rc)) ** 2)

    constraints = ({
        'type': 'eq',
        'fun': lambda w: np.sum(w) - 1
    })

    bounds = tuple((0.05, 1.0) for _ in range(n))
    init = np.ones(n) / n

    result = minimize(
        objective,
        init,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    if not result.success:
        raise RuntimeError(f"Risk Parity optimization failed: {result.message}")

    return result.x


# ============================================================
# 5) MOMENTUM WEIGHTED PORTFOLIO
# ============================================================
def momentum(market:object, lookback=126):
    """
    Weights proportional to trailing returns (approx 6 months = 126 trading days).
    Long-only momentum.
    """
    # rolling average return
    mom = market.returns.rolling(lookback).mean().iloc[-1]
    # long-only (remove negative momentum)
    mom = np.maximum(mom, 0.0)
    if mom.sum() == 0:
        return np.ones(len(mom)) / len(mom)
    weights = mom / mom.sum()
    return weights.values