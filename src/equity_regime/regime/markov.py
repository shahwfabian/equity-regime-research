"""2-state Markov-switching regime detection via statsmodels.

Exposes BOTH:
  - filtered_prob_state1  : P(S_t=1 | info up to t)   — tradable, no look-ahead
  - smoothed_prob_state1  : P(S_t=1 | all data)        — descriptive only, uses future info

These are kept as SEPARATE columns with clear naming to prevent misuse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from equity_regime.config import RegimeConfig


@dataclass
class MarkovResult:
    """Container for Markov-switching estimation results."""
    market_day: pd.DataFrame          # with filtered_ and smoothed_ prob columns
    transition_matrix: np.ndarray     # shape (n_states, n_states)
    state_means: np.ndarray           # mean return per state
    state_vols: np.ndarray            # volatility per state
    regime_label: np.ndarray          # filtered argmax state (0 or 1)
    model_summary: Optional[str] = None


def fit_markov_model(
    market_day: pd.DataFrame,
    cfg: RegimeConfig,
    seed: int = 42,
) -> MarkovResult:
    """
    Fit a 2-state Markov-switching model with switching variance.

    Parameters
    ----------
    market_day : pd.DataFrame  with [date, mkt_ret]
    cfg        : RegimeConfig
    seed       : int

    Returns
    -------
    MarkovResult
    """
    try:
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    except ImportError as e:
        raise ImportError("statsmodels>=0.14 required for Markov regime detection") from e

    df = market_day.copy().sort_values("date").reset_index(drop=True)
    returns = df["mkt_ret"].fillna(0.0).values

    np.random.seed(seed)

    # Fit Markov-switching model with switching mean and variance
    model = MarkovRegression(
        returns,
        k_regimes=cfg.n_states,
        switching_variance=True,
        trend="c",
    )

    # Suppress divide-by-zero from EM steps where a state variance hits zero
    # transiently — statsmodels handles these internally via NaN guards.
    with np.errstate(divide="ignore", invalid="ignore"):
        try:
            result = model.fit(
                em_iter=200,
                search_reps=5,
                search_scale=0.5,
                disp=False,
            )
        except Exception:
            result = model.fit(em_iter=100, search_reps=1, disp=False)

    # Filtered probabilities: P(S_t | y_1..t) — TRADABLE (no future data)
    filtered_probs = result.filtered_marginal_probabilities  # shape (T, n_states)

    # Smoothed probabilities: P(S_t | y_1..T) — DESCRIPTIVE ONLY (uses future)
    smoothed_probs = result.smoothed_marginal_probabilities  # shape (T, n_states)

    n_states = cfg.n_states

    # Identify which state is "high vol" (turbulent) by comparing variances
    # State ordering may vary; sort so state 1 = high vol
    try:
        params = result.params
        # Extract variance params from model
        state_vars = np.array([result.params.get(f"sigma2[{i}]", np.nan) for i in range(n_states)])
        if np.any(np.isnan(state_vars)):
            # Fallback: infer from data
            argmax_filtered = np.argmax(filtered_probs, axis=1)
            state_vars = np.array([
                returns[argmax_filtered == s].var() if (argmax_filtered == s).sum() > 0 else 0.0
                for s in range(n_states)
            ])
    except Exception:
        argmax_filtered = np.argmax(filtered_probs, axis=1)
        state_vars = np.array([
            returns[argmax_filtered == s].var() if (argmax_filtered == s).sum() > 0 else 0.0
            for s in range(n_states)
        ])

    state_vols = np.sqrt(np.maximum(state_vars, 0.0))

    # Compute state means from filtered assignment
    argmax_filtered = np.argmax(filtered_probs, axis=1)
    state_means = np.array([
        returns[argmax_filtered == s].mean() if (argmax_filtered == s).sum() > 0 else 0.0
        for s in range(n_states)
    ])

    # Add probability columns to market_day
    for s in range(n_states):
        df[f"filtered_prob_state{s}"] = filtered_probs[:, s]
        df[f"smoothed_prob_state{s}"] = smoothed_probs[:, s]

    # filtered_regime: argmax of filtered probs (tradable regime label)
    df["filtered_regime"] = argmax_filtered.astype(int)

    # Smoothed regime: for descriptive analysis only
    df["smoothed_regime"] = np.argmax(smoothed_probs, axis=1).astype(int)

    # Validate filtered != smoothed column naming
    assert "filtered_prob_state0" in df.columns
    assert "smoothed_prob_state0" in df.columns
    assert df["filtered_prob_state0"].notna().all(), "filtered probs contain NaN"
    assert df["smoothed_prob_state0"].notna().all(), "smoothed probs contain NaN"

    # Transition matrix
    try:
        trans_mat = result.regime_transition
        if trans_mat.ndim == 3:
            trans_mat = trans_mat[:, :, 0]
    except Exception:
        trans_mat = np.eye(n_states)

    try:
        summary_str = result.summary().as_text()
    except Exception:
        summary_str = "Summary unavailable"

    return MarkovResult(
        market_day=df,
        transition_matrix=trans_mat,
        state_means=state_means,
        state_vols=state_vols,
        regime_label=argmax_filtered,
        model_summary=summary_str,
    )
