from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import statsmodels.api as sm
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression, lasso_path


def split(
    X: np.ndarray, T: np.ndarray, seed: int = 42, test_size: float = 0.3
) -> tuple[np.ndarray, np.ndarray]:
    """Train/test split on SAE feature matrix to fit the Lasso path on train and 
    run the conditional independence test on held-out test.

     Returns (train_idx, test_idx)
    """
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=seed)
    return np.asarray(train_idx), np.asarray(test_idx)


def fit_lasso(
    X: np.ndarray, y: np.ndarray, alpha: float, max_iter: int
) -> tuple[LogisticRegression, np.ndarray]:
    """Fit a Lasso logistic regression at a single alpha (`C=alpha`, so *larger* alpha here
    means *less* regularization). 
    Returns (fitted model, column indices with nonzero coefficients)."""

    lasso_model = LogisticRegression(
        l1_ratio=1,  # Lasso penalty
        solver="liblinear",
        C=alpha,
        max_iter=max_iter,
    )
    lasso_model.fit(X, y)
    selected_features = np.where(lasso_model.coef_.flatten() != 0)[0]
    return lasso_model, selected_features


def run_permutation_test(
    X_full: np.ndarray,
    X_reduced: np.ndarray,
    model_reduced,
    lr_stat_obs: float,
    num_perm: int = 200,
) -> float:
    """Permutation test as an alternative to the chi2 asymptotic approximation. 
    Simulates the null `num_perm` times by refitting both the
    reduced and full logistic models on outcomes drawn from the reduced model's fitted
    probabilities. 
    
    Returns a right-tail p-value."""
    rng = np.random.default_rng(0)
    p0 = model_reduced.predict(X_reduced)
    lr_stats_null = np.empty(num_perm, dtype=float)
    for b in range(num_perm):
        y_b = rng.binomial(1, p0, size=len(p0))
        m0_b = sm.Logit(y_b, X_reduced).fit(method="lbfgs", maxiter=1000, disp=False)
        m1_b = sm.Logit(y_b, X_full).fit(method="lbfgs", maxiter=1000, disp=False)
        lr_stats_null[b] = 2 * (m1_b.llf - m0_b.llf)
    return (1 + np.sum(lr_stats_null >= lr_stat_obs)) / (num_perm + 1)


@dataclass
class SelectionResult:
    selected_features: np.ndarray
    reduced_X: np.ndarray
    alpha_trace: list[float] = field(default_factory=list)
    p_value_trace: list[float] = field(default_factory=list)


def select_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    T_train: np.ndarray,
    T_test: np.ndarray,
    *,
    alpha_grid: np.ndarray | None = None,
    significance_level: float = 0.01,
    ci_test: str = "chi2",
) -> SelectionResult:
    """Iteratively select the subset of SAE features for adjusting confounders.
    """

    max_iter = 1000

    # Logistic regression on the full feature set
    model_full = sm.Logit(T_test, X_test).fit(method="lbfgs", maxiter=max_iter, disp=False)
    ll_full = model_full.llf

    if alpha_grid is None:
        alpha_grid = np.logspace(-3, 3, num=100)
    alphas, _coef_path, _ = lasso_path(X_train, T_train, alphas=alpha_grid)

    alpha_trace: list[float] = []
    p_value_trace: list[float] = []
    selected_features = np.arange(X_test.shape[1])

    # Iteratively compare a subset of features (from Lasso) to the full set
    for a in np.flip(alphas):
        _lasso_model, selected_features = fit_lasso(X_train, T_train, a, max_iter)
        X_lasso_test_reduced = X_test[:, selected_features]
        n_selected = len(selected_features)
        if n_selected == 0:
            continue

        model_reduced = sm.Logit(T_test, X_lasso_test_reduced).fit(
            method="lbfgs", maxiter=1000, disp=False
        )
        ll_reduced = model_reduced.llf
        lr_stat_obs = 2 * (ll_full - ll_reduced)

        # Conditional Independence Test 
        if ci_test == "chi2":
            # Calculate degrees of freedom
            dof = X_test.shape[1] - X_lasso_test_reduced.shape[1]
            p_val = 1.0 if dof == 0 else chi2.sf(lr_stat_obs, df=dof)
        elif ci_test == "permutation":
            p_val = run_permutation_test(
                X_test, X_lasso_test_reduced, model_reduced, lr_stat_obs, num_perm=200
            )
        else:
            raise ValueError(f"Unsupported ci_test: {ci_test!r}")

        alpha_trace.append(float(a))
        p_value_trace.append(float(p_val))

        if p_val >= significance_level:
            break

    reduced_X = np.vstack(
        [X_train[:, selected_features], X_test[:, selected_features]]
    )
    return SelectionResult(
        selected_features=selected_features,
        reduced_X=reduced_X,
        alpha_trace=alpha_trace,
        p_value_trace=p_value_trace,
    )
