from __future__ import annotations

import warnings as _warnings

import numpy as np
import pandas as pd

from causael._env import resolve_openai_key
from causael.embed import TextEmbedder
from causael.estimators.base import EffectResult
from causael.estimators.registry import resolve_estimator
from causael.exceptions import MissingAPIKeyError, NoConfoundersSelected
from causael.representation import SAERepresentation
from causael.results import AdjustmentResult
from causael.sae import SAEFeatureModel
from causael.select import select_features, split


def _unadjusted_effect(T: np.ndarray, Y: np.ndarray) -> tuple[float, tuple[float, float]]:
    """Unadjusted (raw) treatment effect.
    """
    T = np.asarray(T)
    Y = np.asarray(Y)
    effect = Y[T == 1].mean() - Y[T == 0].mean()
    stderr = np.mean([Y[T == 1].std(), Y[T == 0].std()]) / np.sqrt(len(Y))
    return float(effect), (float(effect - 1.96 * stderr), float(effect + 1.96 * stderr))


def _auto_reduce_sae_size(n_docs: int, n_features: int, n_active: int) -> tuple[int, int]:
    """First-pass heuristic to avoid mostly-dead latents at small n. TODO: think on small-sample scenario.
    """
    max_reasonable = max(8, n_docs // 8)
    if n_features <= max_reasonable:
        return n_features, n_active
    reduced_features = max(8, 2 ** int(np.floor(np.log2(max_reasonable))))
    reduced_active = max(2, min(n_active, reduced_features // 4))
    return reduced_features, reduced_active


class ConfounderAdjustment:
    """Run full end-to-end adjustment pipeline.

    - ``fit(texts)`` — Get full SAE representations; cached.
    - ``estimate(data, treatment=, outcome=)`` — Select SAE features and estimate causal effect on the selected features.

    ``embedder`` accepts only ``"openai"`` or ``"local"``. Passing a precomputed embedding
    array or a callable embedder is not yet implemented.
    """

    def __init__(
        self,
        n_features: int = 128,
        n_active: int = 8,
        embedder: str = "openai",
        significance_level: float = 0.01,
        ci_test: str = "chi2",
        cutoffs: tuple[float, ...] = (0.5, 0.8),
        random_state: int = 42,
        cache_dir: str | None = None,
        verbose: bool = True,
    ):
        if embedder not in ("openai", "local"):
            raise ValueError(
                f"embedder={embedder!r} not supported yet — only 'openai' and 'local' are "
                f"implemented (precomputed-array / callable embedders are a documented gap)."
            )
        self.n_features = n_features
        self.n_active = n_active
        self.embedder_type = embedder
        self.significance_level = significance_level
        self.ci_test = ci_test
        self.cutoffs = cutoffs
        self.random_state = random_state
        self.cache_dir = cache_dir
        self.verbose = verbose

        self._embedder: TextEmbedder | None = None
        self._sae: SAEFeatureModel | None = None
        self._X_full: np.ndarray | None = None

    def fit(self, texts) -> ConfounderAdjustment:
        texts = list(texts)
        if self.embedder_type == "openai" and resolve_openai_key() is None:
            raise MissingAPIKeyError(
                "embedder='openai' requires OPENAI_API_KEY to be set. Either set it, or pass "
                "embedder='local' to run fully offline (no API key needed)."
            )

        self._embedder = TextEmbedder(embedder_type=self.embedder_type, emb_cache_dir=self.cache_dir)
        X_emb = self._embedder.fit_transform(texts)

        n_features, n_active = _auto_reduce_sae_size(len(texts), self.n_features, self.n_active)
        if n_features != self.n_features and self.verbose:
            _warnings.warn(
                f"n_features={self.n_features} is large relative to n={len(texts)} docs; "
                f"auto-reducing to n_features={n_features}, n_active={n_active} (heuristic). "
                f"Pass n_features= explicitly to override.",
                stacklevel=2,
            )

        self._sae = SAEFeatureModel(
            n_features=n_features, n_active=n_active, seed=self.random_state, checkpoint_dir=self.cache_dir
        )
        self._sae.fit(X_emb)
        self._X_full = self._sae.transform(X_emb)
        return self

    def estimate(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        *,
        method="matching",
        split_idx: tuple[np.ndarray, np.ndarray] | None = None,
        test_size: float = 0.3,
        **estimator_kwargs,
    ) -> AdjustmentResult:
        if self._X_full is None:
            raise RuntimeError("Call fit(texts) before estimate().")

        T = data[treatment].to_numpy()
        Y = data[outcome].to_numpy()
        if len(T) != len(self._X_full):
            raise ValueError(
                f"data has {len(T)} rows but fit() was called on {len(self._X_full)} texts — "
                f"estimate() must be called on the same rows fit() embedded, in the same order."
            )

        train_idx, test_idx = (
            split_idx
            if split_idx is not None
            else split(self._X_full, T, seed=self.random_state, test_size=test_size)
        )
        X_train, X_test = self._X_full[train_idx], self._X_full[test_idx]
        T_train, T_test = T[train_idx], T[test_idx]

        sel_result = select_features(
            X_train, X_test, T_train, T_test, significance_level=self.significance_level, ci_test=self.ci_test
        )
        if len(sel_result.selected_features) == 0:
            raise NoConfoundersSelected(
                "select_features selected zero features. Try: (1) a lower significance_level, "
                "(2) more documents, or (3) a smaller n_features/n_active SAE. "
                f"alpha_trace={sel_result.alpha_trace}, p_value_trace={sel_result.p_value_trace}"
            )

        representation = SAERepresentation(
            full_features=self._X_full, selected=sel_result, embedder=self._embedder, sae=self._sae
        )

        order = np.hstack([train_idx, test_idx])
        T_matched, Y_matched = T[order], Y[order]

        estimator = resolve_estimator(method)
        raw = estimator.estimate(
            sel_result.reduced_X,
            T_matched,
            Y_matched,
            neuron_ids=sel_result.selected_features,
            cutoffs=self.cutoffs,
            random_state=self.random_state,
            **estimator_kwargs,
        )
        if isinstance(raw, dict):
            raw = EffectResult(**raw)

        unadjusted_effect, unadjusted_ci = _unadjusted_effect(T, Y)

        smds = raw.extra.get("smds")
        rows = []
        for neuron in sel_result.selected_features:
            prevalence = float((self._X_full[:, neuron] != 0).mean())
            row = {
                "neuron": int(neuron),
                "prevalence": prevalence,
                "lrt_p_value": sel_result.p_value_trace[-1] if sel_result.p_value_trace else None,
                "smd_before": None,
                "smd_after": None,
            }
            if smds is not None:
                match = smds.loc[smds["neuron"] == neuron]
                if len(match):
                    row["smd_before"] = float(match["pre_match_smd"].iloc[0])
                    row["smd_after"] = float(match["after_match_smd"].iloc[0])
            rows.append(row)
        confounders = pd.DataFrame(rows)

        result_warnings = []
        if "smd_after" in confounders and confounders["smd_after"].notna().any():
            imbalanced = confounders[confounders["smd_after"].abs() > 0.1]
            for _, row in imbalanced.iterrows():
                result_warnings.append(
                    f"feature #{int(row['neuron'])} is still imbalanced after matching "
                    f"(SMD {abs(row['smd_after']):.2f} > 0.10)."
                )

        return AdjustmentResult(
            effect=raw.effect,
            se=raw.se,
            ci=raw.ci,
            unadjusted_effect=unadjusted_effect,
            unadjusted_ci=unadjusted_ci,
            confounders=confounders,
            representation=representation,
            method=method if isinstance(method, str) else type(method).__name__,
            n=len(T),
            warnings=result_warnings,
            extra=raw.extra,
        )


def estimate_effect(
    data: pd.DataFrame,
    text: str,
    treatment: str,
    outcome: str,
    *,
    confounder: str | None = None,
    n_features: int = 128,
    n_active: int = 8,
    embedder: str = "openai",
    significance_level: float = 0.01,
    ci_test: str = "chi2",
    method="matching",
    cutoffs: tuple[float, ...] = (0.5, 0.8),
    random_state: int = 42,
    cache_dir: str | None = None,
    verbose: bool = True,
    **estimator_kwargs,
) -> AdjustmentResult:
    """One-line wrapper over ConfounderAdjustment.fit()+.estimate(). `confounder` is schema-valid
    but not used by the built-in estimators — it's for simulation/evaluation callers only."""
    required_cols = [text, treatment, outcome] + ([confounder] if confounder else [])
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"missing column(s) in data: {missing}")

    adj = ConfounderAdjustment(
        n_features=n_features,
        n_active=n_active,
        embedder=embedder,
        significance_level=significance_level,
        ci_test=ci_test,
        cutoffs=cutoffs,
        random_state=random_state,
        cache_dir=cache_dir,
        verbose=verbose,
    )
    adj.fit(data[text])
    return adj.estimate(data, treatment=treatment, outcome=outcome, method=method, **estimator_kwargs)
