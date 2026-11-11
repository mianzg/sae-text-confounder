"""Adjust for text-based confounding (e.g. topics) using sparse autoencoders.

The pipeline runs in 4 steps.

1. **Infer Representations** — Text input is encodedas dense text embeddings (:mod:`causael.embed`, :class:`~causael.embed.TextEmbedder`). 
   A sparse autoencoder is trained on those embeddings and is used as the text representation (:mod:`causael.sae`, :class:`~causael.sae.SAEFeatureModel`).

2. **Select** — Iteratively run step 2 and 3 to select a (causally sufficient) subset of SAE features via Lasso and conditional independence testing.
   Regress on the treatment variable (i.e. T) using a subset of SAE features as covariates from no features to all features, and test whether the reduced model is significantly worse than the full model (Step 3).

3. **Test** — a conditional independence test on the selected subset of SAEfeatures (:func:`causael.select.select_features`).

4. **Estimate** — a causal effect on the selected features (:mod:`causael.estimators`).

"""

__version__ = "0.1.0"

from causael.estimator import ConfounderAdjustment, estimate_effect
from causael.results import AdjustmentResult

__all__ = ["AdjustmentResult", "ConfounderAdjustment", "__version__", "estimate_effect"]
