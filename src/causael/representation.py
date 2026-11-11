from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from causael.embed import TextEmbedder
from causael.sae import SAEFeatureModel
from causael.select import SelectionResult


@dataclass
class SAERepresentation:
    """Bundles the full SAE feature matrix with the selection that ran on it, so the unreduced
    (n_docs, M) matrix stays reachable after selection — a custom estimator, dry-run
    diagnostics, or a future re-selection pass can use it without re-running steps 1-2."""

    full_features: np.ndarray
    selected: SelectionResult
    embedder: TextEmbedder | None
    sae: SAEFeatureModel
