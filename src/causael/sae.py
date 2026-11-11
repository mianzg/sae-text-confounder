from __future__ import annotations

import os

import numpy as np


class SAEFeatureModel:
    """Trains a sparse autoencoder on dense embeddings (hypothesaes.quickstart.train_sae).

    Train-mode only — the source repo's "load" mode (reading precomputed .npy feature files,
    used by no live caller) is not ported. See docs/refactor-plan.md's "Not ported" table.
    """

    def __init__(
        self,
        n_features: int = 256,
        n_active: int = 16,
        matryoshka_prefix_lengths: list[int] | None = None,
        val_fraction: float = 0.1,
        seed: int = 42,
        checkpoint_dir: str | None = None,
        checkpoint_root: str = "checkpoints",
        run_name: str = "sae_default_run",
    ):
        self.M = int(n_features)
        self.K = int(n_active)
        self.matryoshka_prefix_lengths = matryoshka_prefix_lengths or [self.K, self.M]
        self.val_fraction = float(val_fraction)
        self.seed = int(seed)
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_root = checkpoint_root
        self.run_name = run_name
        self._sae_model = None

    def fit(self, X: np.ndarray) -> SAEFeatureModel:
        """Train on X, shape (n_samples, embedding_dim)."""
        num_samples = len(X)
        rng = np.random.RandomState(self.seed)
        idx = rng.permutation(num_samples)
        split = max(1, int(num_samples * (1 - self.val_fraction)))
        train_idx, val_idx = idx[:split], idx[split:]
        train_embeddings = X[train_idx]
        val_embeddings = X[val_idx] if len(val_idx) > 0 else None

        try:
            from hypothesaes.quickstart import train_sae
        except ImportError as exc:
            raise ImportError("hypothesaes is required for SAE training") from exc

        checkpoint_dir = self.checkpoint_dir or os.path.join(self.checkpoint_root, self.run_name)

        self._sae_model = train_sae(
            embeddings=train_embeddings,
            val_embeddings=val_embeddings,
            M=self.M,
            K=self.K,
            matryoshka_prefix_lengths=self.matryoshka_prefix_lengths,
            checkpoint_dir=checkpoint_dir,
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Dense embeddings X, shape (n_samples, embedding_dim), to (n_samples, M) K-sparse
        SAE activations."""
        if self._sae_model is None:
            raise RuntimeError("SAEFeatureModel not trained. Call fit() first.")
        return self._sae_model.get_activations(X)
