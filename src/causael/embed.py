from __future__ import annotations

import contextlib
import os
import sys
import warnings

import numpy as np


@contextlib.contextmanager
def _scoped_emb_cache_dir(cache_dir: str | None):
    """Set EMB_CACHE_DIR for hypothesaes' first import in this process, then restore the
    environment. hypothesaes.embedding.CACHE_DIR is a module-level constant read once at import
    time, not per call, and `import hypothesaes` (e.g. from causael.sae) eagerly imports
    hypothesaes.embedding too — so this only has an effect the *first* time anything in the
    process imports hypothesaes. A later TextEmbedder requesting a different cache_dir has no
    effect on hypothesaes and gets a UserWarning here instead of silently using the wrong
    directory (or worse, appearing to use `cache_dir` while actually reading/writing the first
    one).
    """
    already_imported = "hypothesaes.embedding" in sys.modules
    if already_imported and cache_dir is not None:
        import hypothesaes.embedding as _emb

        frozen = _emb.CACHE_DIR
        if os.path.abspath(frozen) != os.path.abspath(cache_dir):
            warnings.warn(
                f"hypothesaes' embedding cache directory is frozen at {frozen!r} (set by the "
                f"first import of hypothesaes.embedding in this process) and cannot be changed "
                f"to {cache_dir!r} for this TextEmbedder. This is an upstream limitation: "
                f"CACHE_DIR is a module-level constant, not read per call.",
                stacklevel=3,
            )
        yield
        return

    previous = os.environ.get("EMB_CACHE_DIR")
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["EMB_CACHE_DIR"] = cache_dir
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("EMB_CACHE_DIR", None)
        else:
            os.environ["EMB_CACHE_DIR"] = previous


_DEFAULT_MODEL = {
    "openai": "text-embedding-3-small",
    "local": "nomic-ai/modernbert-embed-base",  # hypothesaes.embedding.get_local_embeddings' own default
}


class TextEmbedder:
    """Dense text embeddings via OpenAI or a local model, disk-cached by hypothesaes."""

    def __init__(
        self,
        embedder: str | None = None,
        embedder_type: str = "openai",
        dataset_name: str = "default_dataset",
        cache_name: str | None = None,
        emb_cache_dir: str | None = None,
        batch_size: int = 64,
        n_workers: int = 5,
        chunk_size: int = 50000,
    ):
        # embedder=None picks the right default *for embedder_type* — the source repo's single
        # hardcoded default ("text-embedding-3-small") silently breaks embedder_type="local" (it
        # tries to fetch that OpenAI model name from Hugging Face and 404s).
        self.embedder = embedder or _DEFAULT_MODEL.get(embedder_type, "text-embedding-3-small")
        self.embedder_type = embedder_type
        self.dataset_name = dataset_name
        self.cache_name = cache_name
        # emb_cache_dir=None (the default) respects an already-set EMB_CACHE_DIR rather than
        # forcing "./emb_cache" — matches repr.py's current source (commit a2d0768).
        self.emb_cache_dir = emb_cache_dir if emb_cache_dir is not None else os.environ.get("EMB_CACHE_DIR")
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.chunk_size = chunk_size

    def _resolve_cache_name(self) -> str:
        return self.cache_name or f"{self.dataset_name}_{self.embedder}"

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        """Embed texts, in order. Returns (n_docs, dim)."""
        cache_name = self._resolve_cache_name()

        with _scoped_emb_cache_dir(self.emb_cache_dir):
            try:
                from hypothesaes.embedding import (
                    get_local_embeddings,
                    get_openai_embeddings,
                )
            except ImportError as exc:
                raise ImportError("hypothesaes is required. Run: pip install hypothesaes") from exc

            if self.embedder_type == "openai":
                text2embedding = get_openai_embeddings(
                    texts,
                    batch_size=self.batch_size,
                    n_workers=self.n_workers,
                    chunk_size=self.chunk_size,
                    model=self.embedder,
                    cache_name=cache_name,
                )
            elif self.embedder_type == "local":
                text2embedding = get_local_embeddings(
                    texts, batch_size=self.batch_size, model=self.embedder, cache_name=cache_name
                )
            else:
                raise ValueError(f"Unsupported embedder_type: {self.embedder_type!r}")

        return np.stack([text2embedding[text] for text in texts])
