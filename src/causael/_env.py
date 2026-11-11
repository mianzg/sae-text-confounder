from __future__ import annotations

import os


def resolve_openai_key() -> str | None:
    """Resolve the OpenAI key, aliasing ``OPENAI_API_KEY`` onto ``OPENAI_KEY_SAE``.

    Required, not a convenience. hypothesaes authenticates *everything* through
    ``llm_api.get_client()``, which reads only ``OPENAI_KEY_SAE`` and raises if it is unset —
    and ``embedding.get_openai_embeddings`` calls that same ``get_client``. hypothesaes never
    reads ``OPENAI_API_KEY`` anywhere. So without this aliasing, a user with the standard
    variable set would fail at step 1 with an error naming a variable they have never heard
    of.

    Returns None when ``OPENAI_API_KEY`` is unset; never logs the value.

    Provenance: centralizes aliasing that `sae-causal` duplicated across
    `compare_pipelines.py::main` and `interpret_selection.py::main`.
    """
    key = os.environ.get("OPENAI_API_KEY")
    if key and not os.environ.get("OPENAI_KEY_SAE"):
        os.environ["OPENAI_KEY_SAE"] = key
    return key
