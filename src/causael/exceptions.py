class SchemaError(ValueError):
    """Raised when a DataFrame does not match causael's data contract."""


class NoConfoundersSelected(RuntimeError):
    """Raised when select_features selects zero features."""


class MissingAPIKeyError(RuntimeError):
    """Raised when embedder='openai' is requested but no API key is resolvable."""
