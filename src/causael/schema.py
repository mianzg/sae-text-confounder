from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from causael.exceptions import SchemaError


def _is_array_cell(value) -> bool:
    return isinstance(value, (list, tuple, np.ndarray))


def _is_scalar_cell(value) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def column_shape(series: pd.Series) -> tuple[int, ...] | None:
    """None for a uniformly scalar column, else the constant per-cell array shape.

    Raises SchemaError on a null cell, a cell that's neither scalar nor array-like, mixed
    scalar/array cells within one column, or array cells with inconsistent shape.
    """
    kinds: set[str] = set()
    shapes: set[tuple[int, ...]] = set()
    for idx, value in series.items():
        if _is_array_cell(value):
            kinds.add("array")
            shapes.add(tuple(np.shape(value)))
        elif _is_scalar_cell(value):
            if np.isnan(value):
                raise SchemaError(f"column {series.name!r} has a null value at row {idx!r}")
            kinds.add("scalar")
        else:
            raise SchemaError(
                f"column {series.name!r} row {idx!r} is neither a float nor array-like "
                f"(got {type(value).__name__})"
            )
    if len(kinds) > 1:
        raise SchemaError(f"column {series.name!r} mixes scalar and array-valued cells")
    if kinds == {"array"}:
        if len(shapes) > 1:
            raise SchemaError(f"column {series.name!r} has inconsistent array shapes: {sorted(shapes)}")
        return next(iter(shapes))
    return None


def validate_schema(
    df: pd.DataFrame,
    id_col: str,
    text_col: str,
    treatment_col: str,
    outcome_col: str,
    confounder_col: str | None = None,
) -> None:
    """Validate dataframe against required data contract: id, text, treatment, outcome (+ optional
    confounder if provided for simulations). Raises SchemaError naming the offending column/row on the first violation found.
    """
    required = [id_col, text_col, treatment_col, outcome_col]
    if confounder_col is not None:
        required.append(confounder_col)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(f"missing required column(s): {missing}")

    if df[id_col].isnull().any():
        raise SchemaError(f"id column {id_col!r} contains null values")
    if df[id_col].duplicated().any():
        dupes = df.loc[df[id_col].duplicated(), id_col].tolist()
        raise SchemaError(f"id column {id_col!r} has duplicate values: {dupes[:5]}")

    if df[text_col].isnull().any():
        raise SchemaError(f"text column {text_col!r} contains null values")
    if not df[text_col].map(lambda v: isinstance(v, str)).all():
        raise SchemaError(f"text column {text_col!r} must contain only strings")

    column_shape(df[treatment_col])
    column_shape(df[outcome_col])
    if confounder_col is not None:
        column_shape(df[confounder_col])


def read_table(path: str, **kwargs) -> pd.DataFrame:
    """Read a DataFrame from Parquet (array-valued cells preserved) or CSV (scalar-only)."""
    path = str(path)
    if path.endswith(".csv"):
        return pd.read_csv(path, **kwargs)
    return pq.read_table(path, **kwargs).to_pandas()


def write_table(df: pd.DataFrame, path: str, **kwargs) -> None:
    """Write a DataFrame to Parquet (array-valued cells preserved) or CSV (scalar-only)."""
    path = str(path)
    if path.endswith(".csv"):
        df.to_csv(path, index=False, **kwargs)
        return
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path, **kwargs)
