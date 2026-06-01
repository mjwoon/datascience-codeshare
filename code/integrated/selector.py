"""
selector.py — Phase 3: Commodity-level greedy SSE selection with margin guard.

greedy_select(candidates, val_comm_df, baseline_name, margin)
    → dict[commodity → best_candidate_name]

apply_selection(candidates, selection)
    → pd.Series indexed (origin, destination, commodity)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KEY_COMM = ["origin", "destination", "commodity"]


def greedy_select(
    candidates: dict[str, pd.Series],
    val_comm_df: pd.DataFrame,
    baseline_name: str = "M0_median_3",
    margins: float | dict[str, float] = 0.15,
    stability_penalties: dict[str, float] | None = None,
) -> dict[str, str]:
    """
    Per-commodity greedy selection with SSE improvement + dynamic margins and stability penalties.

    Parameters
    ----------
    candidates          : {name: Series indexed (origin, destination, commodity)}
    val_comm_df         : DataFrame with (origin, destination, commodity, tons)
    baseline_name       : name of the baseline candidate (default M0_median_3)
    margins             : base margin float or a dict of commodity -> margin
    stability_penalties : dict of candidate_name -> penalty factor (e.g. 0.1 for 10% penalty)

    Returns
    -------
    dict {commodity: best_candidate_name}
    """
    # Align val actuals to index
    val = val_comm_df.set_index(KEY_COMM)["tons"] if "tons" in val_comm_df.columns else \
          val_comm_df.set_index(KEY_COMM).iloc[:, 0]
    val = val.clip(lower=0)

    commodities = val.index.get_level_values("commodity").unique()
    selection: dict[str, str] = {}

    # Get baseline predictions
    if baseline_name not in candidates:
        baseline_name = next(iter(candidates))

    baseline_pred = candidates[baseline_name]
    
    # Determine margins
    if isinstance(margins, float):
        margin_dict = {comm: margins for comm in commodities}
    else:
        margin_dict = margins

    stability_penalties = stability_penalties or {}

    for comm in commodities:
        # Subset to this commodity
        comm_mask = val.index.get_level_values("commodity") == comm
        y_true = val[comm_mask]
        if len(y_true) == 0:
            selection[comm] = baseline_name
            continue

        comm_margin = margin_dict.get(comm, 0.15)

        # Baseline SSE with penalty
        bp = baseline_pred.reindex(y_true.index).fillna(0.0)
        base_penalty = stability_penalties.get(baseline_name, 0.0)
        baseline_sse = float(((y_true.values - bp.values) ** 2).sum()) * (1.0 + base_penalty)

        best_name = baseline_name
        best_sse  = baseline_sse

        # Evaluate all candidates
        for cand_name, cand_series in candidates.items():
            if cand_name == baseline_name:
                continue
            cp = cand_series.reindex(y_true.index).fillna(0.0)
            
            cand_sse_raw = float(((y_true.values - cp.values) ** 2).sum())
            cand_penalty = stability_penalties.get(cand_name, 0.0)
            cand_sse = cand_sse_raw * (1.0 + cand_penalty)

            # Improvement check: must beat baseline by more than margin
            if baseline_sse > 0:
                improvement = (baseline_sse - cand_sse) / baseline_sse
            else:
                improvement = 0.0 if cand_sse >= 0 else 1.0

            if improvement > comm_margin and cand_sse < best_sse:
                best_name = cand_name
                best_sse  = cand_sse

        selection[comm] = best_name

    return selection


def apply_selection(
    candidates: dict[str, pd.Series],
    selection: dict[str, str],
) -> pd.Series:
    """
    Apply commodity-level selection to build a single prediction series.

    Parameters
    ----------
    candidates : {name: Series indexed (origin, destination, commodity)}
    selection  : {commodity: candidate_name}

    Returns
    -------
    pd.Series indexed (origin, destination, commodity)
    """
    parts = []
    for comm, cand_name in selection.items():
        if cand_name not in candidates:
            # Fallback
            for c in candidates:
                cand_name = c
                break

        series = candidates[cand_name]

        # Filter to this commodity
        try:
            comm_mask = series.index.get_level_values("commodity") == comm
            comm_part = series[comm_mask]
        except Exception:
            continue

        parts.append(comm_part)

    if not parts:
        # Return empty series
        return pd.Series(dtype=float)

    result = pd.concat(parts)
    result.name = "selected_pred"
    return result
