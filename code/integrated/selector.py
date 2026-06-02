"""
selector.py — Phase 3: Commodity-level SSE selection with margin guard and Top-K ensembling.
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
    top_k: int = 3,
    historical_scores: dict[str, dict[str, float]] | None = None,
    hist_weight: float = 0.4,
) -> dict[str, dict[str, float]]:
    """
    Per-commodity greedy selection with SSE improvement + dynamic margins, stability penalties,
    Top-K ensembling, and historical rolling SSE.

    Parameters
    ----------
    candidates          : {name: Series indexed (origin, destination, commodity)}
    val_comm_df         : DataFrame with (origin, destination, commodity, tons)
    baseline_name       : name of the baseline candidate (default M0_median_3)
    margins             : base margin float or a dict of commodity -> margin
    stability_penalties : dict of candidate_name -> penalty factor (e.g. 0.1 for 10% penalty)
    top_k               : number of top eligible candidates to ensemble (average)
    historical_scores   : dict {commodity: {candidate_name: sse}}
    hist_weight         : weight float for historical scores [0, 1]

    Returns
    -------
    dict {commodity: {candidate_name: weight}}
    """
    # Align val actuals to index
    val = val_comm_df.set_index(KEY_COMM)["tons"] if "tons" in val_comm_df.columns else \
          val_comm_df.set_index(KEY_COMM).iloc[:, 0]
    val = val.clip(lower=0)

    commodities = val.index.get_level_values("commodity").unique()
    selection: dict[str, dict[str, float]] = {}

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
            selection[comm] = {baseline_name: 1.0}
            continue

        comm_margin = margin_dict.get(comm, 0.15)

        # Baseline SSE with historical blending & penalty
        bp = baseline_pred.reindex(y_true.index).fillna(0.0)
        base_sse_raw = float(((y_true.values - bp.values) ** 2).sum())
        if historical_scores and comm in historical_scores and baseline_name in historical_scores[comm]:
            hist_base_sse = historical_scores[comm][baseline_name]
            base_sse_raw = (1.0 - hist_weight) * base_sse_raw + hist_weight * hist_base_sse

        base_penalty = stability_penalties.get(baseline_name, 0.0)
        baseline_sse = base_sse_raw * (1.0 + base_penalty)

        # Evaluate all candidates and compute penalized SSE
        candidate_scores = []
        for cand_name, cand_series in candidates.items():
            cp = cand_series.reindex(y_true.index).fillna(0.0)
            
            cand_sse_raw = float(((y_true.values - cp.values) ** 2).sum())
            if historical_scores and comm in historical_scores and cand_name in historical_scores[comm]:
                hist_sse = historical_scores[comm][cand_name]
                cand_sse_raw = (1.0 - hist_weight) * cand_sse_raw + hist_weight * hist_sse

            cand_penalty = stability_penalties.get(cand_name, 0.0)
            cand_sse = cand_sse_raw * (1.0 + cand_penalty)

            # Baseline is always eligible
            if cand_name == baseline_name:
                candidate_scores.append((cand_name, cand_sse))
                continue

            # Improvement check: must beat baseline by more than margin
            if baseline_sse > 0:
                improvement = (baseline_sse - cand_sse) / baseline_sse
            else:
                improvement = 0.0 if cand_sse >= 0 else 1.0

            if improvement > comm_margin:
                candidate_scores.append((cand_name, cand_sse))

        # Sort eligible candidates by SSE (ascending)
        candidate_scores.sort(key=lambda x: x[1])

        # Select Top-K eligible candidates
        selected_candidates = candidate_scores[:top_k]
        
        # Assign equal weights to selected candidates
        n_selected = len(selected_candidates)
        weight = 1.0 / n_selected
        selection[comm] = {name: weight for name, _ in selected_candidates}

    return selection


def apply_selection(
    candidates: dict[str, pd.Series],
    selection: dict[str, dict[str, float]],
) -> pd.Series:
    """
    Apply commodity-level ensemble selection to build a single prediction series.

    Parameters
    ----------
    candidates : {name: Series indexed (origin, destination, commodity)}
    selection  : {commodity: {candidate_name: weight}}

    Returns
    -------
    pd.Series indexed (origin, destination, commodity)
    """
    parts = []
    for comm, cand_weights in selection.items():
        comm_parts = []
        for cand_name, weight in cand_weights.items():
            if cand_name not in candidates:
                # Fallback
                cand_name = next(iter(candidates))

            series = candidates[cand_name]

            # Filter to this commodity and apply weight
            try:
                comm_mask = series.index.get_level_values("commodity") == comm
                comm_part = series[comm_mask] * weight
                comm_parts.append(comm_part)
            except Exception:
                continue

        if comm_parts:
            # Sum the weighted components for this commodity
            comm_sum = comm_parts[0]
            for p in comm_parts[1:]:
                comm_sum = comm_sum + p
            parts.append(comm_sum)

    if not parts:
        return pd.Series(dtype=float)

    result = pd.concat(parts)
    result.name = "selected_pred"
    return result
