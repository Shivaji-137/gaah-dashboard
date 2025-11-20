from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from utils import format_float


def sidebar_filters(meta_df: pd.DataFrame) -> Dict[str, any]:
    """Render shared sidebar selectors and return filtered views."""

    st.sidebar.header("Controls")
    st.sidebar.write("Filter simulation parameters and choose a dataset.")

    filtered = meta_df.copy()

    L_options = sorted(filtered["L"].dropna().unique().tolist())
    selected_L = st.sidebar.multiselect("Lattice size L", L_options, default=L_options)
    if selected_L:
        filtered = filtered[filtered["L"].isin(selected_L)]

    lam_options = sorted(np.unique(filtered["lam"].dropna().values).tolist())
    selected_lambda = st.sidebar.multiselect(
        "Potential strength λ",
        lam_options,
        default=lam_options,
        format_func=lambda x: format_float(float(x)),
    )
    if selected_lambda:
        mask_lam = np.isclose(filtered["lam"].to_numpy()[:, None], np.asarray(selected_lambda)[None, :], atol=1e-8)
        filtered = filtered[mask_lam.any(axis=1)]

    bc_options = sorted(filtered["bc"].dropna().unique().tolist())
    selected_bc = st.sidebar.multiselect("Boundary condition", bc_options, default=bc_options)
    if selected_bc:
        filtered = filtered[filtered["bc"].isin(selected_bc)]

    phi_options = sorted(np.unique(filtered["phi"].dropna().values).tolist())
    selected_phi = st.sidebar.multiselect(
        "Phase offset φ",
        phi_options,
        default=phi_options,
        format_func=lambda x: format_float(float(x), precision=4),
    )
    if selected_phi:
        mask_phi = np.isclose(filtered["phi"].to_numpy()[:, None], np.asarray(selected_phi)[None, :], atol=1e-8)
        filtered = filtered[mask_phi.any(axis=1)]

    match_count = len(filtered)
    st.sidebar.caption(f"Matching datasets: {match_count}")
    if match_count == 0:
        st.sidebar.warning("No files match the current filters.")

    label_map = dict(zip(meta_df["path"], meta_df["label"]))
    dataset_path: Optional[str] = None
    if match_count > 0:
        dataset_path = st.sidebar.selectbox(
            "Dataset for single-file views",
            filtered["path"].tolist(),
            format_func=lambda p: label_map.get(p, p),
        )

    st.sidebar.divider()

    return {
        "filtered": filtered,
        "dataset_path": dataset_path,
        "selection": {
            "L": selected_L,
            "lam": selected_lambda,
            "bc": selected_bc,
            "phi": selected_phi,
        },
    }
