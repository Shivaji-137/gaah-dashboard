from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.data_access import get_dataset, get_metadata
from components.filters import sidebar_filters
from components.layout import inject_css, phase_callout
from components.plot_helpers import plotly_with_export
from utils import (
    build_metadata_frame,
    classify_phase_from_r,
    compute_spacing_ratios,
    estimate_fractal_dimension,
    format_float,
)


def _spectrum_figure(eigvals: np.ndarray, label: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.arange(eigvals.size),
            y=eigvals,
            mode="lines+markers",
            marker=dict(size=5),
            line=dict(width=1.5),
        )
    )
    fig.update_layout(
        title=f"Eigenvalue spectrum — {label}",
        xaxis_title="Eigenvalue index n",
        yaxis_title="Energy E_n",
        template="plotly_dark",
    )
    return fig


def _spacing_histogram(ratios: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=ratios, nbinsx=40, marker=dict(color="#62d5ff")))
    fig.add_vline(x=0.386, line=dict(color="#ff7b72", dash="dash"), annotation_text="Poisson", annotation_position="top left")
    fig.add_vline(x=0.530, line=dict(color="#62d5ff", dash="dash"), annotation_text="GOE", annotation_position="top right")
    fig.update_layout(
        xaxis_title="Spacing ratio r",
        yaxis_title="Counts",
        template="plotly_dark",
    )
    return fig


def _ipr_figure(ipr: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=np.arange(ipr.size), y=ipr, marker=dict(color="#ff7b72")))
    fig.update_layout(
        xaxis_title="Eigenstate index",
        yaxis_title="IPR",
        template="plotly_dark",
    )
    return fig


def main() -> None:
    st.set_page_config(page_title="Spectral Analysis", layout="wide")
    inject_css()
    st.title("Spectral Analysis")
    st.caption("Inspect eigenvalues, spacing statistics, and localization indicators.")

    metadata = get_metadata()
    if not metadata:
        st.info("No datasets available. Add `.npz` outputs to `data/`.")
        return

    meta_df = build_metadata_frame(metadata)
    filters = sidebar_filters(meta_df)
    filtered_df = filters["filtered"]
    dataset_path = filters["dataset_path"]

    if dataset_path is None:
        st.warning("Select at least one dataset from the sidebar.")
        st.dataframe(filtered_df.reset_index(drop=True))
        return

    dataset = get_dataset(dataset_path)
    label = meta_df.set_index("path").loc[dataset_path, "label"]

    eigvals = np.asarray(dataset.get("E", []))
    ipr = np.asarray(dataset.get("IPRs", []))
    mean_r = float(dataset.get("mean_r", np.nan))

    if eigvals.size == 0:
        st.error("Eigenvalues missing in the selected file.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("⟨r⟩ (stored)", format_float(mean_r, 4))
    ratios = compute_spacing_ratios(eigvals)
    col2.metric("⟨r⟩ (recomputed)", format_float(ratios.mean() if ratios.size else float("nan"), 4))
    d2 = estimate_fractal_dimension(ipr, int(filtered_df.loc[filtered_df["path"] == dataset_path, "L"].iloc[0]))
    col3.metric("D₂ estimate", format_float(d2, 3))

    phase_callout(mean_r)

    spectrum_fig = _spectrum_figure(eigvals, label)
    plotly_with_export(spectrum_fig, filename="spectrum")

    if ratios.size:
        plotly_with_export(_spacing_histogram(ratios), filename="spacing_hist")
        st.caption(
            "r ≈ 0.386 → Poisson limit, r ≈ 0.530 → GOE limit. Critical behaviour sits in between."
        )
    else:
        st.info("Spacing ratio histogram requires ≥3 eigenvalues.")

    if ipr.size:
        plotly_with_export(_ipr_figure(ipr), filename="ipr_spectrum")
        st.caption("Large IPR indicates localization; small IPR indicates delocalization.")
    else:
        st.warning("IPR data not stored in this dataset.")

if __name__ == "__main__":
    main()
