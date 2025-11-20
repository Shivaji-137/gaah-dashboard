from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.data_access import get_dataset, get_metadata
from components.filters import sidebar_filters
from components.layout import inject_css
from components.plot_helpers import plotly_with_export
from utils import build_metadata_frame

EPS = 1e-12


def _prepare_eigenvectors(raw: object) -> np.ndarray:
    """Coerce serialized eigenvectors into a 2D array [sites, states]."""

    if raw is None:
        return np.empty((0, 0), dtype=np.complex128)

    if isinstance(raw, np.ndarray):
        if raw.size == 0:
            return np.empty((0, 0), dtype=np.complex128)
        if raw.ndim == 2:
            return raw
        if raw.ndim == 1:
            if raw.dtype == object:
                try:
                    stacked = np.column_stack([np.asarray(col).ravel() for col in raw])
                except ValueError:
                    return np.empty((0, 0), dtype=np.complex128)
                return stacked
            return raw.reshape(-1, 1)

    if isinstance(raw, (list, tuple)):
        try:
            return np.column_stack([np.asarray(col).ravel() for col in raw])
        except ValueError:
            return np.empty((0, 0), dtype=np.complex128)

    return np.empty((0, 0), dtype=np.complex128)


def _ipr_scatter(eigvals: np.ndarray, ipr: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=eigvals,
            y=ipr,
            mode="markers",
            marker=dict(
                size=6,
                color=ipr,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="IPR"),
            ),
            name="IPR",
        )
    )

    fig.update_layout(
        xaxis_title="Energy Eₙ",
        yaxis_title="IPR",
        template="plotly_dark",
        margin=dict(l=60, r=30, t=40, b=60),
    )
    return fig


def _eigenstate_heatmap(evecs: np.ndarray, eigvals: np.ndarray, columns: int) -> go.Figure:
    columns = max(1, min(columns, evecs.shape[1], eigvals.size))
    idx = np.argsort(eigvals)[:columns]
    energies = eigvals[idx]
    probs = np.abs(evecs[:, idx]) ** 2
    log_probs = np.log10(probs + EPS)
    fig = go.Figure(
        data=go.Heatmap(
            z=log_probs,
            x=energies,
            y=np.arange(probs.shape[0]),
            colorscale="Turbo",
            colorbar=dict(title="log₁₀|ψₙ|²"),
        )
    )
    fig.update_layout(
        xaxis_title="Energy Eₙ (sorted)",
        yaxis_title="Site index n",
        template="plotly_dark",
        margin=dict(l=60, r=30, t=40, b=60),
    )
    return fig


def _time_animation(probabilities: np.ndarray, times: np.ndarray, stride: int) -> go.Figure:
    stride = max(1, int(stride))
    samples = np.asarray(probabilities, dtype=float)[::stride]
    sampled_times = np.asarray(times, dtype=float)[::stride]
    if samples.size == 0 or sampled_times.size == 0:
        return go.Figure()

    x = np.arange(samples.shape[1])
    frames: list[go.Frame] = []
    for idx, t in enumerate(sampled_times):
        frames.append(
            go.Frame(
                data=[
                    go.Scatter(
                        x=x,
                        y=samples[idx],
                        mode="lines",
                        line=dict(color="#62d5ff", width=3),
                    )
                ],
                name=f"t={t:.2f}",
            )
        )

    base_trace = go.Scatter(
        x=x,
        y=samples[0],
        mode="lines",
        line=dict(color="#62d5ff", width=3),
        name="|ψ|²",
    )

    frame_duration = max(50, int(1000 / max(1, len(frames))))
    play_button = dict(
        label="Play",
        method="animate",
        args=[
            None,
            {
                "frame": {"duration": frame_duration, "redraw": True},
                "fromcurrent": True,
                "transition": {"duration": 0},
            },
        ],
    )
    pause_button = dict(
        label="Pause",
        method="animate",
        args=[
            [None],
            {
                "frame": {"duration": 0, "redraw": False},
                "mode": "immediate",
                "transition": {"duration": 0},
            },
        ],
    )

    slider_steps = [
        dict(
            label=frame.name,
            method="animate",
            args=[
                [frame.name],
                {
                    "mode": "immediate",
                    "frame": {"duration": 0, "redraw": True},
                    "transition": {"duration": 0},
                },
            ],
        )
        for frame in frames
    ]

    fig = go.Figure(data=[base_trace], frames=frames)
    fig.update_layout(
        xaxis_title="Site index n",
        yaxis_title="Probability |ψ(n,t)|²",
        template="plotly_dark",
        margin=dict(l=60, r=30, t=40, b=60),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=[play_button, pause_button],
                pad=dict(t=50),
            )
        ],
        sliders=[
            dict(
                steps=slider_steps,
                currentvalue=dict(prefix="Time "),
                pad=dict(t=20),
            )
        ],
    )
    return fig


def _time_heatmap(probabilities: np.ndarray, times: np.ndarray, stride: int) -> go.Figure:
    stride = max(1, int(stride))
    probs = np.asarray(probabilities, dtype=float)[::stride]
    timeline = np.asarray(times, dtype=float)[::stride]
    if probs.size == 0 or timeline.size == 0:
        return go.Figure()

    z = np.log10(probs + EPS).T  # sites × time
    y = np.arange(z.shape[0])

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=timeline,
            y=y,
            colorscale="Inferno",
            colorbar=dict(title="log₁₀|ψ|²"),
        )
    )

    fig.update_layout(
        xaxis_title="Time t",
        yaxis_title="Site index n",
        template="plotly_dark",
        margin=dict(l=60, r=30, t=40, b=60),
    )
    return fig



def main() -> None:
    st.set_page_config(page_title="Mobility Edge", layout="wide")
    inject_css()
    st.title("Mobility-edge Analysis")
    st.caption("Explore localization structure via IPR maps, eigenstate heatmaps, and time-dynamics animations.")

    metadata = get_metadata()
    if not metadata:
        st.info("No datasets available. Add `.npz` outputs to `data/`.")
        return

    meta_df = build_metadata_frame(metadata)
    filters = sidebar_filters(meta_df)
    filtered_df = filters["filtered"]
    dataset_path = filters["dataset_path"]

    if dataset_path is None:
        st.warning("Select a dataset containing eigenvectors/IPR.")
        st.dataframe(filtered_df.reset_index(drop=True))
        return

    dataset = get_dataset(dataset_path)
    eigvals = np.asarray(dataset.get("E", []), dtype=float)
    ipr = np.asarray(dataset.get("IPRs", []), dtype=float)
    if eigvals.size == 0 or ipr.size == 0:
        st.error("Selected dataset lacks eigenvalues or IPR data.")
        return

    evecs = _prepare_eigenvectors(dataset.get("V"))

    st.subheader("Inverse participation ratio scatter")
    plotly_with_export(
        _ipr_scatter(eigvals, ipr),
        filename="ipr_scatter",
        caption="Color encodes the inverse participation ratio; higher values indicate localization.",
    )

    if evecs.size:
        if evecs.ndim == 1:
            evecs = evecs.reshape(-1, 1)
        max_columns = int(min(eigvals.size, evecs.shape[1], 200))
        if max_columns >= 1:
            heatmap_cols = st.sidebar.slider(
                "Eigenstates displayed in heatmap",
                min_value=1,
                max_value=max_columns,
                value=min(60, max_columns),
            )
            st.subheader("Eigenstate localization heatmap")
            plotly_with_export(
                _eigenstate_heatmap(evecs, eigvals, heatmap_cols),
                filename="eigenstate_heatmap",
                caption="Probabilities shown on a log scale emphasise localized eigenmodes.",
            )
        else:
            st.warning("Eigenvector tensor malformed after loading; unable to render heatmap.")
    else:
        st.warning("Eigenvectors not stored in this file. Re-run the solver with eigenvector output enabled to view the heatmap.")

    dynamics = dataset.get("dynamics") or {}
    psi_samples = np.asarray(dynamics.get("psi_samples", []))
    psi_times = np.asarray(dynamics.get("psi_times", dynamics.get("t", [])))

    if psi_samples.size and psi_times.size:
        if psi_samples.ndim == 1:
            psi_samples = psi_samples.reshape(-1, 1)
        if psi_times.ndim == 0:
            psi_times = psi_times.reshape(1)
        if psi_times.size != psi_samples.shape[0]:
            psi_times = np.arange(psi_samples.shape[0], dtype=float)

        probs = np.abs(psi_samples) ** 2
        stride_max = max(1, probs.shape[0] // 100)

        heatmap_stride = st.sidebar.slider(
            "Heatmap stride",
            min_value=1,
            max_value=max(1, stride_max),
            value=min(5, max(1, stride_max)),
            help="Downsample time samples before animating the heatmap.",
        )
        st.subheader("Time-evolution heatmap")
        plotly_with_export(
            _time_heatmap(probs, psi_times, heatmap_stride),
            filename="time_heatmap",
            caption="Animated space–time map of |ψ(n,t)|² (log scale).",
        )

        L_sites = probs.shape[1]
        if L_sites > 0:
            st.subheader("Probability density heatmap (j/L vs. sample step)")

            stride = max(1, int(heatmap_stride))
            prob_subset = probs[::stride]
            step_indices = np.arange(probs.shape[0], dtype=int)[::stride]

            if L_sites > 1:
                site_fraction = np.arange(L_sites, dtype=float) / float(L_sites)
            else:
                site_fraction = np.zeros(L_sites, dtype=float)

            heatmap_fig = go.Figure(
                data=go.Heatmap(
                    z=prob_subset,
                    x=site_fraction,
                    y=step_indices,
                    colorscale="Viridis",
                    colorbar=dict(title="|ψ_j|²"),
                )
            )
            heatmap_fig.update_layout(
                xaxis_title="j/L",
                yaxis_title="Sample step",
                template="plotly_dark",
                margin=dict(l=70, r=40, t=40, b=60),
            )
            plotly_with_export(
                heatmap_fig,
                filename="probability_heatmap_normalised",
                caption="Probability density with normalized site coordinate j/L.",
            )

        anim_stride = st.sidebar.slider(
            "Animation stride",
            min_value=1,
            max_value=max(1, stride_max),
            value=1,
            help="Display every nᵗʰ sampled frame to control the line animation length.",
        )
        st.subheader("Animated probability flow")
        anim_fig = _time_animation(probs, psi_times, anim_stride)
        st.plotly_chart(anim_fig, use_container_width=True, config={"displaylogo": False})
    else:
        st.info("Selected dataset lacks stored time dynamics; animation skipped.")

if __name__ == "__main__":
    main()
