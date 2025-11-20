from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.data_access import get_dataset, get_metadata
from components.filters import sidebar_filters
from components.layout import inject_css, msd_interpretation
from components.plot_helpers import plotly_with_export
from utils import build_metadata_frame


def _line_plot(times: np.ndarray, values: np.ndarray, y_label: str, color: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=times,
            y=values,
            mode="lines+markers",
            marker=dict(size=4, color=color),
            line=dict(width=2, color=color),
        )
    )
    fig.update_layout(
        xaxis_title="Time t",
        yaxis_title=y_label,
        template="plotly_dark",
    )
    return fig


def main() -> None:
    st.set_page_config(page_title="Dynamics", layout="wide")
    inject_css()
    st.title("Dynamics")
    st.caption("Mean-squared displacement and survival probability from Crank–Nicolson evolution.")

    metadata = get_metadata()
    if not metadata:
        st.info("No datasets available. Add `.npz` outputs to `data/`.")
        return

    meta_df = build_metadata_frame(metadata)
    filters = sidebar_filters(meta_df)
    filtered_df = filters["filtered"]
    dataset_path = filters["dataset_path"]

    if dataset_path is None:
        st.warning("Select a dataset with dynamics from the sidebar.")
        st.dataframe(filtered_df.reset_index(drop=True))
        return

    dataset = get_dataset(dataset_path)
    dynamics = dataset.get("dynamics") or {}
    times = np.asarray(dynamics.get("t", dynamics.get("psi_times", [])))
    msd = np.asarray(dynamics.get("msd", []))
    survival = np.asarray(dynamics.get("survival", []))

    if msd.size == 0 and survival.size == 0:
        st.error("Selected dataset lacks time-evolution observables. Regenerate with `--evolve`.")
        return

    if msd.size:
        st.subheader("Mean-squared displacement σ²(t)")
        plotly_with_export(
            _line_plot(times[: msd.size], msd, "σ²(t)", color="#62d5ff"),
            filename="msd",
        )
        state, message = msd_interpretation(times[: msd.size], msd)
        getattr(st, state)(message)

    if survival.size:
        st.subheader("Survival probability P(t)")
        plotly_with_export(
            _line_plot(times[: survival.size], survival, "P(t)", color="#ff7b72"),
            filename="survival",
        )
        st.caption("Persistent survival indicates localization; decay indicates transport.")

    # Participation Entropy
    psi_samples = dynamics.get("psi_samples")
    psi_times = np.asarray(dynamics.get("psi_times", []))
    
    if psi_samples is not None and len(psi_samples) > 0 and psi_times.size > 0:
        psi_samples = np.asarray(psi_samples)
        # Compute S_p
        probs = np.abs(psi_samples) ** 2
        # Handle log(0) by masking
        # We want p * ln(p). If p=0, limit is 0.
        # We can use np.where to replace 0 with 1 inside the log, so log(1)=0.
        safe_probs = np.where(probs > 1e-20, probs, 1.0)
        entropy = -np.sum(probs * np.log(safe_probs), axis=1)
        
        st.subheader("Participation Entropy S_p(t)")
        plotly_with_export(
            _line_plot(psi_times, entropy, "S_p(t)", color="#2ea043"),
            filename="participation_entropy",
        )
        st.caption("Measures the spread of information. Logarithmic growth often characterizes multifractal phases.")


if __name__ == "__main__":
    main()
