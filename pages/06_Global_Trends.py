from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Project-specific imports
from components.data_access import get_dataset, get_metadata
from components.filters import sidebar_filters
from components.layout import inject_css
from components.plot_helpers import plotly_with_export
from utils import build_metadata_frame


# ================================
# UTILITIES
# ================================

def _flatten(arrays: Iterable[np.ndarray]) -> np.ndarray:
    """Flatten list of arrays, filtering non-finite values."""
    valid_arrays: List[np.ndarray] = []
    for arr in arrays:
        if arr is None:
            continue
        arr_finite = np.asarray(arr, dtype=float)
        arr_finite = arr_finite[np.isfinite(arr_finite)]
        if arr_finite.size:
            valid_arrays.append(arr_finite)
    return np.concatenate(valid_arrays) if valid_arrays else np.empty(0, dtype=float)


@st.cache_data(show_spinner=True)
def aggregate_datasets(paths: Tuple[str, ...]) -> Dict[str, object]:
    """Load multiple datasets and aggregate key quantities for plotting and analysis."""
    if not paths:
        return {"stats": pd.DataFrame(),
                "energy_map": {}, "ipr_map": {}, "lyapunov_map": {},
                "energy_range": (np.nan, np.nan)}

    energy_map: Dict[float, List[np.ndarray]] = defaultdict(list)
    ipr_map: Dict[str, np.ndarray] = {}
    lyapunov_map: Dict[float, List[np.ndarray]] = defaultdict(list)
    records: List[Dict[str, object]] = []
    e_min, e_max = np.inf, -np.inf

    for path in paths:
        payload = get_dataset(path)
        eigvals = np.asarray(payload.get("E", []), dtype=float)
        ipr = np.asarray(payload.get("IPRs", []), dtype=float)
        lyap = payload.get("lyapunov")

        mean_r = float(payload.get("mean_r", np.nan))
        lam = float(payload.get("lam", np.nan))
        L = int(payload.get("L", -1))
        bc = str(payload.get("bc", ""))
        phi = float(payload.get("phi", np.nan))

        # Aggregate energies
        if eigvals.size and np.isfinite(lam):
            energy_map[lam].append(eigvals)
            e_min = min(e_min, float(np.nanmin(eigvals)))
            e_max = max(e_max, float(np.nanmax(eigvals)))

        # Store IPR
        if ipr.size:
            ipr_map[path] = ipr

        # Aggregate Lyapunov exponent arrays
        if lyap is not None and np.isfinite(lam):
            lyap_arr = np.asarray(lyap, dtype=float)
            if lyap_arr.size:
                lyapunov_map[lam].append(lyap_arr)

        # Summary record
        records.append({
            "path": path, "lam": lam, "L": L, "bc": bc, "phi": phi,
            "mean_r": mean_r if np.isfinite(mean_r) else np.nan,
            "mean_ipr": float(np.nanmean(ipr)) if ipr.size else np.nan,
            "state_count": int(eigvals.size if eigvals.size else ipr.size),
        })

    stats_df = pd.DataFrame.from_records(records)
    if not np.isfinite(e_min) or not np.isfinite(e_max):
        e_min, e_max = np.nan, np.nan

    return {
        "stats": stats_df,
        "energy_map": dict(energy_map),
        "ipr_map": ipr_map,
        "lyapunov_map": dict(lyapunov_map),
        "energy_range": (float(e_min), float(e_max)),
    }


# ================================
# HEATMAPS
# ================================

def _dos_heatmap(energy_map: Dict[float, List[np.ndarray]], energy_edges: np.ndarray) -> go.Figure:
    """Density of states heatmap."""
    lam_values = sorted(energy_map.keys())
    centers = 0.5 * (energy_edges[:-1] + energy_edges[1:])
    dos_rows = []

    for lam in lam_values:
        energies = _flatten(energy_map.get(lam, []))
        hist = np.histogram(energies, bins=energy_edges, density=True)[0] if energies.size else np.zeros_like(centers)
        dos_rows.append(hist)

    z = np.vstack(dos_rows) if dos_rows else np.empty((0, centers.size))

    fig = go.Figure(
        data=go.Heatmap(z=z, x=centers, y=lam_values, colorscale="Viridis",
                         colorbar=dict(title="DOS")))
    fig.update_layout(
        xaxis_title="Energy E", yaxis_title="Potential strength λ",
        template="plotly_dark", margin=dict(l=70, r=40, t=40, b=60))
    return fig


# ================================
# LYAPUNOV PLOT HELPER
# ================================

def plot_lyapunov(
    lyapunov_map: Dict[float, List[np.ndarray]],
    energy_map: Dict[float, List[np.ndarray]],
    edges: np.ndarray | None = None,
    mode: str = "Heatmap"
) -> go.Figure:
    """
    Flexible Lyapunov visualization:
    - Heatmap: requires edges
    - Line plots vs Energy
    - Scatter plot
    - Histogram
    - Average λ_LE vs λ
    """
    fig = go.Figure()
    lam_values = sorted(lyapunov_map.keys())

    if not lam_values:
        return fig

    if mode == "Heatmap":
        if edges is None or edges.size < 2:
            st.warning("Energy edges are required for heatmap mode.")
            return fig
        centers = 0.5 * (edges[:-1] + edges[1:])
        z_rows = []

        for lam in lam_values:
            hist_sum = np.zeros_like(centers, dtype=float)
            hist_count = np.zeros_like(centers, dtype=float)

            for lyap_arr, eig_arr in zip(lyapunov_map[lam], energy_map.get(lam, [])):
                lyap_arr = np.asarray(lyap_arr, dtype=float)
                eig_arr = np.asarray(eig_arr, dtype=float)
                mask = np.isfinite(lyap_arr) & np.isfinite(eig_arr)
                lyap_arr, eig_arr = lyap_arr[mask], eig_arr[mask]
                if eig_arr.size == 0:
                    continue
                counts, _ = np.histogram(eig_arr, bins=edges)
                sums, _ = np.histogram(eig_arr, bins=edges, weights=lyap_arr)
                hist_sum += sums
                hist_count += counts

            avg_lyap = np.divide(hist_sum, hist_count, out=np.zeros_like(hist_sum), where=hist_count > 0)
            z_rows.append(avg_lyap)

        z = np.vstack(z_rows) if z_rows else np.empty((0, centers.size))
        fig.add_trace(go.Heatmap(
            z=z, x=centers, y=lam_values, colorscale="Cividis",
            colorbar=dict(title="⟨λ_LE⟩")
        ))
        fig.update_layout(
            xaxis_title="Energy E", yaxis_title="Potential strength λ",
            template="plotly_dark", margin=dict(l=70, r=40, t=40, b=60)
        )

    elif mode == "Line plots vs Energy":
        for lam in lam_values:
            for lyap_arr, eig_arr in zip(lyapunov_map[lam], energy_map.get(lam, [])):
                lyap_arr = np.asarray(lyap_arr, dtype=float)
                eig_arr = np.asarray(eig_arr, dtype=float)
                mask = np.isfinite(lyap_arr) & np.isfinite(eig_arr)
                lyap_arr, eig_arr = lyap_arr[mask], eig_arr[mask]
                if eig_arr.size == 0:
                    continue
                fig.add_trace(go.Scatter(x=eig_arr, y=lyap_arr,
                                         mode="lines", name=f"λ={lam:.2f}",
                                         line=dict(width=1), opacity=0.5))
        fig.update_layout(
            xaxis_title="Energy E", yaxis_title="Lyapunov exponent λ_LE",
            template="plotly_dark", margin=dict(l=70, r=40, t=40, b=60)
        )

    elif mode == "Scatter plot":
        for lam in lam_values:
            for lyap_arr, eig_arr in zip(lyapunov_map[lam], energy_map.get(lam, [])):
                lyap_arr = np.asarray(lyap_arr, dtype=float)
                eig_arr = np.asarray(eig_arr, dtype=float)
                mask = np.isfinite(lyap_arr) & np.isfinite(eig_arr)
                lyap_arr, eig_arr = lyap_arr[mask], eig_arr[mask]
                if eig_arr.size == 0:
                    continue
                fig.add_trace(go.Scatter(x=eig_arr, y=lyap_arr,
                                         mode="markers", name=f"λ={lam:.2f}",
                                         marker=dict(size=5, opacity=0.6)))
        fig.update_layout(
            xaxis_title="Energy E", yaxis_title="Lyapunov exponent λ_LE",
            template="plotly_dark", margin=dict(l=70, r=40, t=40, b=60)
        )

    elif mode == "Histogram":
        for lam in lam_values:
            lyap_all = np.concatenate([arr for arr in lyapunov_map[lam] if arr.size])
            fig.add_trace(go.Histogram(x=lyap_all, name=f"λ={lam:.2f}", opacity=0.6))
        fig.update_layout(
            xaxis_title="Lyapunov exponent λ_LE", yaxis_title="Count",
            barmode="overlay", template="plotly_dark", margin=dict(l=70, r=40, t=40, b=60)
        )

    elif mode == "Average λ_LE vs λ":
        lam_vals, avg_lyap = [], []
        for lam in lam_values:
            lyap_all = np.concatenate([arr for arr in lyapunov_map[lam] if arr.size])
            if lyap_all.size == 0:
                continue
            lam_vals.append(lam)
            avg_lyap.append(float(np.mean(lyap_all)))
        fig.add_trace(go.Scatter(x=lam_vals, y=avg_lyap, mode="lines+markers",
                                 marker=dict(size=8, color="#ff7b72"), line=dict(width=2)))
        fig.update_layout(
            xaxis_title="λ", yaxis_title="Average Lyapunov exponent",
            template="plotly_dark", margin=dict(l=70, r=40, t=40, b=60)
        )

    return fig



# ================================
# LOCALIZATION METRICS
# ================================

def _mean_series(ipr_map: Dict[str, np.ndarray], stats_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Mean IPR vs lambda."""
    grouped: Dict[float, List[np.ndarray]] = defaultdict(list)
    for lam, subset in stats_df.groupby("lam"):
        if not np.isfinite(lam):
            continue
        for path in subset["path"]:
            arr = ipr_map.get(path)
            if arr is not None and arr.size:
                grouped[lam].append(arr)

    lam_values = sorted(grouped.keys())
    means = [float(np.mean(_flatten(grouped[lam]))) if grouped[lam] else np.nan for lam in lam_values]
    return np.asarray(lam_values), np.asarray(means)


def _fraction_localized(ipr_map: Dict[str, np.ndarray], stats_df: pd.DataFrame, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """Fraction of localized states using IPR*L scaling."""
    lam_values = sorted(stats_df["lam"].dropna().unique())
    fractions: List[float] = []

    for lam in lam_values:
        subset = stats_df[stats_df["lam"] == lam]
        total, localized = 0, 0
        for _, row in subset.iterrows():
            path, L = row["path"], int(row["L"])
            arr = ipr_map.get(path)
            if arr is None or arr.size == 0 or L <= 0:
                continue
            scaled = arr * L
            finite = scaled[np.isfinite(scaled)]
            total += finite.size
            localized += int(np.count_nonzero(finite >= threshold))
        fractions.append(localized / total if total else np.nan)

    return np.asarray(lam_values), np.asarray(fractions)


def _level_spacing(stats_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Mean level-spacing ratio <r> vs lambda."""
    groups = stats_df.dropna(subset=["lam", "mean_r"]).groupby("lam")
    lam_values, means = [], []
    for lam, subset in groups:
        if not np.isfinite(lam):
            continue
        lam_values.append(float(lam))
        means.append(float(subset["mean_r"].mean()))
    sorted_pairs = sorted(zip(lam_values, means))
    return np.asarray([p[0] for p in sorted_pairs]), np.asarray([p[1] for p in sorted_pairs])


# ================================
# STREAMLIT DASHBOARD
# ================================

def main() -> None:
    st.set_page_config(page_title="Global Trends", layout="wide")
    inject_css()
    st.title("Global Spectral Trends")
    st.caption("Aggregate localization diagnostics across λ sweeps and baselines.")

    metadata = get_metadata()
    if not metadata:
        st.info("No datasets available. Add `.npz` outputs to `data/`.")
        return

    meta_df = build_metadata_frame(metadata)
    filters = sidebar_filters(meta_df)
    filtered_df = filters["filtered"]
    if filtered_df.empty:
        st.warning("No datasets survive the current filters; adjust selections on the sidebar.")
        return

    paths = tuple(sorted(filtered_df["path"].tolist()))
    agg = aggregate_datasets(paths)
    stats_df, energy_map, ipr_map, lyapunov_map, (e_min, e_max) = \
        agg["stats"], agg["energy_map"], agg["ipr_map"], agg["lyapunov_map"], agg["energy_range"]
    # Sidebar controls
    st.sidebar.subheader("Global analysis controls")
    ipr_threshold = st.sidebar.slider("Scaled localization threshold (IPR × L ≥ threshold)",
                                      0.0, 10.0, 1.0, 0.1,
                                      help="Localized states have IPR × L ≥ threshold; extended states go to zero as 1/L.")

    bin_count = st.sidebar.slider("Energy bins (DOS / Lyapunov)", 40, 240, 120, 10,
                                  help="Horizontal resolution of heatmaps.")

    energy_window = (e_min, e_max)
    if np.isfinite(e_min) and np.isfinite(e_max) and e_max > e_min:
        energy_window = st.sidebar.slider("Energy window", float(e_min), float(e_max),
                                          value=(float(e_min), float(e_max)),
                                          step=float((e_max - e_min)/200 or 0.01), format="%.3f")
    edges = np.linspace(energy_window[0], energy_window[1], bin_count + 1) \
        if np.isfinite(energy_window[0]) and np.isfinite(energy_window[1]) and energy_window[1] > energy_window[0] else np.array([])

    # --- DOS Heatmap ---
    st.write("### Density of States (DOS λ-map)")
    if energy_map and edges.size > 1:
        dos_fig = _dos_heatmap(energy_map, edges)
        plotly_with_export(dos_fig, filename="dos_lambda_map", caption="Density of States heatmap.")
    else:
        st.info("Eigenvalue data required for the DOS map is missing or energy range is invalid.")

    st.write("### Lyapunov Exponent ⟨λ_LE⟩ Visualization")
    lyap_mode = st.selectbox(
        "Choose Lyapunov visualization type",
        options=["Heatmap", "Line plots vs Energy", "Scatter plot", "Histogram", "Average λ_LE vs λ"],
        index=0,
        help="Select the type of visualization for Lyapunov exponents."
    )

    if lyapunov_map and energy_map and edges.size > 1:
        lyap_fig = plot_lyapunov(lyapunov_map, energy_map, edges=edges, mode=lyap_mode)
        plotly_with_export(
            lyap_fig,
            filename=f"lyapunov_{lyap_mode.replace(' ', '_').lower()}",
            caption=f"Lyapunov exponent visualization ({lyap_mode})."
        )
    else:
        st.info("Lyapunov exponent or eigenvalue data required for the plot is missing or energy range is invalid.")


    # --- Average IPR ---
    st.write("### Average IPR vs. λ")
    lam_vals, avg_ipr = _mean_series(ipr_map, stats_df)
    if lam_vals.size:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=lam_vals, y=avg_ipr, mode="lines+markers",
                                 marker=dict(size=8, color="#ff7b72"), line=dict(width=2)))
        fig.update_layout(
            xaxis_title="λ",
            yaxis_title="⟨IPR⟩",
            template="plotly_dark",
            margin=dict(l=70, r=40, t=40, b=60),
        )
        plotly_with_export(fig, filename="avg_ipr_vs_lambda", caption="Mean IPR across datasets.")
    else:
        st.info("No IPR data available.")

    # --- Fraction Localized ---
    st.write("### Fraction Localized vs. λ")
    lam_frac, frac_values = _fraction_localized(ipr_map, stats_df, ipr_threshold)
    if lam_frac.size:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=lam_frac, y=frac_values, mode="lines+markers",
                                 marker=dict(size=8, color="#62d5ff"), line=dict(width=2)))
        fig.update_layout(
            xaxis_title="λ",
            yaxis_title="Localized fraction",
            template="plotly_dark",
            margin=dict(l=70, r=40, t=40, b=60),
        )
        plotly_with_export(
            fig,
            filename="fraction_localized_vs_lambda",
            caption=f"Fraction of localized states using threshold IPR × L ≥ {ipr_threshold:.2f}.",
        )
    else:
        st.info("Cannot compute localization fraction without IPR arrays.")

    # --- Level spacing ---
    st.write("### Level Spacing Statistics vs. λ")
    lam_r, mean_r = _level_spacing(stats_df)
    if lam_r.size:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=lam_r, y=mean_r, mode="lines+markers",
                                 marker=dict(size=8, color="#fcd34d"), line=dict(width=2)))
        fig.add_hline(y=0.386, line=dict(color="#ff7b72", dash="dash"),
                      annotation=dict(text="Poisson (Localized)", xanchor="left", x=1.05))
        fig.add_hline(y=0.530, line=dict(color="#62d5ff", dash="dash"),
                      annotation=dict(text="GOE (Extended)", xanchor="left", x=1.05))
        fig.update_layout(
            xaxis_title="λ",
            yaxis_title="⟨r⟩",
            template="plotly_dark",
            margin=dict(l=70, r=40, t=40, b=60),
        )
        plotly_with_export(
            fig,
            filename="mean_spacing_vs_lambda",
            caption="Mean nearest-neighbour spacing ratio ⟨r⟩.",
        )
    else:
        st.info("Spacing statistics unavailable; ensure mean_r is stored in source files.")

    st.divider()
    st.subheader("References")
    st.markdown(
        """
**References**
- S. Ganeshan, J. H. Pixley, S. Das Sarma, *PRL* **114**, 146601 (2015) — Exact mobility edge in 1D AA model.
- D. J. Thouless, *Phys. Rep.* **13**, 93 (1974) — DOS and localization review.
- F. Evers, A. D. Mirlin, *RMP* **80**, 1355 (2008) — IPR scaling theory.
        """
    )


if __name__ == "__main__":
    main()
