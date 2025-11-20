from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import streamlit as st

from utils import classify_phase_from_r, format_float


STYLE_PATH = Path(__file__).resolve().parent.parent / "theme" / "style.css"


def inject_css(path: Path = STYLE_PATH) -> None:
    """Apply the custom dark theme across pages."""

    if path.exists():
        st.markdown(f"<style>{path.read_text()}</style>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="footer">
            Copyright &copy; Shivaji Chaulagain
        </div>
        """,
        unsafe_allow_html=True,
    )


def phase_callout(mean_r: float) -> None:
    """Display a badge linking ⟨r⟩ to the expected phase."""

    label = classify_phase_from_r(mean_r)
    spacing_text = f"Mean spacing ratio ≈ {format_float(mean_r, 4)}"

    if label == "Poisson limit":
        st.error("Phase classification: Poisson limit")
        st.caption(spacing_text)
    elif label == "GOE limit":
        st.success("Phase classification: GOE limit")
        st.caption(spacing_text)
    elif label == "Insulating regime":
        st.error("Phase classification: Insulating regime")
        st.caption(spacing_text)
    elif label == "Extended regime":
        st.success("Phase classification: Extended regime")
        st.caption(spacing_text)
    elif label == "Critical regime":
        st.warning("Phase classification: Critical regime")
        st.caption(spacing_text)
    else:
        st.info("Phase classification undetermined; require more statistics.")


def msd_interpretation(times: np.ndarray, msd: np.ndarray) -> Tuple[str, str]:
    """Return qualitative interpretation of MSD evolution."""

    if msd.size < 2 or times.size < 2:
        return ("info", "Insufficient MSD samples for trend analysis.")

    tail = msd[-min(10, msd.size):]
    tail_times = times[-tail.size:]
    slope = np.polyfit(tail_times, tail, 1)[0]
    if abs(slope) < 1e-3:
        return ("error", "MSD saturates → localization.")
    if slope < 0:
        return ("warning", "MSD decreases; check normalization or time step.")
    return ("success", "MSD continues to grow → extended/transporting regime.")
