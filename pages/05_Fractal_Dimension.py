from __future__ import annotations

from typing import List

import numpy as np
import plotly.express as px
import streamlit as st

from components.data_access import get_dataset, get_metadata
from components.filters import sidebar_filters
from components.layout import inject_css
from components.plot_helpers import plotly_with_export
from utils import build_metadata_frame, estimate_fractal_dimension

import pandas as pd

MAX_DATASETS = 60  # limit for responsive computation


def _compute_fractal_records(filtered_df) -> List[dict]:
    records: List[dict] = []
    for _, row in filtered_df.iterrows():
        dataset = get_dataset(row["path"])
        ipr = np.asarray(dataset.get("IPRs", []))
        if ipr.size == 0:
            continue
        d2 = estimate_fractal_dimension(ipr, int(row["L"]))
        records.append(
            {
                "path": row["path"],
                "L": row["L"],
                "lam": row["lam"],
                "bc": row["bc"],
                "D2": d2,
            }
        )
    return records


def main() -> None:
    st.set_page_config(page_title="Fractal Dimension", layout="wide")
    inject_css()
    st.title("Fractal-dimension Probe")
    st.caption("Estimate the correlation dimension D₂ from IPR statistics.")

    metadata = get_metadata()
    if not metadata:
        st.info("No datasets available. Add `.npz` outputs to `data/`.")
        return

    meta_df = build_metadata_frame(metadata)
    filters = sidebar_filters(meta_df)
    filtered_df = filters["filtered"]

    if filtered_df.empty:
        st.warning("Adjust the filters to select at least one dataset.")
        return

    if len(filtered_df) > MAX_DATASETS:
        st.warning(f"Fractal analysis limited to the first {MAX_DATASETS} datasets for responsiveness.")
        filtered_df = filtered_df.head(MAX_DATASETS)

    with st.spinner("Computing D₂ estimates..."):
        records = _compute_fractal_records(filtered_df)

    if not records:
        st.error("No usable IPR data found for fractal analysis.")
        return

    result_df = pd.DataFrame(records)

    st.dataframe(result_df)

    fig = px.scatter(
        result_df,
        x="lam",
        y="D2",
        color="L",
        symbol="bc",
        labels={"lam": "λ", "D2": "D₂"},
        template="plotly_dark",
    )
    fig.add_hline(y=1.0, line=dict(color="#62d5ff", dash="dot"))
    plotly_with_export(fig, filename="fractal_dimension")

    st.caption("D₂ ≈ 1 signals extended states; D₂ ≈ 0 signals localization. Intermediate values hint at criticality.")


if __name__ == "__main__":
    main()
