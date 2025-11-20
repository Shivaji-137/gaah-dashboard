from __future__ import annotations

import pandas as pd
import streamlit as st

from components.data_access import get_metadata
from components.layout import inject_css
from utils import build_metadata_frame, serialise_metadata_table


def main() -> None:
    st.set_page_config(page_title="Data Explorer", layout="wide")
    inject_css()
    st.title("Dataset Explorer")
    st.caption("Inspect available simulations, filter metadata, and download summaries.")

    metadata = get_metadata()
    if not metadata:
        st.info("No datasets available. Add `.npz` outputs to `data/`.")
        return

    meta_df = build_metadata_frame(metadata)
    st.dataframe(meta_df)

    with st.expander("Aggregate statistics", expanded=False):
        grouped = (
            meta_df.groupby(["L", "bc"], as_index=False)
            .agg(dataset_count=("path", "count"), mean_r=("mean_r", "mean"))
            .sort_values(["L", "bc"])
        )
        st.dataframe(grouped)

    st.download_button(
        "Download metadata JSON",
        data=serialise_metadata_table(metadata),
        file_name="gaah_metadata.json",
        mime="application/json",
    )


if __name__ == "__main__":
    main()
