from __future__ import annotations

import io
from typing import Optional

import plotly.graph_objects as go
import streamlit as st


def plotly_with_export(
    fig: go.Figure,
    *,
    filename: str,
    caption: Optional[str] = None,
    help_text: Optional[str] = None,
    config: Optional[dict] = None,
) -> None:
    """Render a Plotly figure with download buttons for PNG/JSON exports."""

    cfg = {"displaylogo": False, "responsive": True}
    if config:
        cfg.update(config)
    st.plotly_chart(fig, use_container_width=True, config=cfg)

    if caption:
        st.caption(caption)
    if help_text:
        st.info(help_text)

    export_col, json_col = st.columns(2)

    png_bytes: Optional[bytes] = None
    try:
        png_bytes = fig.to_image(format="png", scale=2)  # requires kaleido
    except Exception as exc:  # pragma: no cover - kaleido missing at runtime
        json_col.warning("PNG export unavailable (install kaleido).", icon="ℹ️")

    if png_bytes:
        export_col.download_button(
            label="Download PNG",
            data=png_bytes,
            file_name=f"{filename}.png",
            mime="image/png",
        )
    else:
        export_col.write("\u00a0")

    json_col.download_button(
        label="Download JSON",
        data=fig.to_json().encode("utf-8"),
        file_name=f"{filename}.json",
        mime="application/json",
    )
