from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import streamlit as st

from utils import DatasetMetadata, discover_datasets, load_npz

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@st.cache_data(show_spinner=False)
def get_metadata() -> List[DatasetMetadata]:
    return [entry for entry in discover_datasets(DATA_DIR) if entry.L > 0]


@st.cache_data(show_spinner=False)
def get_dataset(path: str) -> Dict:
    return load_npz(Path(path))
