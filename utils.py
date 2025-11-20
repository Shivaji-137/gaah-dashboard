from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetMetadata:
    """Lightweight container for metadata discovered in a simulation .npz file."""

    path: Path
    L: int
    lam: float
    bc: str
    phi: float
    mean_r: float
    eig_method: str
    has_dynamics: bool

    @property
    def label(self) -> str:
        return f"L={self.L} | λ={self.lam:.3f} | φ={self.phi:.3f} | bc={self.bc}"


def _normalise_dynamics(payload: Any) -> Dict[str, Any]:
    """Convert the serialized dynamics payload back into a standard dict."""

    if payload is None:
        return {}
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, np.ndarray) and payload.dtype == object:
        try:
            return payload.item()
        except ValueError:
            # Fall back to list conversion for unexpected shapes
            return {"data": payload.tolist()}
    if isinstance(payload, np.void):  # pragma: no cover - defensive branch
        return dict(zip(payload.dtype.names, payload.tolist()))
    return {}


def load_npz(path: Path) -> Dict[str, Any]:
    """Load a simulation result file, returning plain Python / NumPy objects."""

    with np.load(path, allow_pickle=True) as data:
        payload: Dict[str, Any] = {key: data[key] for key in data.files}
    payload["dynamics"] = _normalise_dynamics(payload.get("dynamics"))
    payload["path"] = Path(path)
    return payload


def discover_datasets(data_dir: Path) -> List[DatasetMetadata]:
    """Scan ``data_dir`` for .npz files and extract lightweight metadata."""

    if not data_dir.exists():
        return []

    metadata: List[DatasetMetadata] = []
    for npz_path in sorted(data_dir.glob("*.npz")):
        try:
            record = load_npz(npz_path)
        except (IOError, ValueError) as exc:
            # Skip corrupted files but keep a small breadcrumb for debugging.
            error_stub = DatasetMetadata(
                path=npz_path,
                L=-1,
                lam=float("nan"),
                bc="invalid",
                phi=float("nan"),
                mean_r=float("nan"),
                eig_method="",
                has_dynamics=False,
            )
            metadata.append(error_stub)
            print(f"[warn] Failed to load {npz_path}: {exc}")
            continue

        dyn = record.get("dynamics", {})
        has_dyn = bool(dyn) and any(key in dyn for key in ("psi_samples", "msd", "survival"))

        metadata.append(
            DatasetMetadata(
                path=npz_path,
                L=int(record.get("L", -1)),
                lam=float(record.get("lam", np.nan)),
                bc=str(record.get("bc", "unknown")),
                phi=float(record.get("phi", np.nan)),
                mean_r=float(record.get("mean_r", np.nan)),
                eig_method=str(record.get("eig_method", "")),
                has_dynamics=has_dyn,
            )
        )

    return metadata


def serialise_metadata_table(entries: Iterable[DatasetMetadata]) -> str:
    """Return a JSON payload for quick inspection/download in the UI."""

    table = [
        {
            "path": str(item.path),
            "L": item.L,
            "lambda": item.lam,
            "bc": item.bc,
            "phi": item.phi,
            "mean_r": item.mean_r,
            "eig_method": item.eig_method,
            "has_dynamics": item.has_dynamics,
        }
        for item in entries
    ]
    return json.dumps(table, indent=2)


def build_metadata_frame(entries: Sequence[DatasetMetadata]) -> pd.DataFrame:
    """Convert metadata entries into a tidy pandas DataFrame."""

    return pd.DataFrame(
        [
            {
                "path": str(item.path),
                "label": item.label,
                "L": item.L,
                "lam": item.lam,
                "bc": item.bc,
                "phi": item.phi,
                "mean_r": item.mean_r,
                "eig_method": item.eig_method,
                "has_dynamics": item.has_dynamics,
            }
            for item in entries
        ]
    )


def compute_spacing_ratios(eigenvalues: np.ndarray) -> np.ndarray:
    """Compute nearest-neighbour spacing ratios from a sorted eigenvalue array."""

    spectrum = np.asarray(eigenvalues, dtype=float)
    if spectrum.size < 3:
        return np.empty(0)
    spacings = np.diff(spectrum)
    # Avoid division by zero by masking small gaps.
    mask = (spacings[:-1] > 0) & (spacings[1:] > 0)
    safe_spacings = np.minimum(spacings[:-1], spacings[1:])[mask] / np.maximum(
        spacings[:-1], spacings[1:]
    )[mask]
    return safe_spacings


def format_float(value: float, precision: int = 3) -> str:
    """Helper for consistent float formatting in the UI."""

    if np.isnan(value):
        return "NaN"
    return f"{value:.{precision}f}"


def classify_phase_from_r(mean_r: float, tol: float = 0.015) -> str:
    """Return a qualitative phase label based on the average spacing ratio."""

    if not np.isfinite(mean_r):
        return "undetermined"
    if abs(mean_r - 0.386) <= tol:
        return "Poisson limit"
    if abs(mean_r - 0.530) <= tol:
        return "GOE limit"
    if mean_r < 0.43:
        return "Insulating regime"
    if mean_r > 0.5:
        return "Extended regime"
    return "Critical regime"


def estimate_fractal_dimension(ipr: np.ndarray, L: int, q: float = 2.0) -> float:
    """Estimate the generalized fractal dimension D_q from IPR data.

    For q=2 the relation IPR ~ L^{-D2}. We use a simple log scaling of the
    mean IPR as a proxy, suitable for exploratory analysis with a single L.
    """

    ipr = np.asarray(ipr, dtype=float)
    ipr = ipr[np.isfinite(ipr) & (ipr > 0)]
    if ipr.size == 0 or L <= 1:
        return float("nan")
    mean_ipr = float(np.mean(ipr))
    if mean_ipr <= 0:
        return float("nan")
    return -np.log(mean_ipr) / (np.log(L) * (q - 1)) if q != 1 else float("nan")

def analytic_mobility_edge(lam: float, t1: float = 1.0, alpha: float = 0.5) -> tuple[float, float]:
    """Return ±E_c for the generalized Aubry–André (GAA) mobility edge.

    The closed-form expression follows Ganeshan *et al.* (arXiv:1411.7375):
    ``E_c = 2·sgn(λ)·(|t| - |λ|) / α``.

    Parameters
    ----------
    lam : float
        Onsite modulation strength λ.
    t1 : float, default 1.0
        Nearest-neighbour hopping amplitude t.
    alpha : float, default 0.5
        Deformation parameter α (|α| < 1).

    Returns
    -------
    (float, float)
        Tuple ``(-|E_c|, |E_c|)``. NaNs returned if α≈0 to avoid division by zero.
    """

    if not np.isfinite(lam) or not np.isfinite(t1) or not np.isfinite(alpha):
        return float("nan"), float("nan")

    if abs(alpha) < 1e-9:
        return float("nan"), float("nan")

    prefactor = 2.0 * np.sign(lam) * (abs(t1) - abs(lam)) / alpha
    Ec = float(prefactor)
    bound = abs(Ec)
    return -bound, bound



