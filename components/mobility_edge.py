import numpy as np
from dataclasses import dataclass

@dataclass
class MobilityEdgeResult:
    smoothed_ipr: np.ndarray
    threshold: float
    crossings: list
    explanation: str

def estimate_mobility_edge(eigvals: np.ndarray, ipr: np.ndarray, threshold: float = None, smoothing: int = 5) -> MobilityEdgeResult:
    """
    Estimate mobility edges using smoothed IPR.

    Parameters
    ----------
    eigvals : np.ndarray
        Eigenvalues.
    ipr : np.ndarray
        Inverse participation ratios.
    threshold : float, optional
        IPR threshold to distinguish localized vs extended states. Default: median.
    smoothing : int
        Moving average window for smoothing IPR.

    Returns
    -------
    MobilityEdgeResult
        Contains smoothed IPR, threshold, crossings, explanation.
    """
    if eigvals.size == 0 or ipr.size == 0:
        return MobilityEdgeResult(np.array([]), threshold, [], "Empty eigenvalues/IPR array.")

    # Sort eigenvalues and reorder IPR accordingly
    idx_sort = np.argsort(eigvals)
    eigvals_sorted = eigvals[idx_sort]
    ipr_sorted = ipr[idx_sort]

    # Apply simple moving average
    smoothed_ipr = np.convolve(ipr_sorted, np.ones(smoothing)/smoothing, mode='same')

    # Use median if threshold not given
    if threshold is None:
        threshold = float(np.median(smoothed_ipr))

    # Detect crossings
    localized = smoothed_ipr >= threshold
    crossings_idx = np.where(np.diff(localized.astype(int)) != 0)[0]
    crossings = [(eigvals_sorted[i] + eigvals_sorted[i+1])/2 for i in crossings_idx]

    explanation = (
        f"Mobility edges estimated using IPR threshold {threshold:.5f} "
        f"and smoothing window {smoothing}. "
        f"Red vertical lines indicate energy values where states change from extended ↔ localized."
    )

    return MobilityEdgeResult(
        smoothed_ipr=smoothed_ipr,
        threshold=threshold,
        crossings=crossings,
        explanation=explanation
    )
