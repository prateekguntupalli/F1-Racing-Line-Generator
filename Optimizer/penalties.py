import numpy as np

AQUAPLANING_PENALTY_WEIGHT = 50.0
KERB_PENALTY_WEIGHT = 10.0

def aquaplaning_penalty(
    alphas:           np.ndarray,
    aquaplaning_mask: np.ndarray,
) -> float:
    if not np.any(aquaplaning_mask):
        return 0.0

    penalty = AQUAPLANING_PENALTY_WEIGHT * np.sum(aquaplaning_mask.astype(float))
    return float(penalty)

def kerb_penalty(
    alphas:     np.ndarray,
    margin:     float = 0.05,
) -> float:
    near_left  = np.maximum(0.0, margin - alphas)
    near_right = np.maximum(0.0, alphas - (1.0 - margin))
    return float(KERB_PENALTY_WEIGHT * np.sum(near_left + near_right))

def total_penalty(
    alphas:           np.ndarray,
    aquaplaning_mask: np.ndarray,
    condition:        str,
) -> float:
    penalty = kerb_penalty(alphas)

    if condition == "heavy_rain":
        penalty += aquaplaning_penalty(alphas, aquaplaning_mask)

    return penalty