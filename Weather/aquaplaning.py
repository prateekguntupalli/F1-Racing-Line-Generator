import numpy as np
from Track.curvature import compute_curvature


AQUAPLANING_CURVATURE_THRESHOLD = 0.003
AQUAPLANING_OUTSIDE_MARGIN      = 0.7


def detect_aquaplaning_zones(
    path:       np.ndarray,
    kappa:      np.ndarray,
    left_edge:  np.ndarray,
    right_edge: np.ndarray,
    alphas:     np.ndarray,
) -> np.ndarray:
    n    = len(path)
    mask = np.zeros(n, dtype=bool)

    corner_mask = kappa > AQUAPLANING_CURVATURE_THRESHOLD

    for i in range(n):
        if corner_mask[i]:
            left_dist  = np.linalg.norm(path[i] - left_edge[i])
            right_dist = np.linalg.norm(path[i] - right_edge[i])
            half_width = left_dist + right_dist

            if half_width < 1e-6:
                continue

            outside_threshold = half_width * AQUAPLANING_OUTSIDE_MARGIN

            if left_dist > outside_threshold or right_dist > outside_threshold:
                mask[i] = True

    flat_mask = kappa < 1e-5
    mask      = mask | flat_mask

    return mask


def get_aquaplaning_mask_for_alphas(
    alphas:     np.ndarray,
    left_edge:  np.ndarray,
    right_edge: np.ndarray,
    kappa:      np.ndarray,
) -> np.ndarray:
    n_points = len(alphas)
    n_edge   = len(left_edge)
    indices  = np.linspace(0, n_edge - 1, n_points).astype(int)

    kappa_cp  = kappa[indices]
    left_cp   = left_edge[indices]
    right_cp  = right_edge[indices]

    path_cp = np.array([
        (1 - alphas[i]) * left_cp[i] + alphas[i] * right_cp[i]
        for i in range(n_points)
    ])

    return detect_aquaplaning_zones(path_cp, kappa_cp, left_cp, right_cp, alphas)