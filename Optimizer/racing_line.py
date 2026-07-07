import numpy as np
from scipy.interpolate import splprep, splev


def build_control_points(
    left_edge:  np.ndarray,
    right_edge: np.ndarray,
    n_points:   int = 100,
) -> np.ndarray:
    return np.full(n_points, 0.5)


def alphas_to_path(
    alphas:     np.ndarray,
    left_edge:  np.ndarray,
    right_edge: np.ndarray,
) -> np.ndarray:
    n_points = len(alphas)
    n_edge   = len(left_edge)

    indices  = np.linspace(0, n_edge - 1, n_points).astype(int)

    control_xy = np.zeros((n_points, 2))
    for i, idx in enumerate(indices):
        control_xy[i] = (1 - alphas[i]) * left_edge[idx] + alphas[i] * right_edge[idx]

    tck, _ = splprep(
        [control_xy[:, 0], control_xy[:, 1]],
        s=0, per=True, k=3,
    )

    u_uniform = np.linspace(0, 1, n_edge)
    x, y      = splev(u_uniform, tck)

    return np.stack([x, y], axis=1)


def clamp_alphas(alphas: np.ndarray) -> np.ndarray:
    return np.clip(alphas, 0.0, 1.0)