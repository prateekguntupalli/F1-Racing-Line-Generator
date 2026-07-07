import numpy as np


def build_bounds(n_points: int) -> list[tuple[float, float]]:
    return [(0.0, 1.0)] * n_points


def is_within_bounds(alphas: np.ndarray) -> bool:
    return bool(np.all(alphas >= 0.0) and np.all(alphas <= 1.0))


def violations(alphas: np.ndarray) -> np.ndarray:
    below = np.maximum(0.0, -alphas)
    above = np.maximum(0.0, alphas - 1.0)
    return below + above