import numpy as np


def compute_lap_time(path: np.ndarray, v: np.ndarray) -> float:
    if len(path) != len(v):
        raise ValueError(
            f"path and v must have the same length. "
            f"Got path={len(path)}, v={len(v)}"
        )

    diffs = np.diff(path, axis=0, append=path[[0]])
    ds = np.linalg.norm(diffs, axis=1)

    v_next = np.roll(v, -1)
    v_avg = (v + v_next) / 2.0
    v_avg = np.maximum(v_avg, 0.1)

    dt = ds / v_avg

    return float(np.sum(dt))


def format_lap_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    remaining = seconds - minutes * 60
    return f"{minutes}:{remaining:06.3f}"