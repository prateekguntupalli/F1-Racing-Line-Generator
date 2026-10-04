import numpy as np

COLOR_STOPS = [
    (0.00, (0,   0,   255)),
    (0.25, (0,   200, 255)),
    (0.50, (0,   220, 0)),
    (0.75, (255, 230, 0)),
    (1.00, (255, 0,   0)),
]


def speed_to_color(t: float) -> tuple:
    """Map a normalised speed t in [0, 1] to an (r, g, b) tuple."""
    t = min(max(t, 0.0), 1.0)
    for (t0, c0), (t1, c1) in zip(COLOR_STOPS, COLOR_STOPS[1:]):
        if t <= t1:
            f = (t - t0) / (t1 - t0)
            return tuple(int(c0[k] + f * (c1[k] - c0[k])) for k in range(3))
    return COLOR_STOPS[-1][1]


def colors_for_speeds(v: np.ndarray) -> list:
    """One (r, g, b) per speed value, normalised between the lap's min and max."""
    v_min = float(np.min(v))
    v_max = float(np.max(v))
    span  = max(v_max - v_min, 1e-6)
    return [speed_to_color((s - v_min) / span) for s in v]