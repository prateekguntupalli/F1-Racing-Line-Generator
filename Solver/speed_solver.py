import numpy as np
from Vehicle.car_model import CarModel
from Weather.conditions import validate as validate_condition


def solve(
    path:        np.ndarray,
    kappa:       np.ndarray,
    grip_map:    np.ndarray,
    car:         CarModel,
    condition:   str,
    lap_type:    str,
    warmup_factors: np.ndarray | None = None,
) -> np.ndarray:
    validate_condition(condition)

    if lap_type not in ("flying", "standing_start"):
        raise ValueError(
            f"Unknown lap_type '{lap_type}'. Must be 'flying' or 'standing_start'."
        )

    n = len(path)
    if warmup_factors is None:
        warmup_factors = np.ones(n)

    effective_grip = grip_map * warmup_factors

    diffs = np.diff(path, axis=0, append=path[[0]])
    ds = np.linalg.norm(diffs, axis=1)
    ds = np.maximum(ds, 1e-6)

    v_corner = car.v_max_cornering(kappa, effective_grip)

    # Global top speed from drag
    v_top = car.v_max_drag()
    v_corner = np.minimum(v_corner, v_top)

    if lap_type == "standing_start":
        v_corner[0] = 0.0

    v = v_corner.copy()

    for i in range(n - 1, -1, -1):
        j = (i + 1) % n
        v_next = v[j]

        decel = car.deceleration(
            np.array([v[i]]),
            effective_grip[[i]],
            kappa[[i]],
        )[0]

        v_from_braking = np.sqrt(np.maximum(0.0, v_next ** 2 + 2.0 * decel * ds[i]))
        v[i] = min(v[i], v_from_braking)

    for i in range(n):
        j = (i + 1) % n

        accel = car.acceleration(
            np.array([v[i]]),
            effective_grip[[i]],
            kappa[[i]],
        )[0]

        v_after_accel = np.sqrt(np.maximum(0.0, v[i] ** 2 + 2.0 * accel * ds[i]))
        v[j] = min(v[j], v_after_accel)

    if lap_type == "standing_start":
        v[0] = 0.0

    return v