import numpy as np

GRIP_COLD        = 0.60
GRIP_HOT         = 1.00

T_INITIAL        = 0.0
T_OPERATING      = 1.0

K_COOL           = 0.02

K_HEAT_TABLE = {
    "sunny":      0.18,
    "dry":        0.15,
    "damp":       0.10,
    "wet":        0.07,
    "heavy_rain": 0.04,
}

def compute_warmup_factors(
    f_lat:     np.ndarray,
    f_lon:     np.ndarray,
    mass:      float,
    condition: str,
    lap_type:  str,
) -> np.ndarray:
    n = len(f_lat)

    if lap_type == "flying":
        return np.ones(n)

    if lap_type != "standing_start":
        raise ValueError(f"Unknown lap_type '{lap_type}'. Must be 'flying' or 'standing_start'.")

    if condition not in K_HEAT_TABLE:
        raise KeyError(
            f"Unknown weather condition '{condition}'. "
            f"Valid options: {list(K_HEAT_TABLE.keys())}"
        )

    k_heat  = K_HEAT_TABLE[condition]
    f_ref_sq = (mass * 9.81) ** 2

    T               = T_INITIAL
    warmup_factors  = np.zeros(n)

    for i in range(n):
        heat_input = k_heat * (f_lat[i] ** 2 + f_lon[i] ** 2) / f_ref_sq

        heat_loss = K_COOL * T

        T = np.clip(T + heat_input - heat_loss, T_INITIAL, T_OPERATING)

        warmup_factors[i] = GRIP_COLD + T * (GRIP_HOT - GRIP_COLD)

    return warmup_factors


def initial_warmup_factor(lap_type: str) -> float:
    if lap_type == "flying":
        return GRIP_HOT
    elif lap_type == "standing_start":
        return GRIP_COLD
    else:
        raise ValueError(f"Unknown lap_type '{lap_type}'.")