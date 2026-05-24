import numpy as np

#Constants
GRAVITY          = 9.81
MU_BASE          = 1.6

POWER_MIN_W      = 100_000.0
POWER_MAX_W      = 1_000_000.0

BRAKE_MIN        = 0.3
BRAKE_MAX        = 1.0

DOWNFORCE_MIN    = 0.0
DOWNFORCE_MAX    = 3.0

DRAG_MIN         = 0.5
DRAG_MAX         = 3.0

MASS_MIN_KG      = 400.0
MASS_MAX_KG      = 1500.0


class CarModel:

    def __init__(
        self,
        power:     float = 0.5,
        braking:   float = 0.5,
        downforce: float = 0.5,
        drag:      float = 0.5,
        mass_kg: float = 750.0,
    ):
        for name, val in [("power", power), ("braking", braking),("downforce", downforce), ("drag", drag)]:
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"'{name}' must be in [0, 1], got {val}")
        
        if not (MASS_MIN_KG <= mass_kg <= MASS_MAX_KG):
            raise ValueError(f"'mass_kg' must be between {MASS_MIN_KG} and {MASS_MAX_KG}, got {mass_kg}")

        self.power_norm     = power
        self.braking_norm   = braking
        self.downforce_norm = downforce
        self.drag_norm      = drag

        # Derive physical values from normalised inputs
        self.power_w  = POWER_MIN_W   + power     * (POWER_MAX_W   - POWER_MIN_W)
        self.k_brake  = BRAKE_MIN     + braking   * (BRAKE_MAX     - BRAKE_MIN)
        self.k_df     = DOWNFORCE_MIN + downforce * (DOWNFORCE_MAX - DOWNFORCE_MIN)
        self.k_drag   = DRAG_MIN      + drag      * (DRAG_MAX      - DRAG_MIN)
        self.mass = mass_kg

        self.mu_base  = MU_BASE

    def f_max(self, v: np.ndarray, grip_scale: np.ndarray) -> np.ndarray:
        downforce = self.k_df * v ** 2
        return self.mu_base * (self.mass * GRAVITY + downforce) * grip_scale

    def f_lateral(self, v: np.ndarray, kappa: np.ndarray) -> np.ndarray:
        return self.mass * v ** 2 * kappa

    def f_longitudinal_available(
        self,
        v:          np.ndarray,
        kappa:      np.ndarray,
        grip_scale: np.ndarray,
    ) -> np.ndarray:
        f_m   = self.f_max(v, grip_scale)
        f_lat = self.f_lateral(v, kappa)
        return np.sqrt(np.maximum(0.0, f_m ** 2 - f_lat ** 2))

    def v_max_cornering(self, kappa: np.ndarray, grip_scale: np.ndarray) -> np.ndarray:
        mu  = self.mu_base
        m   = self.mass
        g   = GRAVITY

        numerator   = mu * m * g * grip_scale
        denominator = m * kappa - mu * self.k_df * grip_scale

        v_sq = np.where(
            denominator > 1e-6,
            numerator / denominator,
            150.0 ** 2,
        )
        return np.sqrt(np.maximum(0.0, v_sq))

    def v_max_drag(self) -> float:
        return (self.power_w / self.k_drag) ** (1.0 / 3.0)

    def acceleration(self, v: np.ndarray, grip_scale: np.ndarray, kappa: np.ndarray) -> np.ndarray:
        v_safe     = np.maximum(v, 0.1)      # avoid division by zero at standing start
        a_power    = self.power_w / (self.mass * v_safe)
        a_traction = self.f_longitudinal_available(v, kappa, grip_scale) / self.mass
        return np.minimum(a_power, a_traction)

    def deceleration(self, v: np.ndarray, grip_scale: np.ndarray, kappa: np.ndarray) -> np.ndarray:
        f_lon = self.f_longitudinal_available(v, kappa, grip_scale)
        return self.k_brake * f_lon / self.mass

    def __repr__(self) -> str:
        return (
            f"CarModel("
            f"power={self.power_norm:.2f} [{self.power_w/1000:.0f} kW], "
            f"braking={self.braking_norm:.2f}, "
            f"downforce={self.downforce_norm:.2f} [k_df={self.k_df:.2f}], "
            f"drag={self.drag_norm:.2f} [k_drag={self.k_drag:.2f}], "
            f"mass={self.mass:.0f} kg)"
        )