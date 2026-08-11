import numpy as np
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize


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


def minimum_curvature_guess(
    left_edge:  np.ndarray,
    right_edge: np.ndarray,
    n_points:   int = 100,
    iterations: int = 200,
    step:       float = 0.01,
) -> np.ndarray:
    from Track.curvature import compute_curvature

    alphas = np.full(n_points, 0.5)

    for _ in range(iterations):
        path  = alphas_to_path(alphas, left_edge, right_edge)
        kappa = compute_curvature(path)

        n_edge   = len(left_edge)
        indices  = np.linspace(0, n_edge - 1, n_points).astype(int)
        kappa_cp = kappa[indices]

        gradient = np.zeros(n_points)
        for i in range(n_points):
            alphas_left = alphas.copy()
            alphas_left[i] = np.clip(alphas[i] - step, 0.0, 1.0)
            path_left = alphas_to_path(alphas_left, left_edge, right_edge)
            kappa_left = compute_curvature(path_left)
            kappa_left_cp = kappa_left[indices]

            alphas_right = alphas.copy()
            alphas_right[i] = np.clip(alphas[i] + step, 0.0, 1.0)
            path_right = alphas_to_path(alphas_right, left_edge, right_edge)
            kappa_right = compute_curvature(path_right)
            kappa_right_cp = kappa_right[indices]

            gradient[i] = (kappa_right_cp.sum() - kappa_left_cp.sum()) / (2 * step)

        alphas = np.clip(alphas - step * gradient, 0.0, 1.0)

    return alphas


def optimize_racing_line(
    left_edge: np.ndarray,
    right_edge: np.ndarray,
    kappa: np.ndarray,
    grip_map: np.ndarray,
    car,
    condition: str,
    lap_type: str,
    n_points: int   = 100,
    convergence: float = 0.001,
    max_iter: int   = 500,
    verbose: bool  = True,
) -> tuple:
    
    from Track.curvature import compute_curvature
    from Vehicle.tyre_warmup import compute_warmup_factors
    from Solver.speed_solver import solve
    from Solver.lap_time import compute_lap_time, format_lap_time
    from Optimizer.constraints import build_bounds
    from Optimizer.penalties import total_penalty
    from Weather.aquaplaning import get_aquaplaning_mask_for_alphas

    iteration_count = [0]

    def cost(alphas: np.ndarray) -> float:
        alphas = clamp_alphas(alphas)

        path         = alphas_to_path(alphas, left_edge, right_edge)
        kappa_path   = compute_curvature(path)
        grip         = grip_map

        warmup = None
        if lap_type == "standing_start":
            v_rough = solve(path, kappa_path, grip, car, condition, "flying")
            f_lat   = car.f_lateral(v_rough, kappa_path)
            f_lon   = car.f_longitudinal_available(v_rough, kappa_path, grip)
            warmup  = compute_warmup_factors(f_lat, f_lon, car.mass, condition, lap_type)

        v = solve(path, kappa_path, grip, car, condition, lap_type, warmup)
        lap_time = compute_lap_time(path, v)

        aquaplaning_mask = get_aquaplaning_mask_for_alphas(alphas, left_edge, right_edge, kappa_path)
        penalty = total_penalty(alphas, aquaplaning_mask, condition)

        total = lap_time + penalty

        iteration_count[0] += 1
        if verbose and iteration_count[0] % 10 == 0:
            print(f"  Iter {iteration_count[0]:4d} — lap time: {format_lap_time(lap_time)}  penalty: {penalty:.3f}")

        return total

    print("Computing minimum curvature initial guess...")
    alphas_init = minimum_curvature_guess(left_edge, right_edge, n_points)

    lap_time_init = compute_lap_time(
        alphas_to_path(alphas_init, left_edge, right_edge),
        solve(
            alphas_to_path(alphas_init, left_edge, right_edge),
            kappa, grip_map, car, condition, lap_type,
        )
    )
    print(f"  Initial lap time: {format_lap_time(lap_time_init)}")
    print("Running SQP optimizer...")

    bounds = build_bounds(n_points)

    result = minimize(
        cost,
        alphas_init,
        method  = "SLSQP",
        bounds  = bounds,
        options = {
            "maxiter": max_iter,
            "ftol":    convergence,
        },
    )

    alphas_final = clamp_alphas(result.x)
    path_final   = alphas_to_path(alphas_final, left_edge, right_edge)
    kappa_final  = compute_curvature(path_final)

    warmup_final = None
    if lap_type == "standing_start":
        v_rough = solve(path_final, kappa_final, grip_map, car, condition, "flying")
        f_lat = car.f_lateral(v_rough, kappa_final)
        f_lon = car.f_longitudinal_available(v_rough, kappa_final, grip_map)
        warmup_final = compute_warmup_factors(f_lat, f_lon, car.mass, condition, lap_type)

    v_final = solve(path_final, kappa_final, grip_map, car, condition, lap_type, warmup_final)
    lap_time_final = compute_lap_time(path_final, v_final)

    print(f"\nOptimization complete.")
    print(f"  Converged : {result.success}")
    print(f"  Iterations : {iteration_count[0]}")
    print(f"  Initial time : {format_lap_time(lap_time_init)}")
    print(f"  Final time : {format_lap_time(lap_time_final)}")
    print(f"  Improvement : {lap_time_init - lap_time_final:.3f} s")

    return alphas_final, path_final, lap_time_final