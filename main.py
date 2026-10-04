import os
os.environ["SDL_VIDEO_WINDOW_POS"] = "centered"

import pygame
import sys

from Track.loader import load_centerline
from Track.geometry import fit_spline, resample, compute_edges
from Track.runoff import generate_runoff
from Track.curvature import compute_curvature
from Track.grip_zones import build_grip_map
from Vehicle.car_model import CarModel
from Vehicle.tyre_warmup import compute_warmup_factors
from Solver.speed_solver import solve
from Solver.lap_time import compute_lap_time, format_lap_time
from Weather.conditions import DRY
from Visualization.renderer import TrackRenderer

WINDOW_WIDTH  = 1400
WINDOW_HEIGHT = 780

BACKGROUND_COLOR = (15, 15, 18)

FPS = 60

CIRCUIT     = "silverstone"
TRACK_WIDTH = 15.0
CONDITION   = DRY
LAP_TYPE    = "flying"


def load_track(circuit_name: str, track_width: float):
    centerline  = load_centerline(circuit_name)
    tck, u      = fit_spline(centerline)
    path        = resample(tck, spacing_m=1.0)
    left, right = compute_edges(path, track_width)
    left_run, right_run = generate_runoff(left, right)
    return path, left, right, left_run, right_run


def compute_lap(path, left, right, car, condition, lap_type):
    kappa    = compute_curvature(path)
    grip_map = build_grip_map(path, left, right, condition)

    warmup = None
    if lap_type == "standing_start":
        v_rough = solve(path, kappa, grip_map, car, condition, "flying")
        f_lat   = car.f_lateral(v_rough, kappa)
        f_lon   = car.f_longitudinal_available(v_rough, kappa, grip_map)
        warmup  = compute_warmup_factors(f_lat, f_lon, car.mass, condition, lap_type)

    v        = solve(path, kappa, grip_map, car, condition, lap_type, warmup)
    lap_time = compute_lap_time(path, v)
    return v, lap_time


def main():
    pygame.init()
    pygame.display.set_caption("F1 Racing Line Generator")

    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock  = pygame.time.Clock()

    print(f"Loading {CIRCUIT}...")
    path, left_edge, right_edge, left_runoff, right_runoff = load_track(CIRCUIT, TRACK_WIDTH)

    car = CarModel(power=0.8, braking=0.7, downforce=0.6, drag=0.4, mass_kg=798.0)

    print("Running speed solver...")
    v, lap_time = compute_lap(path, left_edge, right_edge, car, CONDITION, LAP_TYPE)
    print(f"  Min {v.min() * 3.6:.1f} km/h | Max {v.max() * 3.6:.1f} km/h | Lap {format_lap_time(lap_time)}")

    renderer = TrackRenderer(WINDOW_WIDTH, WINDOW_HEIGHT)
    renderer.fit(left_edge, right_edge, left_runoff, right_runoff)
    renderer.set_racing_line(path, v)

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0  # delta time in seconds

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        screen.fill(BACKGROUND_COLOR)

        renderer.draw(screen)
        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()