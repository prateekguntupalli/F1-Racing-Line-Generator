import pygame
import sys
import os

os.environ["SDL_VIDEO_WINDOW_POS"] = "centered"

from Track.loader import load_centerline
from Track.geometry import fit_spline, resample, compute_edges
from Track.runoff import generate_runoff
from Visualization.renderer import TrackRenderer

WINDOW_WIDTH  = 1400
WINDOW_HEIGHT = 780

BACKGROUND_COLOR = (15, 15, 18)

FPS = 60

CIRCUIT      = "silverstone"
TRACK_WIDTH  = 15.0


def load_track(circuit_name: str, track_width: float):
    centerline  = load_centerline(circuit_name)
    tck, u      = fit_spline(centerline)
    path        = resample(tck, spacing_m=1.0)
    left, right = compute_edges(path, track_width)
    left_run, right_run = generate_runoff(left, right)
    return left, right, left_run, right_run


def main():
    pygame.init()
    pygame.display.set_caption("F1 Racing Line Generator")

    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock  = pygame.time.Clock()

    print(f"Loading {CIRCUIT}...")
    left_edge, right_edge, left_runoff, right_runoff = load_track(CIRCUIT, TRACK_WIDTH)

    renderer = TrackRenderer(WINDOW_WIDTH, WINDOW_HEIGHT)
    renderer.fit(left_edge, right_edge, left_runoff, right_runoff)

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

        # Remaining pieces to add here as each Phase 4 card is completed:
        # - Racing line overlay
        # - Car animation
        # - Telemetry panel
        # - UI controls

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()