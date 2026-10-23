import math

import numpy as np
import pygame

from Solver.lap_time import format_lap_time


CAR_LENGTH_PX = 14
CAR_WIDTH_PX  = 6
CAR_COLOR     = (255, 255, 255)
CAR_OUTLINE   = (20, 20, 20)

SPEED_STEPS = [0.5, 1.0, 2.0, 4.0]

HUD_COLOR = (230, 230, 230)
HUD_POS   = (20, 16)


class CarAnimation:
    def __init__(self, path: np.ndarray, v: np.ndarray):
        self.set_lap(path, v)

        self.playing    = True
        self.speed_idx  = 1  # index into SPEED_STEPS, starts at 1x
        self.clock      = 0.0

        self._font = None  # created lazily so pygame.font is ready by then

    def set_lap(self, path: np.ndarray, v: np.ndarray) -> None:
        if len(path) != len(v):
            raise ValueError(
                f"path and v must have the same length. "
                f"Got path={len(path)}, v={len(v)}"
            )

        old_fraction = 0.0
        if hasattr(self, "lap_time") and self.lap_time > 0:
            old_fraction = self.clock / self.lap_time

        # Close the loop so the last segment runs back to the first point
        self._path = np.vstack([path, path[0]])

        diffs = np.diff(self._path, axis=0)
        ds    = np.linalg.norm(diffs, axis=1)

        v_next = np.roll(v, -1)
        v_avg  = np.maximum((v + v_next) / 2.0, 0.1)  # same guard as lap_time.py
        seg_dt = ds / v_avg

        # _times[i] = time at which the car reaches point i, _times[-1] = lap time
        self._times   = np.concatenate([[0.0], np.cumsum(seg_dt)])
        self._headings = np.arctan2(diffs[:, 1], diffs[:, 0])
        self.lap_time = float(self._times[-1])

        self.clock = old_fraction * self.lap_time

    @property
    def multiplier(self) -> float:
        return SPEED_STEPS[self.speed_idx]

    def update(self, dt: float) -> None:
        """Advance the lap clock. Call once per frame with delta time in seconds."""
        if not self.playing:
            return
        self.clock = (self.clock + dt * self.multiplier) % self.lap_time

    def toggle_play(self) -> None:
        self.playing = not self.playing

    def restart(self) -> None:
        self.clock = 0.0

    def faster(self) -> None:
        self.speed_idx = min(self.speed_idx + 1, len(SPEED_STEPS) - 1)

    def slower(self) -> None:
        self.speed_idx = max(self.speed_idx - 1, 0)

    def handle_key(self, key: int) -> None:
        if key == pygame.K_SPACE:
            self.toggle_play()
        elif key == pygame.K_r:
            self.restart()
        elif key == pygame.K_UP:
            self.faster()
        elif key == pygame.K_DOWN:
            self.slower()

    def state_at(self, t: float) -> tuple:
        t = t % self.lap_time

        i = int(np.searchsorted(self._times, t, side="right")) - 1
        i = max(0, min(i, len(self._headings) - 1))

        seg_len = self._times[i + 1] - self._times[i]
        frac    = (t - self._times[i]) / seg_len if seg_len > 1e-12 else 0.0

        p0 = self._path[i]
        p1 = self._path[i + 1]
        pos = p0 + (p1 - p0) * frac

        return float(pos[0]), float(pos[1]), float(self._headings[i])

    def current_index(self) -> int:
        """Index of the path point the car has most recently passed."""
        i = int(np.searchsorted(self._times, self.clock, side="right")) - 1
        return max(0, min(i, len(self._headings) - 1))

    def draw(self, screen: pygame.Surface, renderer) -> None:
        """Draw the car and a small time readout. Call every frame."""
        x, y, heading = self.state_at(self.clock)
        cx, cy = renderer.world_to_screen((x, y))

        cos_h = math.cos(heading)
        sin_h = math.sin(heading)

        half_l = CAR_LENGTH_PX / 2.0
        half_w = CAR_WIDTH_PX  / 2.0

        # Rectangle corners in the car's own frame (x forward, y sideways),
        # rotated by the heading and moved to the screen position.
        corners = []
        for fx, fy in [(half_l, -half_w), (half_l, half_w), (-half_l, half_w), (-half_l, -half_w)]:
            sx = cx + fx * cos_h - fy * sin_h
            sy = cy + fx * sin_h + fy * cos_h
            corners.append((sx, sy))

        pygame.draw.polygon(screen, CAR_COLOR, corners)
        pygame.draw.polygon(screen, CAR_OUTLINE, corners, 1)

        self._draw_hud(screen)

    def _draw_hud(self, screen: pygame.Surface) -> None:
        if self._font is None:
            self._font = pygame.font.SysFont(None, 24)

        state = "playing" if self.playing else "paused"
        text = (
            f"{format_lap_time(self.clock)} / {format_lap_time(self.lap_time)}"
            f"   {self.multiplier:g}x  {state}"
        )
        surface = self._font.render(text, True, HUD_COLOR)
        screen.blit(surface, HUD_POS)