import pygame
import numpy as np


TRACK_SURFACE_COLOR = (70, 70, 78)
TRACK_EDGE_COLOR     = (230, 230, 230)
RUNOFF_COLOR         = (35, 35, 40)

EDGE_LINE_WIDTH   = 1
SCREEN_MARGIN_PX  = 80  # empty space kept around the track when auto-scaling


class TrackRenderer:
    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width  = screen_width
        self.screen_height = screen_height

        self.scale  = 1.0
        self.offset = (0.0, 0.0)

        self._left_edge_px  = None
        self._right_edge_px = None
        self._left_runoff_px  = None
        self._right_runoff_px = None

    def fit(
        self,
        left_edge:    np.ndarray,
        right_edge:   np.ndarray,
        left_runoff:  np.ndarray = None,
        right_runoff: np.ndarray = None,
    ) -> None:

        all_points = np.vstack([left_edge, right_edge])
        if left_runoff is not None and right_runoff is not None:
            all_points = np.vstack([all_points, left_runoff, right_runoff])

        min_x, min_y = all_points.min(axis=0)
        max_x, max_y = all_points.max(axis=0)

        track_width_m  = max_x - min_x
        track_height_m = max_y - min_y

        available_w = self.screen_width  - 2 * SCREEN_MARGIN_PX
        available_h = self.screen_height - 2 * SCREEN_MARGIN_PX

        scale_x = available_w / track_width_m  if track_width_m  > 1e-6 else 1.0
        scale_y = available_h / track_height_m if track_height_m > 1e-6 else 1.0

        self.scale = min(scale_x, scale_y)

        # Centre the track: offset maps track-space (min_x, min_y) to the
        # top-left of the margin, then centres any leftover space.
        scaled_w = track_width_m  * self.scale
        scaled_h = track_height_m * self.scale

        extra_x = (available_w - scaled_w) / 2.0
        extra_y = (available_h - scaled_h) / 2.0

        offset_x = SCREEN_MARGIN_PX + extra_x - min_x * self.scale
        offset_y = SCREEN_MARGIN_PX + extra_y - min_y * self.scale
        self.offset = (offset_x, offset_y)

        self._left_edge_px  = self._to_screen(left_edge)
        self._right_edge_px = self._to_screen(right_edge)

        if left_runoff is not None and right_runoff is not None:
            self._left_runoff_px  = self._to_screen(left_runoff)
            self._right_runoff_px = self._to_screen(right_runoff)
        else:
            self._left_runoff_px  = None
            self._right_runoff_px = None

    def world_to_screen(self, point: np.ndarray) -> tuple:
        # Convert a single (x, y) point in metres to screen pixel coords.
        x, y = point
        sx = x * self.scale + self.offset[0]
        sy = y * self.scale + self.offset[1]
        return (sx, sy)

    def _to_screen(self, points: np.ndarray) -> list:
        return [self.world_to_screen(p) for p in points]

    def draw(self, screen: pygame.Surface) -> None:
        # Draw runoff (if set), track surface, and edges. Call every frame.
        if self._left_runoff_px is not None:
            self._draw_ribbon(screen, self._left_runoff_px, self._right_runoff_px, RUNOFF_COLOR)

        self._draw_ribbon(screen, self._left_edge_px, self._right_edge_px, TRACK_SURFACE_COLOR)

        pygame.draw.lines(screen, TRACK_EDGE_COLOR, True, self._left_edge_px, EDGE_LINE_WIDTH)
        pygame.draw.lines(screen, TRACK_EDGE_COLOR, True, self._right_edge_px, EDGE_LINE_WIDTH)

    @staticmethod
    def _draw_ribbon(screen: pygame.Surface, left_px: list, right_px: list, color: tuple) -> None:

        n = len(left_px)
        for i in range(n):
            j = (i + 1) % n
            quad = [left_px[i], left_px[j], right_px[j], right_px[i]]
            pygame.draw.polygon(screen, color, quad)