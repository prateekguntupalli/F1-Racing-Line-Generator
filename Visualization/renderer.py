import pygame
import numpy as np
from Visualization.speedcolormap import colors_for_speeds

TRACK_SURFACE_COLOR = (70, 70, 78)
TRACK_EDGE_COLOR     = (230, 230, 230)
RUNOFF_COLOR         = (35, 35, 40)

EDGE_LINE_WIDTH    = 2
RACING_LINE_RADIUS = 1
SCREEN_MARGIN_PX   = 80  # empty space kept around the track when auto-scaling
class TrackRenderer:
    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width  = screen_width
        self.screen_height = screen_height

        self.scale  = 1.0
        self.offset = (0.0, 0.0)

        self._track_layer = None
        self._line_layer  = None

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

        scaled_w = track_width_m  * self.scale
        scaled_h = track_height_m * self.scale

        extra_x = (available_w - scaled_w) / 2.0
        extra_y = (available_h - scaled_h) / 2.0

        offset_x = SCREEN_MARGIN_PX + extra_x - min_x * self.scale
        offset_y = SCREEN_MARGIN_PX + extra_y - min_y * self.scale
        self.offset = (offset_x, offset_y)

        left_px  = self._to_screen(left_edge)
        right_px = self._to_screen(right_edge)

        self._track_layer = self._new_layer()

        if left_runoff is not None and right_runoff is not None:
            self._draw_ribbon(
                self._track_layer,
                self._to_screen(left_runoff),
                self._to_screen(right_runoff),
                RUNOFF_COLOR,
            )

        self._draw_ribbon(self._track_layer, left_px, right_px, TRACK_SURFACE_COLOR)

        pygame.draw.lines(self._track_layer, TRACK_EDGE_COLOR, True, left_px,  EDGE_LINE_WIDTH)
        pygame.draw.lines(self._track_layer, TRACK_EDGE_COLOR, True, right_px, EDGE_LINE_WIDTH)

        # Any previously drawn racing line is at the old scale, so drop it
        self._line_layer = None

    def set_racing_line(self, path: np.ndarray, v: np.ndarray) -> None:
        colors = colors_for_speeds(v)
        points = self._to_screen(path)

        self._line_layer = self._new_layer()
        for point, color in zip(points, colors):
            pygame.draw.circle(
                self._line_layer, color,
                (int(point[0]), int(point[1])), RACING_LINE_RADIUS,
            )

    def world_to_screen(self, point: np.ndarray) -> tuple:
        x, y = point
        return (x * self.scale + self.offset[0], y * self.scale + self.offset[1])

    def draw(self, screen: pygame.Surface) -> None:
        if self._track_layer is not None:
            screen.blit(self._track_layer, (0, 0))
        if self._line_layer is not None:
            screen.blit(self._line_layer, (0, 0))

    def _new_layer(self) -> pygame.Surface:
        return pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)

    def _to_screen(self, points: np.ndarray) -> list:
        return [self.world_to_screen(p) for p in points]

    @staticmethod
    def _draw_ribbon(surface: pygame.Surface, left_px: list, right_px: list, color: tuple) -> None:
        n = len(left_px)
        for i in range(n):
            j = (i + 1) % n
            quad = [left_px[i], left_px[j], right_px[j], right_px[i]]
            pygame.draw.polygon(surface, color, quad)