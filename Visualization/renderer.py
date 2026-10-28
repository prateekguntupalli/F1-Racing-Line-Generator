import pygame
import numpy as np

from Visualization.speedcolormap import colors_for_speeds


TRACK_SURFACE_COLOR = (70, 70, 78)
TRACK_EDGE_COLOR = (230, 230, 230)
RUNOFF_COLOR = (35, 35, 40)

EDGE_LINE_WIDTH = 2
RACING_LINE_RADIUS = 1

START_LINE_COLOR = (255, 255, 255)
START_LINE_WIDTH = 4
SECTOR_LINE_COLOR = (255, 210, 60)
SECTOR_LINE_WIDTH = 2
MARKER_OVERHANG = 1.0 # how far a marker line sticks out past each track edge (fraction of track width)

ARROW_COLOR = (255, 255, 255)
ARROW_DISTANCE_PX = 22 # how far ahead of the start line the arrow sits
ARROW_SIZE_PX = 8
ARROW_LOOKAHEAD = 30 # path points ahead used to work out the direction of travel
SCREEN_MARGIN_PX = 80 # empty space kept around the track when auto-scaling


class TrackRenderer:
    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width  = screen_width
        self.screen_height = screen_height

        self.scale  = 1.0
        self.offset = (0.0, 0.0)

        self._track_layer = None
        self._line_layer = None
        self._marker_layer = None

    def fit(
        self,
        left_edge: np.ndarray,
        right_edge: np.ndarray,
        left_runoff: np.ndarray = None,
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

        # Any previously drawn racing line and markers are at the old scale, so drop them
        self._line_layer = None
        self._marker_layer = None

    def set_racing_line(self, path: np.ndarray, v: np.ndarray) -> None:
        colors = colors_for_speeds(v)
        points = self._to_screen(path)

        self._line_layer = self._new_layer()
        for point, color in zip(points, colors):
            pygame.draw.circle(
                self._line_layer, color,
                (int(point[0]), int(point[1])), RACING_LINE_RADIUS,
            )

    def set_markers(
        self,
        left_edge:   np.ndarray,
        right_edge:  np.ndarray,
        sector_fractions=None,
    ) -> None:
        n = len(left_edge)
        self._marker_layer = self._new_layer()

        # Sector lines go first so the start line draws on top of them
        for fraction in (sector_fractions or []):
            index = int(round(fraction * n)) % n
            self._draw_cross_line(
                self._marker_layer, left_edge[index], right_edge[index],
                SECTOR_LINE_COLOR, SECTOR_LINE_WIDTH,
            )

        self._draw_cross_line(
            self._marker_layer, left_edge[0], right_edge[0],
            START_LINE_COLOR, START_LINE_WIDTH,
        )

        # Direction arrow: centre of the track a few points ahead minus the centre at the start tells us which way the car drives
        centre_start = (np.asarray(left_edge[0])  + np.asarray(right_edge[0])) / 2.0
        ahead        = min(ARROW_LOOKAHEAD, n - 1)
        centre_ahead = (np.asarray(left_edge[ahead]) + np.asarray(right_edge[ahead])) / 2.0

        start_px = np.array(self.world_to_screen(centre_start))
        ahead_px = np.array(self.world_to_screen(centre_ahead))
        direction = ahead_px - start_px
        length = np.linalg.norm(direction)
        if length < 1e-9:
            return
        direction /= length
        sideways = np.array([-direction[1], direction[0]])

        tip  = start_px + direction * (ARROW_DISTANCE_PX + ARROW_SIZE_PX)
        base = start_px + direction * ARROW_DISTANCE_PX
        points = [
            tuple(tip),
            tuple(base + sideways * ARROW_SIZE_PX * 0.8),
            tuple(base - sideways * ARROW_SIZE_PX * 0.8),
        ]
        pygame.draw.polygon(self._marker_layer, ARROW_COLOR, points)

    def _draw_cross_line(self, surface, left_point, right_point, color, width) -> None:
        """Draw a line across the track, sticking out a little past both edges."""
        left  = np.asarray(left_point,  dtype=float)
        right = np.asarray(right_point, dtype=float)
        extra = (right - left) * MARKER_OVERHANG
        a = self.world_to_screen(left  - extra)
        b = self.world_to_screen(right + extra)
        pygame.draw.line(surface, color, a, b, width)

    def world_to_screen(self, point: np.ndarray) -> tuple:
        """Convert a single (x, y) point in metres to screen pixel coordinates."""
        x, y = point
        return (x * self.scale + self.offset[0], y * self.scale + self.offset[1])

    def draw(self, screen: pygame.Surface) -> None:
        """Blit the cached layers. Call every frame."""
        if self._track_layer is not None:
            screen.blit(self._track_layer, (0, 0))
        if self._line_layer is not None:
            screen.blit(self._line_layer, (0, 0))
        if self._marker_layer is not None:
            screen.blit(self._marker_layer, (0, 0))

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