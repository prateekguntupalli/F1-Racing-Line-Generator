import math

import numpy as np
import pygame

from Solver.lap_time import format_lap_time
from Visualization.speedcolormap import colors_for_speeds


PANEL_COLOR = (24, 24, 30)
BORDER_COLOR = (55, 55, 66)
TEXT_COLOR = (235, 235, 240)
MUTED_COLOR = (130, 130, 145)
GRID_COLOR = (48, 48, 58)
CURSOR_COLOR = (255, 255, 255)

PADDING = 20
TRACE_HEIGHT = 190
AXIS_LABEL_W = 38 # space left of the trace for the km/h labels
TOP_PAD = 10 # space above the trace so the top km/h label isn't clipped
TIME_LABEL_H = 20 # space under the trace for the start/end time labels
GRID_STEP_KMH = 100
KMH_ROUND_UP = 50 # the trace's top edge is the lap's max speed rounded up to this


class TelemetryPanel:
    def __init__(self, rect, car_anim):
        self.rect = pygame.Rect(rect)
        self.car  = car_anim

        self._fonts = {}
        self._trace_layer = None
        self._plot = None # (x, y, w, h) of the plot area inside the trace layer
        self._v_top_kmh = 0.0

        self.rebuild()

    def _font(self, size: int) -> pygame.font.Font:
        if size not in self._fonts:
            self._fonts[size] = pygame.font.Font(None, size)
        return self._fonts[size]

    def _text(self, screen, text, size, color, x, y) -> None:
        screen.blit(self._font(size).render(text, True, color), (x, y))

    @property
    def _trace_origin(self) -> tuple:
        """Top-left of the trace layer on screen."""
        return self.rect.x + PADDING, self.rect.y + 410

    def rebuild(self) -> None:
        """Redraw the cached speed trace from the car's current lap."""
        width = self.rect.width - 2 * PADDING
        height = TOP_PAD + TRACE_HEIGHT + TIME_LABEL_H
        layer = pygame.Surface((width, height), pygame.SRCALPHA)

        plot_x, plot_y = AXIS_LABEL_W, TOP_PAD
        plot_w, plot_h = width - AXIS_LABEL_W, TRACE_HEIGHT
        self._plot = (plot_x, plot_y, plot_w, plot_h)

        # One speed sample per pixel column, spread evenly over lap time
        lap_time = self.car.lap_time
        times = np.linspace(0.0, lap_time, plot_w, endpoint=False)
        speeds = np.array([self.car.speed_at(t) for t in times])  # m/s
        speeds_kmh = speeds * 3.6

        top = math.ceil(speeds_kmh.max() / KMH_ROUND_UP) * KMH_ROUND_UP
        self._v_top_kmh = float(max(top, KMH_ROUND_UP))

        def y_of(kmh: float) -> float:
            return plot_y + plot_h - (kmh / self._v_top_kmh) * plot_h

        # Grid lines and km/h labels
        label_font = self._font(18)
        kmh = 0
        while kmh <= self._v_top_kmh:
            y = y_of(kmh)
            pygame.draw.line(layer, GRID_COLOR, (plot_x, y), (plot_x + plot_w, y), 1)
            label = label_font.render(str(kmh), True, MUTED_COLOR)
            layer.blit(label, (AXIS_LABEL_W - 6 - label.get_width(), y - label.get_height() / 2))
            kmh += GRID_STEP_KMH

        # The trace, coloured with the same map as the racing line on the track
        colors = colors_for_speeds(speeds)
        points = [(plot_x + i, y_of(s)) for i, s in enumerate(speeds_kmh)]
        for i in range(len(points) - 1):
            pygame.draw.line(layer, colors[i], points[i], points[i + 1], 2)

        # Start and end times under the trace
        start = label_font.render(format_lap_time(0.0), True, MUTED_COLOR)
        end = label_font.render(format_lap_time(lap_time), True, MUTED_COLOR)
        layer.blit(start, (plot_x, plot_y + plot_h + 4))
        layer.blit(end,   (plot_x + plot_w - end.get_width(), plot_y + plot_h + 4))

        self._trace_layer = layer

    def draw(self, screen: pygame.Surface) -> None:
        """Draw the whole panel. Call every frame, after the track and car."""
        car = self.car
        x   = self.rect.x + PADDING
        y   = self.rect.y

        pygame.draw.rect(screen, PANEL_COLOR, self.rect)
        pygame.draw.line(screen, BORDER_COLOR, self.rect.topleft, self.rect.bottomleft, 1)

        self._text(screen, "TELEMETRY", 22, MUTED_COLOR, x, y + PADDING)

        # Speed
        speed_kmh = car.speed_at(car.clock) * 3.6
        self._text(screen, f"{speed_kmh:.0f}", 110, TEXT_COLOR, x, y + 54)
        self._text(screen, "km/h", 30, MUTED_COLOR, x + 175, y + 92)

        # Lap clock
        self._text(screen, "LAP TIME", 20, MUTED_COLOR, x, y + 185)
        self._text(screen, format_lap_time(car.clock), 52, TEXT_COLOR, x, y + 206)
        self._text(screen, f"of {format_lap_time(car.lap_time)}", 24, MUTED_COLOR, x, y + 262)

        # Playback state and key hints (stand-in until the UI buttons exist)
        state = "playing" if car.playing else "paused"
        self._text(screen, f"{car.multiplier:g}x   {state}", 30, TEXT_COLOR, x, y + 310)
        self._text(screen, "Space: play/pause   R: restart", 20, MUTED_COLOR, x, y + 345)
        self._text(screen, "Up/Down: playback speed", 20, MUTED_COLOR, x, y + 366)

        # Speed trace with a cursor at the car's position in the lap
        self._text(screen, "SPEED TRACE", 20, MUTED_COLOR, x, y + 396)
        ox, oy = self._trace_origin
        screen.blit(self._trace_layer, (ox, oy))

        plot_x, plot_y, plot_w, plot_h = self._plot
        fraction = car.clock / car.lap_time
        cx = ox + plot_x + fraction * plot_w
        cy = oy + plot_y + plot_h - (speed_kmh / self._v_top_kmh) * plot_h
        pygame.draw.line(screen, CURSOR_COLOR, (cx, oy + plot_y), (cx, oy + plot_y + plot_h), 1)
        pygame.draw.circle(screen, CURSOR_COLOR, (cx, cy), 5)