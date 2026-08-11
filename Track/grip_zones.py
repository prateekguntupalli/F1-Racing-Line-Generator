import numpy as np
from Track.corner_detection import detect_corners

RACING_LINE = "racing_line"
CLEAN = "clean"
KERB = "kerb"

GRIP_TABLE = {
    "sunny": {
        RACING_LINE: 0.98,
        CLEAN:       0.96,
        KERB:        0.85,
    },
    "dry": {
        RACING_LINE: 1.00,
        CLEAN:       0.92,
        KERB:        0.80,
    },
    "damp": {
        RACING_LINE: 0.65,
        CLEAN:       0.75,
        KERB:        0.55,
    },
    "wet": {
        RACING_LINE: 0.45,
        CLEAN:       0.55,
        KERB:        0.35,
    },
    "heavy_rain": {
        RACING_LINE: 0.25,
        CLEAN:       0.35,
        KERB:        0.20,
    },
}

RACING_LINE_HALF_WIDTH = 1.25
KERB_WIDTH = 1.0

def assign_zones(path: np.ndarray, left_edge: np.ndarray, right_edge: np.ndarray) -> list:
    n = len(path)
    zones = []

    for i in range(n):
        half_width = np.linalg.norm(left_edge[i] - path[i])
        lateral_offset = 0.0

        if lateral_offset <= RACING_LINE_HALF_WIDTH:
            zones.append(RACING_LINE)
        elif lateral_offset >= (half_width - KERB_WIDTH):
            zones.append(KERB)
        else:
            zones.append(CLEAN)

    return zones


def get_grip_scale(zone: str, condition: str) -> float:
    if condition not in GRIP_TABLE:
        raise KeyError(f"Unknown weather condition: '{condition}'. "f"Valid options: {list(GRIP_TABLE.keys())}")
    if zone not in GRIP_TABLE[condition]:
        raise KeyError(f"Unknown grip zone: '{zone}'.")

    return GRIP_TABLE[condition][zone]


def build_grip_map(path: np.ndarray, left_edge: np.ndarray, right_edge: np.ndarray, condition: str) -> np.ndarray:
    zones = assign_zones(path, left_edge, right_edge)
    grip_scales = np.array([get_grip_scale(z, condition) for z in zones])
    return grip_scales