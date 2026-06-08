from __future__ import annotations

DRY        = "dry"
SUNNY      = "sunny"
DAMP       = "damp"
WET        = "wet"
HEAVY_RAIN = "heavy_rain"

ALL_CONDITIONS: tuple[str, ...] = (DRY, SUNNY, DAMP, WET, HEAVY_RAIN)

BASE_GRIP: dict[str, float] = {
    DRY:        1.00,
    SUNNY:      0.95,
    DAMP:       0.72,
    WET:        0.52,
    HEAVY_RAIN: 0.32,
}

# Display labels for the UI
DISPLAY_LABELS: dict[str, str] = {
    DRY:        "Dry (Optimal)",
    SUNNY:      "Sunny",
    DAMP:       "Damp",
    WET:        "Wet",
    HEAVY_RAIN: "Heavy Rain",
}


def validate(condition: str) -> None:
    if condition not in ALL_CONDITIONS:
        raise ValueError(
            f"Unknown weather condition '{condition}'. "
            f"Valid options: {list(ALL_CONDITIONS)}"
        )