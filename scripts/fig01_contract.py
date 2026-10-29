"""Factual contract for Figure 1.

The values below are the ones reported in the manuscript and its supplement.
They are kept in one place so the graphic cannot drift from the frozen text.
"""

FACTS = {
    "grid_cells": 400_006,
    "viirs_resolution_m": 375,
    "era5_resolution_deg": 0.25,
    "era5_nodes": 640,
    "gate_vpd_kpa": 0.8,
    "gate_rain_48h_mm": 10,
    "gate_retained_detections_pct": 97.17,
    "gate_removed_cell_days_pct": 21.7,
    "d2_radius_km": 25,
    "d2_window_days": 10,
    "m1_features": 24,
    "m2_features": 35,
    "m3_features_chaco": 44,
    "m3_features_oriental": 45,
    "train_years": "2018–2022",
    "validation_year": "2023",
    "test_years": "2024–2025",
    "spatial_block_km": 50,
    "spatial_buffer_km": 20,
    "auc_bootstrap": 500,
    "paired_bootstrap": 1_000,
}

FORBIDDEN_TEXT = (
    "39.2",
    "89.94",
    "hourly",
    "VDP",
    "no history lags",
    "2.28 s",
    "M3 sin historia",
)


def validate_contract() -> None:
    assert FACTS["grid_cells"] == 400_006
    assert FACTS["m3_features_chaco"] == 44
    assert FACTS["m3_features_oriental"] == 45
    assert FACTS["d2_radius_km"] == 25
    assert FACTS["d2_window_days"] == 10
    assert FACTS["spatial_block_km"] == 50
    assert FACTS["spatial_buffer_km"] == 20
