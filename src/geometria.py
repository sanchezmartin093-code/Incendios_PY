"""
geometria.py module: Deterministic generation of the national 1 km² grid and O(1) spatial manipulation.
SOTA philosophy:
- Integer O(1) spatial indexing based on UTM 21S metric coordinates (EPSG:32721).
- Columnar representation compatible with Apache Arrow / GeoParquet.
- Zero-overhead: Avoids unnecessary computation of vector polygons for massive tabular operations.
"""

from __future__ import annotations

import numpy as np

def coordenadas_a_cell_id(x_utm: float | np.ndarray, y_utm: float | np.ndarray, res_m: int = 1000) -> int | np.ndarray:
    """
    Transforms projected UTM coordinates into a unique, deterministic integer identifier in O(1).
    cell_id = (x // res_m) * 10_000_000 + (y // res_m)
    Enables instant spatial joins and lookups at the CPU level without heavy R-Tree structures.
    """
    col = np.int64(np.floor(x_utm / res_m))
    row = np.int64(np.floor(y_utm / res_m))
    return col * 10_000_000 + row

def cell_id_a_coordenadas_centro(cell_id: int | np.ndarray, res_m: int = 1000) -> tuple[float | np.ndarray, float | np.ndarray]:
    """
    Decodes the cell_id to obtain the exact centroid (X, Y) of the cell in UTM 21S in O(1).
    """
    col = cell_id // 10_000_000
    row = cell_id % 10_000_000
    centro_x = (col * res_m) + (res_m / 2.0)
    centro_y = (row * res_m) + (res_m / 2.0)
    return centro_x, centro_y

def generar_vecindad_3x3(cell_id: int, res_m: int = 1000) -> list[int]:
    """
    Computes the 8 immediate neighboring cell_ids (3x3 window) purely arithmetically in O(1) time.
    Does not require GIS geometric queries.
    """
    col = cell_id // 10_000_000
    row = cell_id % 10_000_000
    vecinos = []
    for d_col in [-1, 0, 1]:
        for d_row in [-1, 0, 1]:
            if d_col == 0 and d_row == 0:
                continue
            v_col = col + d_col
            v_row = row + d_row
            vecinos.append(int(v_col * 10_000_000 + v_row))
    return vecinos
