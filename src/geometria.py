"""
Módulo geometria.py: Generación determinística de la grilla nacional de 1 km² y manipulación espacial O(1).
Filosofía SOTA:
- Indexación espacial entera O(1) basada en coordenadas métricas UTM 21S (EPSG:32721).
- Representación columnar compatible con Apache Arrow / GeoParquet.
- Zero-overhead: Evita la computación innecesaria de polígonos vectoriales para operaciones tabulares masivas.
"""

import numpy as np

def coordenadas_a_cell_id(x_utm: float | np.ndarray, y_utm: float | np.ndarray, res_m: int = 1000) -> int | np.ndarray:
    """
    Transforma coordenadas proyectadas UTM en un identificador entero único y determinístico O(1).
    cell_id = (x // res_m) * 10_000_000 + (y // res_m)
    Permite joins y lookups espaciales instantáneos a nivel de CPU sin árboles R-Tree pesados.
    """
    col = np.int64(np.floor(x_utm / res_m))
    row = np.int64(np.floor(y_utm / res_m))
    return col * 10_000_000 + row

def cell_id_a_coordenadas_centro(cell_id: int | np.ndarray, res_m: int = 1000) -> tuple[float | np.ndarray, float | np.ndarray]:
    """
    Decodifica el cell_id para obtener el centroide exacto (X, Y) de la celda en UTM 21S en O(1).
    """
    col = cell_id // 10_000_000
    row = cell_id % 10_000_000
    centro_x = (col * res_m) + (res_m / 2.0)
    centro_y = (row * res_m) + (res_m / 2.0)
    return centro_x, centro_y

def generar_vecindad_3x3(cell_id: int, res_m: int = 1000) -> list[int]:
    """
    Calcula los 8 cell_ids vecinos inmediatos (ventana 3x3) de forma puramente aritmética en tiempo O(1).
    No requiere consultas geométricas GIS.
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
