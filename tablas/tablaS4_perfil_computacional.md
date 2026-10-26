# Tabla Suplementaria S4. Perfil de Rendimiento Computacional y Consumo de Memoria

| etapa                                            | motor                         |   ram_pico_mb |   latencia_arm64_s |   latencia_x86_s |
|:-------------------------------------------------|:------------------------------|--------------:|-------------------:|-----------------:|
| 1. Ingesta y cruce relacional (DuckDB Zero-Copy) | DuckDB C++                    |           450 |               0.05 |             0.07 |
| 2. Gating Biofísico de Etapa 1                   | Polars Vectorizado            |           220 |               0.08 |             0.11 |
| 3. Inferencia Dual LightGBM y Kernel t-1         | LightGBM C++ / OpenMP / SciPy |           380 |               0.2  |             0.26 |
| 4. Calibración y Exportación GeoTIFF/Parquet     | NumPy / Rasterio / Polars     |           140 |               0.1  |             0.14 |
| TOTAL PIPELINE NACIONAL (PUNTA A PUNTA)          | Consolidado O(1)              |           580 |               0.43 |             0.56 |
