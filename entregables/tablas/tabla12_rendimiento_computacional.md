# Tabla 12. Rendimiento computacional

| etapa                     | metodo          |      n |   tiempo_s |   ram_mb | nota                                |
|:--------------------------|:----------------|-------:|-----------:|---------:|:------------------------------------|
| Lookup cell_id O(1)       | NumPy int64     | 400006 |    0.00073 |    nan   | mediana 100 repeticiones            |
| Point-in-Polygon          | GeoPandas sjoin |  50000 |    0.0261  |    nan   | 50k puntos; extrapolación no lineal |
| Ingesta + priors + gating | Polars/cKDTree  | 400006 |    1.268   |    nan   | 2024-09-09                          |
| Inferencia dual           | LightGBM        | 400006 |    0.285   |    nan   | 2024-09-09                          |
| Agregación/boletín        | Polars          | 400006 |    0.016   |    nan   | 2024-09-09                          |
| Total pipeline            | Consolidado     | 400006 |    1.569   |   2182.1 | ARM64; x86_64 no medido             |
