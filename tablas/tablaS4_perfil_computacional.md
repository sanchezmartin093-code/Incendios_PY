# Tabla Suplementaria S4. Perfil de Rendimiento Computacional y Consumo de Memoria

| etapa                                   | motor                |   ram_pico_mb |   latencia_arm64_s |   latencia_x86_s | nota                               |
|:----------------------------------------|:---------------------|--------------:|-------------------:|-----------------:|:-----------------------------------|
| 1. Ingesta, priors causales y gating    | Polars / cKDTree     |         nan   |              1.792 |              nan | medido en el equipo de ejecución   |
| 2. Inferencia dual LightGBM             | LightGBM C++ / NumPy |         nan   |              0.645 |              nan | medido en el equipo de ejecución   |
| 3. Agregación y boletín                 | Polars               |         nan   |              0.024 |              nan | medido en el equipo de ejecución   |
| TOTAL PIPELINE NACIONAL (PUNTA A PUNTA) | Consolidado          |        1975.6 |              2.461 |              nan | fecha=2024-09-09; x86_64 no medido |
