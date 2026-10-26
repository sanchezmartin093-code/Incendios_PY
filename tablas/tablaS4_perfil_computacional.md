# Tabla Suplementaria S4. Perfil de Rendimiento Computacional y Consumo de Memoria

| etapa                                   | motor                |   ram_pico_mb |   latencia_arm64_s |   latencia_x86_s | nota                               |
|:----------------------------------------|:---------------------|--------------:|-------------------:|-----------------:|:-----------------------------------|
| 1. Ingesta, priors causales y gating    | Polars / cKDTree     |         nan   |              1.129 |              nan | medido en el equipo de ejecución   |
| 2. Inferencia dual LightGBM             | LightGBM C++ / NumPy |         nan   |              0.266 |              nan | medido en el equipo de ejecución   |
| 3. Agregación y boletín                 | Polars               |         nan   |              0.018 |              nan | medido en el equipo de ejecución   |
| TOTAL PIPELINE NACIONAL (PUNTA A PUNTA) | Consolidado          |        2185.5 |              1.413 |              nan | fecha=2024-09-09; x86_64 no medido |
