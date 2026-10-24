# Tabla 4. Benchmarking Internacional de Modelos de Peligro de Incendios sobre Test Ciego 2024

| region   | modelo                             |   roc_auc |   pr_auc |   brier_score |   brier_skill_score |   cutoff_f05 |   sensibilidad |   precision |   f05_score |   falsos_positivos |
|:---------|:-----------------------------------|----------:|---------:|--------------:|--------------------:|-------------:|---------------:|------------:|------------:|-------------------:|
| Oriental | Incendios_PY (Ignición Física)     |    0.7711 |   0.281  |        0.0765 |              0.074  |       0.1623 |         0.375  |      0.3409 |      0.3472 |                 29 |
| Oriental | Incendios_PY (Propagación Táctica) |    0.865  |   0.543  |        0.0586 |              0.291  |       0.7082 |         0.225  |      1      |      0.5921 |                  0 |
| Oriental | Proxy INPE Risco de Fogo           |    0.3995 |   0.0703 |        0.7631 |             -8.2341 |       0.9982 |         0.85   |      0.0969 |      0.1177 |                317 |
| Oriental | Canadian FWI (ISI)                 |    0.3892 |   0.0779 |        0.3085 |             -2.7331 |       0.2162 |         0.925  |      0.1025 |      0.1247 |                324 |
| Oriental | Línea Base Climatológica           |    0.5    |   0.0909 |        0.0826 |              0      |       0.0909 |         0      |      0      |      0      |                  0 |
| Chaco    | Incendios_PY (Ignición Física)     |    0.5952 |   0.129  |        0.0825 |              0.0013 |       0.0923 |         0.1928 |      0.2388 |      0.2279 |                102 |
| Chaco    | Incendios_PY (Propagación Táctica) |    0.941  |   0.8486 |        0.0276 |              0.6664 |       0.8046 |         0.7711 |      0.9771 |      0.9275 |                  3 |
| Chaco    | Proxy INPE Risco de Fogo           |    0.6457 |   0.1314 |        0.7496 |             -8.0707 |       1      |         0.7289 |      0.1389 |      0.1658 |                750 |
| Chaco    | Canadian FWI (ISI)                 |    0.5432 |   0.0967 |        0.1597 |             -0.9321 |       0.2827 |         0.7831 |      0.1125 |      0.1357 |               1026 |
| Chaco    | Línea Base Climatológica           |    0.5    |   0.0909 |        0.0826 |              0      |       0.0909 |         0      |      0      |      0      |                  0 |
