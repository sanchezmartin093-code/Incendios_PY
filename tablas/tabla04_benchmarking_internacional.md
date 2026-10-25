# Tabla 4. Benchmarking Internacional de Modelos de Peligro de Incendios sobre Test Ciego 2024

| region   | modelo                             |   roc_auc |   pr_auc |   brier_score |   brier_skill_score |   cutoff_f05 |   sensibilidad |   precision |   f05_score |   falsos_positivos |
|:---------|:-----------------------------------|----------:|---------:|--------------:|--------------------:|-------------:|---------------:|------------:|------------:|-------------------:|
| Oriental | Incendios_PY (Ignición Física)     |    0.7106 |   0.2741 |        0.0783 |              0.0521 |       0.0948 |         0.475  |      0.3167 |      0.3393 |                 41 |
| Oriental | Incendios_PY (Propagación Táctica) |    0.6917 |   0.3031 |        0.078  |              0.056  |       0.106  |         0.25   |      0.5    |      0.4167 |                 10 |
| Oriental | Proxy INPE Risco de Fogo           |    0.3995 |   0.0703 |        0.7631 |             -8.2341 |       0.9982 |         0.85   |      0.0969 |      0.1177 |                317 |
| Oriental | Canadian FWI (ISI)                 |    0.3892 |   0.0779 |        0.3085 |             -2.7331 |       0.2162 |         0.925  |      0.1025 |      0.1247 |                324 |
| Oriental | Línea Base Climatológica           |    0.5    |   0.0909 |        0.0826 |              0      |       0.0909 |         0      |      0      |      0      |                  0 |
| Chaco    | Incendios_PY (Ignición Física)     |    0.8305 |   0.5714 |        0.0517 |              0.3746 |       0.3728 |         0.5723 |      0.6051 |      0.5982 |                 62 |
| Chaco    | Incendios_PY (Propagación Táctica) |    0.8494 |   0.7057 |        0.0507 |              0.386  |       0.2432 |         0.6627 |      0.9016 |      0.841  |                 12 |
| Chaco    | Proxy INPE Risco de Fogo           |    0.6457 |   0.1314 |        0.7496 |             -8.0707 |       1      |         0.7289 |      0.1389 |      0.1658 |                750 |
| Chaco    | Canadian FWI (ISI)                 |    0.5432 |   0.0967 |        0.1597 |             -0.9321 |       0.2827 |         0.7831 |      0.1125 |      0.1357 |               1026 |
| Chaco    | Línea Base Climatológica           |    0.5    |   0.0909 |        0.0826 |              0      |       0.0909 |         0      |      0      |      0      |                  0 |
