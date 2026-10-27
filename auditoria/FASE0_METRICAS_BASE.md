# FASE 0 - METRICAS BASE (PROVISIONALES)

Generado: 2026-10-27T20:06:19.053537+00:00

Estas metricas se copian de las tablas existentes y NO se regeneran en Fase 0. Son la linea base pre-auditoria; no son resultados finales. El test 2024-2025 ya fue observado y no es un holdout ciego.

## Tabla 2 - Test retrospectivo 2024-2025 (caso-control)

Fuente: `tablas/tabla02_metricas_desempeno_modelos.csv`

| region   | capa_modelo         | umbral_estrategia          |   cutoff |   roc_auc |   pr_auc |   precision |   sensibilidad_recall |   f05_score |   falsos_positivos |
|:---------|:--------------------|:---------------------------|---------:|----------:|---------:|------------:|----------------------:|------------:|-------------------:|
| Chaco    | Ignición Primaria   | Estándar (0.50)            |    0.5   |    0.7945 |   0.2677 |      0.4873 |                0.0131 |      0.0593 |                222 |
| Chaco    | Ignición Primaria   | Youden J (0.08)            |    0.081 |    0.7945 |   0.2677 |      0.1798 |                0.8281 |      0.2132 |              60642 |
| Chaco    | Ignición Primaria   | Alta Precisión F0.5 (0.20) |    0.2   |    0.7945 |   0.2677 |      0.3218 |                0.2758 |      0.3114 |               9332 |
| Chaco    | Propagación Táctica | Estándar (0.50)            |    0.5   |    0.9032 |   0.691  |      0.8208 |                0.5375 |      0.7425 |               6151 |
| Chaco    | Propagación Táctica | Youden J (0.08)            |    0.076 |    0.9032 |   0.691  |      0.4201 |                0.7466 |      0.4603 |              54013 |
| Chaco    | Propagación Táctica | Alta Precisión F0.5 (0.58) |    0.578 |    0.9032 |   0.691  |      0.8511 |                0.5041 |      0.7481 |               4621 |
| Chaco    | Ignición Híbrida M3 | Estándar (0.50)            |    0.5   |    0.8021 |   0.2843 |      0.5326 |                0.0097 |      0.045  |                136 |
| Chaco    | Ignición Híbrida M3 | Youden J (0.09)            |    0.087 |    0.8021 |   0.2843 |      0.1876 |                0.7939 |      0.2214 |              55217 |
| Chaco    | Ignición Híbrida M3 | Alta Precisión F0.5 (0.23) |    0.231 |    0.8021 |   0.2843 |      0.3397 |                0.2802 |      0.3258 |               8746 |
| Oriental | Ignición Primaria   | Estándar (0.50)            |    0.5   |    0.7369 |   0.2102 |      0      |                0      |      0      |                  0 |
| Oriental | Ignición Primaria   | Youden J (0.09)            |    0.089 |    0.7369 |   0.2102 |      0.163  |                0.7148 |      0.1928 |              66702 |
| Oriental | Ignición Primaria   | Alta Precisión F0.5 (0.21) |    0.21  |    0.7369 |   0.2102 |      0.2695 |                0.1675 |      0.2403 |               8253 |
| Oriental | Propagación Táctica | Estándar (0.50)            |    0.5   |    0.7842 |   0.371  |      0.6764 |                0.1753 |      0.4304 |               6565 |
| Oriental | Propagación Táctica | Youden J (0.09)            |    0.088 |    0.7842 |   0.371  |      0.2231 |                0.6089 |      0.2554 |             165997 |
| Oriental | Propagación Táctica | Alta Precisión F0.5 (0.43) |    0.433 |    0.7842 |   0.371  |      0.6325 |                0.2038 |      0.4452 |               9269 |
| Oriental | Ignición Híbrida M3 | Estándar (0.50)            |    0.5   |    0.7533 |   0.2338 |      0.583  |                0.0147 |      0.0667 |                191 |
| Oriental | Ignición Híbrida M3 | Youden J (0.10)            |    0.104 |    0.7533 |   0.2338 |      0.1886 |                0.6156 |      0.219  |              48138 |
| Oriental | Ignición Híbrida M3 | Alta Precisión F0.5 (0.21) |    0.207 |    0.7533 |   0.2338 |      0.2915 |                0.2079 |      0.2698 |               9182 |

## Tabla 4 - Benchmarking internacional

Fuente: `tablas/tabla04_benchmarking_internacional.csv`

| region   | task        | modelo                             |   roc_auc |   pr_auc |   cutoff_f05 |   precision |   sensibilidad |   f05_score |
|:---------|:------------|:-----------------------------------|----------:|---------:|-------------:|------------:|---------------:|------------:|
| Oriental | ignicion    | Incendios_PY (Ignición Híbrida M3) |    0.7533 |   0.2338 |       0.2072 |      0.2915 |         0.2079 |      0.2698 |
| Oriental | ignicion    | Incendios_PY (Ignición Base M2)    |    0.7369 |   0.2102 |       0.21   |      0.2695 |         0.1675 |      0.2403 |
| Oriental | ignicion    | Canadian FWI completo              |    0.5564 |   0.1061 |       2.1509 |      0.1076 |         0.3899 |      0.1258 |
| Oriental | ignicion    | INPE Risco de Fogo                 |    0.499  |   0.0908 |       0.075  |      0.0924 |         0.7947 |      0.1123 |
| Oriental | ignicion    | Línea Base Climatológica           |    0.5    |   0.0909 |       0.0909 |      0      |         0      |      0      |
| Oriental | propagacion | Incendios_PY (Propagación Táctica) |    0.7842 |   0.371  |       0.433  |      0.6325 |         0.2038 |      0.4452 |
| Oriental | propagacion | Línea Base Climatológica           |    0.5    |   0.0909 |       0.0909 |      0      |         0      |      0      |
| Chaco    | ignicion    | Incendios_PY (Ignición Híbrida M3) |    0.8021 |   0.2843 |       0.2305 |      0.3397 |         0.2802 |      0.3258 |
| Chaco    | ignicion    | Incendios_PY (Ignición Base M2)    |    0.7945 |   0.2677 |       0.2003 |      0.3218 |         0.2758 |      0.3114 |
| Chaco    | ignicion    | Canadian FWI completo              |    0.5173 |   0.0957 |       0.3029 |      0.0928 |         0.9608 |      0.1133 |
| Chaco    | ignicion    | INPE Risco de Fogo                 |    0.5389 |   0.1031 |       0.8963 |      0.1172 |         0.1257 |      0.1188 |
| Chaco    | ignicion    | Línea Base Climatológica           |    0.5    |   0.0909 |       0.0909 |      0      |         0      |      0      |
| Chaco    | propagacion | Incendios_PY (Propagación Táctica) |    0.9032 |   0.691  |       0.5784 |      0.8511 |         0.5041 |      0.7481 |
| Chaco    | propagacion | Línea Base Climatológica           |    0.5    |   0.0909 |       0.0909 |      0      |         0      |      0      |

## Tabla 5 - Ablacion controlada

Fuente: `tablas/tabla05_ablacion_hibrido_socioecologico.csv`

| region   | arquitectura                           |   n_vars |   roc_auc |   pr_auc |   precision_pct |   recall_pct |   f05_score |
|:---------|:---------------------------------------|---------:|----------:|---------:|----------------:|-------------:|------------:|
| Chaco    | M3: multiscale socio-ecological hybrid |       44 |    0.8021 |   0.2843 |           33.97 |        28.01 |      0.3259 |
| Chaco    | M2: macro-anchor ignition (v1)         |       35 |    0.7988 |   0.2757 |           33.34 |        26.79 |      0.3178 |
| Chaco    | M1: biophysical only (no anthropic)    |       24 |    0.6425 |   0.158  |           17.06 |        21.89 |      0.1785 |
| Chaco    | Benchmark: full Canadian FWI           |        6 |    0.5173 |   0.0957 |            9.28 |        96.08 |      0.1133 |
| Chaco    | Benchmark: INPE Fire Risk              |        7 |    0.5389 |   0.1031 |           11.72 |        12.57 |      0.1188 |
| Oriental | M3: multiscale socio-ecological hybrid |       45 |    0.7533 |   0.2338 |           29.15 |        20.79 |      0.2698 |
| Oriental | M2: macro-anchor ignition (v1)         |       35 |    0.7408 |   0.2153 |           26.05 |        20.69 |      0.2477 |
| Oriental | M1: biophysical only (no anthropic)    |       24 |    0.6844 |   0.168  |           19.63 |        24.15 |      0.2039 |
| Oriental | Benchmark: full Canadian FWI           |        6 |    0.5564 |   0.1061 |           10.76 |        38.99 |      0.1258 |
| Oriental | Benchmark: INPE Fire Risk              |        7 |    0.499  |   0.0908 |            9.24 |        79.47 |      0.1123 |

## Tabla 7 - Evaluacion poblacional

Fuente: `tablas/tabla07_evaluacion_poblacional.csv`

| region   | task        | modelo                 |   roc_auc_poblacional |   pr_auc_poblacional |   precision_poblacional_pct |   recall_poblacional_pct |   brier_poblacional |   bss_poblacional |
|:---------|:------------|:-----------------------|----------------------:|---------------------:|----------------------------:|-------------------------:|--------------------:|------------------:|
| Chaco    | ignicion    | M3_ignicion_hibrida    |                0.806  |               0.0011 |                       0.086 |                   24.931 |            0.000153 |            0.0003 |
| Chaco    | ignicion    | M2_ignicion_base       |                0.8021 |               0.0007 |                       0.086 |                   25.704 |            0.000153 |            0.0003 |
| Chaco    | propagacion | M2_propagacion_tactica |                0.9089 |               0.1218 |                      10.241 |                   48.768 |            0.00145  |            0.0649 |
| Oriental | ignicion    | M3_ignicion_hibrida    |                0.7632 |               0.0019 |                       0.191 |                   23.146 |            0.000414 |            0.0006 |
| Oriental | ignicion    | M2_ignicion_base       |                0.7479 |               0.0014 |                       0.167 |                   18.833 |            0.000414 |            0.0004 |
| Oriental | propagacion | M2_propagacion_tactica |                0.7938 |               0.0277 |                       4.538 |                   19.217 |            0.002071 |            0.0126 |

## Tabla 8 - Validacion espacial por bloques

Fuente: `tablas/tabla08_validacion_espacial_bloques.csv`

| region   | modelo       | evaluacion        |      n |   n_pos |   roc_auc |   pr_auc |   precision |   recall |    f05 |
|:---------|:-------------|:------------------|-------:|--------:|----------:|---------:|------------:|---------:|-------:|
| Chaco    | M1_biofisico | holdout_espacial  |  34367 |    3390 |    0.6544 |   0.1703 |      0.164  |   0.3183 | 0.1816 |
| Chaco    | M1_biofisico | heldin_referencia | 142249 |   12666 |    0.637  |   0.1529 |      0.155  |   0.2897 | 0.1708 |
| Chaco    | M2_base      | holdout_espacial  |  34367 |    3390 |    0.6896 |   0.1816 |      0.252  |   0.0094 | 0.041  |
| Chaco    | M2_base      | heldin_referencia | 142249 |   12666 |    0.8087 |   0.2909 |      0.3686 |   0.2478 | 0.3359 |
| Chaco    | M3_hibrido   | holdout_espacial  |  34367 |    3390 |    0.6725 |   0.1685 |      0.3043 |   0.0062 | 0.0286 |
| Chaco    | M3_hibrido   | heldin_referencia | 142249 |   12666 |    0.8078 |   0.2902 |      0.3515 |   0.2534 | 0.3263 |
| Oriental | M1_biofisico | holdout_espacial  |  58756 |    4389 |    0.6875 |   0.1365 |      0.1687 |   0.1948 | 0.1734 |
| Oriental | M1_biofisico | heldin_referencia | 141180 |   13787 |    0.6743 |   0.1754 |      0.2117 |   0.2018 | 0.2097 |
| Oriental | M2_base      | holdout_espacial  |  58756 |    4389 |    0.712  |   0.1582 |      0.2381 |   0.0011 | 0.0056 |
| Oriental | M2_base      | heldin_referencia | 141180 |   13787 |    0.7323 |   0.2224 |      0.2709 |   0.1963 | 0.2518 |
| Oriental | M3_hibrido   | holdout_espacial  |  58756 |    4389 |    0.701  |   0.1514 |      0.3824 |   0.003  | 0.0144 |
| Oriental | M3_hibrido   | heldin_referencia | 141180 |   13787 |    0.744  |   0.2396 |      0.3116 |   0.1723 | 0.2683 |

## Tabla 9 - Lead time t-1 (persistencia)

Fuente: `tablas/tabla09_evaluacion_lead_time.csv`

| region   | modelo          |      n |   n_pos |   roc_auc |   pr_auc |   cutoff_f05 |   precision |   recall |    f05 |
|:---------|:----------------|-------:|--------:|----------:|---------:|-------------:|------------:|---------:|-------:|
| Chaco    | M1_biofisico_t1 | 176616 |   16056 |    0.6358 |   0.1529 |      0.13161 |      0.1656 |   0.2117 | 0.1732 |
| Chaco    | M3_hibrido_t1   | 176616 |   16056 |    0.8014 |   0.2831 |      0.22926 |      0.3342 |   0.279  | 0.3215 |
| Oriental | M1_biofisico_t1 | 199936 |   18176 |    0.6794 |   0.1631 |      0.15272 |      0.1875 |   0.2731 | 0.2    |
| Oriental | M3_hibrido_t1   | 199936 |   18176 |    0.7516 |   0.2315 |      0.21181 |      0.2897 |   0.1991 | 0.2655 |

## Tabla 10 - Significancia pareada

Fuente: `tablas/tabla10_significancia_pareada.csv`

| region   | task     | comparacion                                                 |   delta_roc |   p_value_bootstrap |   n_bootstrap |
|:---------|:---------|:------------------------------------------------------------|------------:|--------------------:|--------------:|
| Oriental | ignicion | Incendios_PY (Ignición Híbrida M3) vs Canadian FWI completo |      0.1969 |               0.001 |          1000 |
| Oriental | ignicion | Incendios_PY (Ignición Híbrida M3) vs INPE Risco de Fogo    |      0.2543 |               0.001 |          1000 |
| Chaco    | ignicion | Incendios_PY (Ignición Híbrida M3) vs Canadian FWI completo |      0.2848 |               0.001 |          1000 |
| Chaco    | ignicion | Incendios_PY (Ignición Híbrida M3) vs INPE Risco de Fogo    |      0.2632 |               0.001 |          1000 |

## Nota de congelamiento

La tabla 2 y la tabla 5 no usan exactamente la misma capacidad de entrenamiento para M2/M3; esa inconsistencia queda registrada como C7. La tabla 8 actual contradice la conclusion espacial del manuscrito actual; se registra como C4. La tabla 9 contiene fuga de variables antecedentes del mismo dia; se registra como C2. La tabla 11 y las tablas S3-S5 no se regeneraron despues del fuel anual; se registran como H10.

