# Tabla 4b. Benchmarking por año (2024 lag 1; 2025 lag 2)

| region   | task        | modelo                             |   anio_eval |   fuel_lag |      n |   n_pos |   roc_auc |   pr_auc |   precision |   recall |   f05_score |   falsos_positivos |
|:---------|:------------|:-----------------------------------|------------:|-----------:|-------:|--------:|----------:|---------:|------------:|---------:|------------:|-------------------:|
| Oriental | ignicion    | Incendios_PY (Ignición Híbrida M3) |        2024 |          1 | 100507 |    9137 |    0.7532 |   0.2316 |      0.2884 |   0.2038 |      0.2663 |               4595 |
| Oriental | ignicion    | Incendios_PY (Ignición Híbrida M3) |        2025 |          2 |  99429 |    9039 |    0.7539 |   0.2377 |      0.3189 |   0.1678 |      0.2702 |               3240 |
| Oriental | ignicion    | Incendios_PY (Ignición Base M2)    |        2024 |          1 | 100507 |    9137 |    0.7366 |   0.2095 |      0.2441 |   0.2085 |      0.236  |               5899 |
| Oriental | ignicion    | Incendios_PY (Ignición Base M2)    |        2025 |          2 |  99429 |    9039 |    0.7451 |   0.222  |      0.2836 |   0.1829 |      0.2554 |               4176 |
| Oriental | ignicion    | Canadian FWI completo              |        2024 |          1 | 100507 |    9137 |    0.5794 |   0.1181 |      0.1293 |   0.2637 |      0.144  |              16216 |
| Oriental | ignicion    | Canadian FWI completo              |        2025 |          2 |  99429 |    9039 |    0.5453 |   0.106  |      0.1126 |   0.1728 |      0.121  |              12310 |
| Oriental | ignicion    | INPE Risco de Fogo                 |        2024 |          1 | 100507 |    9137 |    0.5119 |   0.0949 |      0.0934 |   0.8507 |      0.1136 |              75477 |
| Oriental | ignicion    | INPE Risco de Fogo                 |        2025 |          2 |  99429 |    9039 |    0.4825 |   0.0854 |      0.0913 |   0.8002 |      0.111  |              71962 |
| Oriental | ignicion    | Línea Base Climatológica           |        2024 |          1 | 100507 |    9137 |    0.5    |   0.0909 |      0.0909 |   1      |      0.1111 |              91370 |
| Oriental | ignicion    | Línea Base Climatológica           |        2025 |          2 |  99429 |    9039 |    0.5    |   0.0909 |      0.0909 |   1      |      0.1111 |              90390 |
| Oriental | propagacion | Incendios_PY (Propagación Táctica) |        2024 |          1 | 514162 |   46742 |    0.7911 |   0.3872 |      0.6459 |   0.2125 |      0.4587 |               5444 |
| Oriental | propagacion | Incendios_PY (Propagación Táctica) |        2025 |          2 | 346797 |   31527 |    0.7785 |   0.3578 |      0.6583 |   0.1776 |      0.4271 |               2907 |
| Oriental | propagacion | Línea Base Climatológica           |        2024 |          1 | 514162 |   46742 |    0.5    |   0.0909 |      0.0909 |   1      |      0.1111 |             467420 |
| Oriental | propagacion | Línea Base Climatológica           |        2025 |          2 | 346797 |   31527 |    0.5    |   0.0909 |      0.0909 |   1      |      0.1111 |             315270 |
| Chaco    | ignicion    | Incendios_PY (Ignición Híbrida M3) |        2024 |          1 | 101244 |    9204 |    0.803  |   0.282  |      0.3265 |   0.3124 |      0.3236 |               5930 |
| Chaco    | ignicion    | Incendios_PY (Ignición Híbrida M3) |        2025 |          2 |  75372 |    6852 |    0.8011 |   0.2874 |      0.3306 |   0.2904 |      0.3217 |               4030 |
| Chaco    | ignicion    | Incendios_PY (Ignición Base M2)    |        2024 |          1 | 101244 |    9204 |    0.7993 |   0.2764 |      0.3227 |   0.2831 |      0.3139 |               5470 |
| Chaco    | ignicion    | Incendios_PY (Ignición Base M2)    |        2025 |          2 |  75372 |    6852 |    0.7991 |   0.278  |      0.3489 |   0.2583 |      0.326  |               3303 |
| Chaco    | ignicion    | Canadian FWI completo              |        2024 |          1 | 101244 |    9204 |    0.5069 |   0.0896 |      0.0921 |   0.9662 |      0.1124 |              87680 |
| Chaco    | ignicion    | Canadian FWI completo              |        2025 |          2 |  75372 |    6852 |    0.5046 |   0.0896 |      0.0932 |   0.9453 |      0.1137 |              63025 |
| Chaco    | ignicion    | INPE Risco de Fogo                 |        2024 |          1 | 101244 |    9204 |    0.5356 |   0.0979 |      0.0927 |   0.9271 |      0.113  |              83562 |
| Chaco    | ignicion    | INPE Risco de Fogo                 |        2025 |          2 |  75372 |    6852 |    0.5138 |   0.0942 |      0.0919 |   0.9215 |      0.1121 |              62356 |
| Chaco    | ignicion    | Línea Base Climatológica           |        2024 |          1 | 101244 |    9204 |    0.5    |   0.0909 |      0.0909 |   1      |      0.1111 |              92040 |
| Chaco    | ignicion    | Línea Base Climatológica           |        2025 |          2 |  75372 |    6852 |    0.5    |   0.0909 |      0.0909 |   1      |      0.1111 |              68520 |
| Chaco    | propagacion | Incendios_PY (Propagación Táctica) |        2024 |          1 | 412401 |   37491 |    0.9069 |   0.6937 |      0.8242 |   0.53   |      0.7418 |               4238 |
| Chaco    | propagacion | Incendios_PY (Propagación Táctica) |        2025 |          2 | 164054 |   14914 |    0.8982 |   0.6915 |      0.8719 |   0.5091 |      0.7631 |               1115 |
| Chaco    | propagacion | Línea Base Climatológica           |        2024 |          1 | 412401 |   37491 |    0.5    |   0.0909 |      0.0909 |   1      |      0.1111 |             374910 |
| Chaco    | propagacion | Línea Base Climatológica           |        2025 |          2 | 164054 |   14914 |    0.5    |   0.0909 |      0.0909 |   1      |      0.1111 |             149140 |
