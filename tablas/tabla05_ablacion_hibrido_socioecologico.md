# Tabla 5. Estudio de Ablación: Modelo Biofísico Puro vs. Híbrido Socio-Ecológico y Benchmarks

| region   | arquitectura                            |   n_vars |   roc_auc | roc_ci_95       |   pr_auc |   brier_score |     bss |    ece |   precision_pct |   recall_pct |   f05_score |   falsos_positivos |
|:---------|:----------------------------------------|---------:|----------:|:----------------|---------:|--------------:|--------:|-------:|----------------:|-------------:|------------:|-------------------:|
| Chaco    | M3: Híbrido Socio-Ecológico Multiescala |       34 |    0.7259 | [0.724 - 0.728] |   0.2766 |        0.0791 |  0.0425 | 0.0528 |           43.01 |        17.71 |      0.3345 |              16339 |
| Chaco    | M2: Ignición con Ancla Macro (v1)       |       25 |    0.7219 | [0.720 - 0.724] |   0.2534 |        0.0815 |  0.0133 | 0.0505 |           34.16 |        19.46 |      0.2968 |              26117 |
| Chaco    | M1: Biofísico Puro (Sin Antrópico)      |       18 |    0.6116 | [0.609 - 0.614] |   0.1325 |        0.0817 |  0.0112 | 0.0106 |           16.13 |        26.57 |      0.1751 |              96234 |
| Chaco    | Benchmark: Canadian FWI (ISI)           |        4 |    0.5183 | [0.516 - 0.521] |   0.0928 |        0.109  | -0.3193 | 0.1168 |            9.54 |        72.48 |      0.1155 |             478612 |
| Chaco    | Benchmark: INPE Risco de Fogo           |        3 |    0.5036 | [0.501 - 0.506] |   0.0924 |        0.3667 | -3.4375 | 0.4429 |            9.09 |       100    |      0.1111 |             696459 |
| Oriental | M3: Híbrido Socio-Ecológico Multiescala |       35 |    0.8168 | [0.816 - 0.818] |   0.364  |        0.0702 |  0.151  | 0.0197 |           44.9  |        29    |      0.4046 |              34790 |
| Oriental | M2: Ignición con Ancla Macro (v1)       |       25 |    0.7978 | [0.796 - 0.799] |   0.3085 |        0.0731 |  0.1161 | 0.0156 |           34.82 |        31.2  |      0.3403 |              57098 |
| Oriental | M1: Biofísico Puro (Sin Antrópico)      |       18 |    0.6132 | [0.611 - 0.615] |   0.1498 |        0.0828 | -0.0014 | 0.0176 |           17.59 |        18.75 |      0.1781 |              85849 |
| Oriental | Benchmark: Canadian FWI (ISI)           |        4 |    0.6055 | [0.604 - 0.607] |   0.1411 |        0.0883 | -0.0675 | 0.0602 |           17.83 |        16.7  |      0.1759 |              75201 |
| Oriental | Benchmark: INPE Risco de Fogo           |        3 |    0.5865 | [0.585 - 0.588] |   0.1404 |        0.2425 | -1.9328 | 0.3184 |           22.41 |        12.27 |      0.1923 |              41521 |
