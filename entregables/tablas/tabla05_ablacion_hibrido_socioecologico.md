# Tabla 5. Estudio de Ablación: Modelo Biofísico Puro vs. Híbrido Socio-Ecológico y Benchmarks

| region   | arquitectura                           |   n_vars |   roc_auc | roc_ci_95       |   pr_auc |   brier_score |      bss |      ece |   precision_pct |   recall_pct |   f05_score |   falsos_positivos |
|:---------|:---------------------------------------|---------:|----------:|:----------------|---------:|--------------:|---------:|---------:|----------------:|-------------:|------------:|-------------------:|
| Chaco    | M3: multiscale socio-ecological hybrid |       44 |    0.8021 | [0.797 - 0.807] |   0.2842 |        0.0733 |   0.1131 |   0.0082 |           32.81 |        30.31 |      0.3228 |               9963 |
| Chaco    | M2: macro-anchor ignition (v1)         |       35 |    0.7991 | [0.795 - 0.805] |   0.2766 |        0.0737 |   0.1083 |   0.0104 |           33.27 |        27.26 |      0.3187 |               8778 |
| Chaco    | M1: biophysical only (no anthropic)    |       24 |    0.6421 | [0.636 - 0.649] |   0.1574 |        0.0806 |   0.025  |   0.0045 |           17.78 |        18.22 |      0.1787 |              13525 |
| Chaco    | Benchmark: full Canadian FWI           |        6 |    0.5059 | [0.498 - 0.513] |   0.0897 |      nan      | nan      | nan      |            9.25 |        95.73 |      0.113  |             150705 |
| Chaco    | Benchmark: INPE Fire Risk              |        7 |    0.5266 | [0.514 - 0.538] |   0.0964 |      nan      | nan      | nan      |            9.24 |        92.47 |      0.1126 |             145918 |
| Oriental | M3: multiscale socio-ecological hybrid |       45 |    0.7534 | [0.747 - 0.759] |   0.234  |        0.0765 |   0.0748 |   0.0082 |           30.13 |        18.59 |      0.268  |               7835 |
| Oriental | M2: macro-anchor ignition (v1)         |       35 |    0.7406 | [0.735 - 0.746] |   0.2146 |        0.0774 |   0.064  |   0.007  |           26.09 |        19.59 |      0.2447 |              10085 |
| Oriental | M1: biophysical only (no anthropic)    |       24 |    0.684  | [0.678 - 0.690] |   0.166  |        0.0798 |   0.0345 |   0.0052 |           19.17 |        24.63 |      0.2006 |              18876 |
| Oriental | Benchmark: full Canadian FWI           |        6 |    0.5611 | [0.552 - 0.571] |   0.1122 |      nan      | nan      | nan      |           12.22 |        21.85 |      0.134  |              28526 |
| Oriental | Benchmark: INPE Fire Risk              |        7 |    0.4973 | [0.486 - 0.507] |   0.0902 |      nan      | nan      | nan      |            9.24 |        82.56 |      0.1123 |             147439 |
