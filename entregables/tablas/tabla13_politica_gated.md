# Tabla 13. Política operativa gated M3/M2 en el test retrospectivo 2024–2025

M3 se usa en celdas con historia (`freq_hist_1km > 0`); M2 en celdas sin historia. El cutoff F0.5 se seleccionó en validación 2023 y se congeló para test.

|   roc_auc | roc_ci_95       |   pr_auc |     brier |       bss |        ece |   precision |   recall |      f05 |   tp |    fp |    fn |     tn | region   | policy      |   cutoff_f05 |      n |   n_pos |   n_fallback |
|----------:|:----------------|---------:|----------:|----------:|-----------:|------------:|---------:|---------:|-----:|------:|------:|-------:|:---------|:------------|-------------:|-------:|--------:|-------------:|
|  0.803509 | [0.799 - 0.809] | 0.284272 | 0.073268  | 0.113458  | 0.010649   |    0.327044 | 0.308732 | 0.32321  | 4957 | 10200 | 11099 | 150360 | Chaco    | M3/M2 gated |     0.221822 | 176616 |   16056 |       132562 |
|  0.750203 | [0.744 - 0.756] | 0.232734 | 0.0765615 | 0.0736058 | 0.00706368 |    0.301467 | 0.185464 | 0.267948 | 3371 |  7811 | 14805 | 173949 | Oriental | M3/M2 gated |     0.218606 | 199936 |   18176 |       116022 |

## Poblacional

| region   | modelo      |   prevalencia_poblacion |   prevalencia_muestra |   cutoff_f05_poblacional |   roc_auc_poblacional |   pr_auc_poblacional |   precision_poblacional_pct |   recall_poblacional_pct |   f05_poblacional |   brier_poblacional |   bss_poblacional |   ece_poblacional |   n_poblacion |   n_pos_poblacion |   n_muestra |
|:---------|:------------|------------------------:|----------------------:|-------------------------:|----------------------:|---------------------:|----------------------------:|-------------------------:|------------------:|--------------------:|------------------:|------------------:|--------------:|------------------:|------------:|
| Chaco    | M3/M2 gated |             0.000152948 |             0.0909091 |                 0.000436 |                0.8081 |               0.0009 |                       0.077 |                   30.873 |            0.001  |            0.000153 |            0.0003 |                 0 |     104976526 |             16056 |      176616 |
| Oriental | M3/M2 gated |             0.000414151 |             0.0909091 |                 0.001158 |                0.7598 |               0.0019 |                       0.21  |                   18.546 |            0.0026 |            0.000414 |            0.0005 |                 0 |      43887357 |             18176 |      199936 |
