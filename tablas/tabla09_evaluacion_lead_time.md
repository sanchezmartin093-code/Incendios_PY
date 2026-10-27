# Tabla 9. Evaluación con clima de t-1 (persistencia, 24 h de anticipación)

Las variables meteorológicas se sustituyen por las del día anterior. No es un pronóstico numérico real; es una cota de persistencia. Comparar con la Tabla 5 (clima del mismo día).

| region   | modelo          |      n |   n_pos |   roc_auc |   pr_auc |   brier |    bss |    ece |   cutoff_f05 |   precision |   recall |    f05 |    fp |    fn |
|:---------|:----------------|-------:|--------:|----------:|---------:|--------:|-------:|-------:|-------------:|------------:|---------:|-------:|------:|------:|
| Chaco    | M1_biofisico_t1 | 176616 |   16056 |    0.6356 |   0.1525 |  0.0807 | 0.023  | 0.0048 |      0.13151 |      0.1678 |   0.2149 | 0.1755 | 17115 | 12605 |
| Chaco    | M3_hibrido_t1   | 176616 |   16056 |    0.8013 |   0.2818 |  0.0734 | 0.112  | 0.0084 |      0.21265 |      0.3201 |   0.3252 | 0.3211 | 11090 | 10835 |
| Oriental | M1_biofisico_t1 | 199936 |   18176 |    0.68   |   0.1627 |  0.0799 | 0.033  | 0.0033 |      0.15309 |      0.1871 |   0.2733 | 0.1997 | 21580 | 13209 |
| Oriental | M3_hibrido_t1   | 199936 |   18176 |    0.7513 |   0.2315 |  0.0766 | 0.0732 | 0.007  |      0.21392 |      0.291  |   0.1977 | 0.2659 |  8757 | 14582 |
