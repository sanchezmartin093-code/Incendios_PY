# Tabla 9. Evaluación con clima de t-1 (persistencia, 24 h de anticipación)

Las variables meteorológicas se sustituyen por las del día anterior. No es un pronóstico numérico real; es una cota de persistencia. Comparar con la Tabla 5 (clima del mismo día).

| region   | modelo          |      n |   n_pos |   roc_auc |   pr_auc |   brier |    bss |    ece |   cutoff_f05 |   precision |   recall |    f05 |    fp |    fn |
|:---------|:----------------|-------:|--------:|----------:|---------:|--------:|-------:|-------:|-------------:|------------:|---------:|-------:|------:|------:|
| Chaco    | M1_biofisico_t1 | 176616 |   16056 |    0.6361 |   0.1533 |  0.0807 | 0.0233 | 0.0045 |      0.12919 |      0.163  |   0.2287 | 0.1729 | 18854 | 12384 |
| Chaco    | M3_hibrido_t1   | 176616 |   16056 |    0.8013 |   0.2833 |  0.0734 | 0.1122 | 0.0084 |      0.21986 |      0.3264 |   0.3051 | 0.3219 | 10111 | 11157 |
| Oriental | M1_biofisico_t1 | 199936 |   18176 |    0.6795 |   0.1625 |  0.0799 | 0.0329 | 0.0038 |      0.15724 |      0.1911 |   0.2557 | 0.2012 | 19680 | 13528 |
| Oriental | M3_hibrido_t1   | 199936 |   18176 |    0.7516 |   0.2313 |  0.0766 | 0.0732 | 0.0072 |      0.21301 |      0.2918 |   0.1976 | 0.2664 |  8719 | 14584 |
