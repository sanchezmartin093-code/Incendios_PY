# Tabla 9. Evaluación con clima de t-1 (persistencia, 24 h de anticipación)

Las variables meteorológicas se sustituyen por las del día anterior. No es un pronóstico numérico real; es una cota de persistencia. Comparar con la Tabla 5 (clima del mismo día).

| region   | modelo          |      n |   n_pos |   roc_auc |   pr_auc |   brier |    bss |    ece |   cutoff_f05 |   precision |   recall |    f05 |    fp |    fn |
|:---------|:----------------|-------:|--------:|----------:|---------:|--------:|-------:|-------:|-------------:|------------:|---------:|-------:|------:|------:|
| Chaco    | M1_biofisico_t1 | 176616 |   16056 |    0.6221 |   0.1475 |  0.0811 | 0.0187 | 0.0105 |      0.13783 |      0.1454 |   0.217  | 0.1557 | 20472 | 12572 |
| Chaco    | M3_hibrido_t1   | 176616 |   16056 |    0.7834 |   0.2373 |  0.0754 | 0.0873 | 0.012  |      0.18886 |      0.2655 |   0.3386 | 0.2775 | 15042 | 10619 |
| Oriental | M1_biofisico_t1 | 199936 |   18176 |    0.6339 |   0.1412 |  0.0812 | 0.018  | 0.0027 |      0.13496 |      0.1555 |   0.2703 | 0.17   | 26676 | 13263 |
| Oriental | M3_hibrido_t1   | 199936 |   18176 |    0.7462 |   0.2287 |  0.0767 | 0.072  | 0.0057 |      0.205   |      0.2847 |   0.213  | 0.2667 |  9727 | 14305 |
