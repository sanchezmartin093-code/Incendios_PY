# Tabla Suplementaria S19. Baseline acoplado t-1 vs arquitectura desacoplada

Mismo conjunto de test combinado (ignición + propagación, 1:10 por tarea). El baseline acoplado se entrena con un único target de fuego e incorpora las tres variables de fuego t-1; la arquitectura desacoplada usa M3 en filas de ignición y el modelo de propagación en filas de propagación. No es un modelo operativo.

| region   | comparator            |   roc_auc |   pr_auc |   brier |    bss |    ece |       n |   n_pos |
|:---------|:----------------------|----------:|---------:|--------:|-------:|-------:|--------:|--------:|
| Chaco    | coupled_t1            |    0.8862 |   0.6223 |  0.0513 | 0.3788 | 0.0091 |  753071 |   68461 |
| Chaco    | decoupled_task_models |    0.8868 |   0.6262 |  0.051  | 0.3829 | 0.0084 |  753071 |   68461 |
| Oriental | coupled_t1            |    0.7871 |   0.3598 |  0.0701 | 0.1512 | 0.0089 | 1060895 |   96445 |
| Oriental | decoupled_task_models |    0.7806 |   0.3532 |  0.0705 | 0.1467 | 0.009  | 1060895 |   96445 |
