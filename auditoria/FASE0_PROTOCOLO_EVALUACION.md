# Fase 0 - Protocolo de evaluacion congelado

**Version:** 1.0  
**Fecha:** 2026-10-27  
**Commit base:** `94d5b5e`  
**Estado:** congelado antes de mirar el holdout nuevo.

## 1. Principio

Ninguna decision cientifica se toma usando el holdout de confirmacion. El test 2024-2025 es retrospectivo y se usa solo para diagnostico y comparacion con la literatura. El holdout nuevo se usa una sola vez en Fase 5.

## 2. Conjuntos

| Conjunto | Periodo / espacio | Uso |
|---|---|---|
| Train | 2018-2022 | Entrenamiento |
| Validation | 2023 | Calibracion, umbrales y seleccion |
| Test retrospectivo | 2024-2025 | Diagnostico; no seleccion |
| Holdout nuevo | Bloques espaciales reservados | Confirmacion unica |
| 2026 futuro | Si hay cobertura completa | Validacion prospectiva secundaria |

## 3. Definicion de target

Se mantiene D2:

- Ignicion primaria: fuego en `t`, sin fuego en la misma celda en los 10 dias previos y sin fuego `t-1` a menos de 25 km.
- Propagacion: fuego en `t` dentro de 25 km de un fuego `t-1`.
- No se reemplaza D2 por el target “fuego activo” de PoF.

## 4. Modelos

- **M3:** modelo hibrido primario.
- **M2:** comparador directo.
- **M1:** baseline biofisico y fallback operativo.
- **Propagacion:** capa tactica separada.

Regla de capacidad:

- M1, M2 y M3 se comparan con la misma capacidad de entrenamiento.
- Si se reporta una comparacion con capacidades distintas, debe etiquetarse como tal y no usarse para atribuir la diferencia a las features.
- M3 y M2 finales deben salir del mismo esquema de entrenamiento y estar guardados de forma reproducible.

## 5. Features

- La lista de features se congela al inicio de Fase 1.
- Las variables nuevas de Fase 4 entran una por una, con ablacion preespecificada.
- No se agregan variables despues de mirar el holdout nuevo.
- No se usa informacion futura: fuel `Y-1` o anterior, Hansen `Y-1/Y-2`, clima hasta `t` para nowcast y hasta `t-1` para lead time.
- Las features antecedentes de 60 dias deben calcularse con ventanas que terminen en `t-1` cuando se evalue lead time.

## 6. Metricas primarias

- ROC-AUC.
- PR-AUC.
- Brier score.
- Brier Skill Score.
- ECE y curvas de fiabilidad.
- Precision, recall y F0.5 en el umbral de validacion.
- Metricas poblacionales con pesos de muestreo.
- Metricas espaciales en bloques.
- Lead time `t-1`, `t-2` y `t-3`.
- Cost-loss por region y estacion.
- Analisis de extremos: recall@K y precision@K para top 1 % y top 100.

No se reporta solo ROC-AUC.

## 7. Umbrales

- Se seleccionan exclusivamente en validacion.
- Se usa F0.5 para el corte de alta precision y cost-loss para utilidad operativa.
- No se calculan umbrales en test.
- Si falta el JSON de umbrales, el script debe fallar, no recalcular en test.
- La alerta debe fallar si falta el fuel anual o el umbral; no debe caer silenciosamente a estatico.

## 8. Calibracion y prevalencia

- Calibracion Platt monotona en validacion.
- Correccion de prior caso-control a poblacion.
- Pesos de muestreo documentados.
- Prevalencia por ano/target/region, no solo global.
- Se reportan metricas caso-control y poblacionales por separado.

## 9. Incertidumbre y comparaciones

- Bootstrap agrupado por bloques espaciales o eventos, no por filas independientes.
- 1.000 replicas para intervalos y diferencias pareadas.
- Comparaciones M3 vs M2, M3 vs M1 y M3 vs FWI/INPE.
- Multiplicidad: declarar analisis confirmatorios vs exploratorios.
- No se interpretan p-valores como prueba causal.

## 10. Validacion espacial

- El holdout nuevo es independiente del usado en `20`.
- Buffer de 20 km.
- Priors recalculados sin fuegos del holdout.
- Reportar held-in como referencia, no como confirmacion.
- No afirmar transferibilidad espacial sin este analisis.

## 11. Analisis de sensibilidad

Se reportan, sin usarlos para seleccionar el modelo final:

- fuel C2 con lag 1/2/3;
- C3 vs C2 si C3 se incorpora;
- PEV vs Hargreaves;
- VIIRS vs area quemada independiente;
- con/sin Hansen;
- con/sin humedad de suelo;
- capacidad fija vs capacidad alternativa.

## 12. Regla de seleccion

1. La seleccion se hace con validacion 2023 y validacion espacial de desarrollo.
2. No se usa 2024-2025 ni el holdout nuevo.
3. Se preespecifica un modelo primario y un comparador.
4. Si M3 no supera a M2 con la misma capacidad, se reporta el resultado negativo y se reconsidera el fallback.
5. No se cambia el modelo despues de la confirmacion.

## 13. Criterios de parada

- Si no hay holdout nuevo implementado y bloqueado, no hay confirmacion final.
- Si el test 2024-2025 se usa para seleccion, el resultado se marca como exploratorio.
- Si falta el fuel anual, la alerta falla; no se publica como operativa.
- Si el manuscrito no se puede regenerar desde las tablas, no se actualiza el DOI.

## 14. Congelamiento

Este protocolo no se modifica despues de ver el holdout nuevo. Cualquier cambio posterior se registra como enmienda con fecha y justificacion, y invalida la confirmacion si afecta el modelo o las metricas primarias.
