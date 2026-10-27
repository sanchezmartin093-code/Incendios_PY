# Fase 0 - Decision de holdout de confirmacion

**Fecha:** 2026-10-27  
**Estado:** decision congelada; implementacion pendiente en Fase 1.

## 1. Situacion

- El test 2024-2025 ya fue observado y no es un holdout ciego.
- La validacion espacial actual (`20_validacion_espacial_bloques.py`) tambien fue usada durante el desarrollo, por lo que no sirve como confirmacion final.
- 2026 no esta disponible como test prospectivo completo:
  - FIRMS llega hasta 2026-01-04.
  - El clima procesado llega hasta 2025-12-31.
  - No hay un ano 2026 completo de labels, clima y fuel.

## 2. Decision

**Holdout primario de confirmacion: nuevo holdout espacial por bloques, independiente del usado en `20`.**

Diseno:

- Unidad: bloques de 50 x 50 km en UTM 21S.
- Proporcion reservada: 20 % de los bloques elegibles.
- Buffer: 20 km alrededor de los bloques reservados; esas celdas no entran en entrenamiento.
- Sal: nueva, distinta de la usada por `20_validacion_espacial_bloques.py`.
- Excluir explicitamente los bloques del holdout actual de `20` para que no haya reutilizacion.
- Recalcular priors historicos excluyendo los fuegos de las celdas reservadas.
- Reservar el conjunto antes de mirar cualquier métrica sobre el.
- Usarlo una sola vez, en Fase 5.

Si el 20 % restante no alcanza un numero minimo predefinido de positivos por region, se podra ampliar a 25 % antes de la primera evaluacion, nunca despues.

## 3. Holdout secundario

Si en el futuro se completa 2026:

- usarlo como validacion prospectiva secundaria;
- no usarlo para elegir el modelo;
- reportar por separado 2024, 2025 y 2026;
- recordar que 2026 usaria fuel 2023 con lag 3 si no se incorpora C3.

## 4. Validacion externa

La validacion en Chaco argentino/boliviano queda como mejora de Fase 4. No reemplaza al holdout espacial interno y requiere harmonizacion de clases, labels y fuel.

## 5. Reglas

1. El holdout nuevo no se usa para elegir variables, umbrales, capacidad, C3 o ET0.
2. No se reportan metricas del holdout nuevo antes de Fase 5.
3. La evaluacion final es una sola; si se cambia el modelo despues, se invalida la confirmacion.
4. El holdout actual de `20` queda como validacion de desarrollo, no como confirmacion.

## 6. Criterio de cierre

- Algoritmo de particion documentado e implementado en Fase 1.
- Lista de celdas/bloques reservados guardada y hasheada.
- Test 2024-2025 y holdout nuevo separados en todos los scripts.
- Manuscrito informa claramente que 2024-2025 es retrospectivo y que el holdout nuevo es la confirmacion.
