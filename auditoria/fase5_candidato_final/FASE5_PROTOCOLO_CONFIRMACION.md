# Fase 5 - Protocolo de confirmación espacial

**Fecha:** 2026-10-27  
**Estado:** congelado antes de la primera evaluación del holdout.  
**Candidato:** `submission-candidate-2026-10-27-r2`  
**Holdout:** `datos/processed/holdout_confirmacion/`  
**Uso:** una sola evaluación. No se usa para seleccionar modelo, variables ni umbrales.

## 1. Principio

La confirmación final es espacial. El holdout de 31 bloques se mantuvo separado
del desarrollo y nunca se usó para elegir el modelo. La evaluación primaria será
sobre los bloques de confirmación en 2024–2025, reportando 2024 y 2025 por separado.

El test 2024–2025 ya fue observado, por lo que la confirmación se describe como
**espacial retrospectiva**, no como un test ciego.

## 2. Particiones

### 2.1 Entrenamiento de confirmación

- Periodo: 2018–2022.
- Excluir todas las celdas del holdout de confirmación.
- Excluir todas las celdas del buffer de 20 km.
- Recalcular priors causales excluyendo los fuegos de las celdas holdout y buffer.
- Usar exactamente la misma capacidad y features que el candidato congelado.

### 2.2 Calibración de confirmación

- Periodo: 2023.
- Excluir celdas holdout y buffer.
- Ajustar Platt monótono solo con estas celdas.
- Calcular cutoffs F0.5 solo con estas celdas y congelarlos antes de la evaluación.
- No usar 2024–2025 ni el holdout para calibrar o elegir cutoffs.

### 2.3 Evaluación de confirmación

- Periodo primario: 2024–2025.
- Celdas: únicamente bloques de confirmación.
- Reportar 2024 y 2025 separados.
- Reportar el modelo operativo completo:
  - M3 para celdas con `freq_hist_1km > 0`;
  - M2 para celdas con `freq_hist_1km == 0`.
- Reportar también M1, M2 y M3 por separado, como diagnóstico.
- Reportar propagación en su dominio `t-1`.
- Evaluar una sola vez. No repetir por resultados.

### 2.4 Referencia held-in

Las celdas de desarrollo 2024–2025 pueden reportarse como referencia, pero no
son confirmación y no se usan para la decisión.

## 3. Priors y features

- `freq_hist_1km`, `dist_foco_hist_km`, `densidad_hist_*` y corredores deben
  recalcularse para el entrenamiento de confirmación.
- Los fuegos de las celdas holdout no pueden entrar en priors de train/cal.
- Los fuegos del buffer se excluyen de train/cal; para la evaluación del holdout
  se permite el contexto histórico de celdas vecinas, pero nunca los fuegos de
  la propia celda holdout.
- No se agregan variables nuevas.
- No se cambia la definición D2.

## 4. Modelos y calibración

- M1, M2, M3 y propagación con la misma capacidad del candidato: 45/60 rondas,
  23/31 hojas, seed 42.
- M3 se usa para celdas con historia; M2 para celdas sin historia.
- M1 queda como baseline.
- Calibración Platt monótona en 2023 de desarrollo.
- Cutoffs congelados antes de mirar confirmación.
- No se reajusta ningún hiperparámetro.

## 5. Métricas primarias

- ROC-AUC.
- PR-AUC.
- Brier, BSS y ECE.
- Precisión, recall y F0.5 con cutoff congelado.
- Métricas poblacionales.
- Comparaciones M3−M2, M2−M1 y M3−M1, con bootstrap agrupado por bloque.

## 6. Regla de lectura

- Si M2/M3 mantienen el patrón de 2023, se reporta confirmación y se mantiene la decisión.
- Si M3 cae más de lo esperado, se reporta el resultado negativo; no se reentrena.
- Si M2 no confirma su ventaja espacial, se suaviza la afirmación correspondiente.
- Si la confirmación es mala, el paper se reescribe como benchmark y resultado negativo.

## 7. Condiciones de parada

La evaluación se detiene y se documenta, sin mirar métricas, si:

- cambió el hash del holdout;
- falta un modelo o calibrador;
- falta el cutoff;
- el train de confirmación conserva celdas holdout o buffer;
- hay fuga de `test_*` en la construcción de confirmación.

## 8. Artefactos esperados

- `auditoria/FASE5_CONFIRMACION.md`
- `auditoria/FASE5_CONFIRMACION.json`
- `tablas/tabla13_confirmacion_espacial.csv`
- `tablas/tabla13_confirmacion_poblacional.csv`
- `tablas/tabla13_confirmacion_propagacion.csv`
- `tablas/tabla13_confirmacion_comparaciones.csv`

## 9. Regla de cierre

El holdout se evalúa una vez. Cualquier cambio posterior de modelo invalida la
confirmación y obliga a declararlo explícitamente en el manuscrito.
