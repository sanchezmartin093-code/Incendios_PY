# Fase 2 científica - Protocolo de selección congelado

**Fecha:** 2026-10-27  
**Rama:** `audit/fase0-freeze`  
**Estado:** congelado antes de ejecutar la validación de selección.

## 1. Principio

La selección de modelo se hace exclusivamente con:

- train 2018-2022;
- validación temporal 2023;
- holdout espacial de desarrollo evaluado en 2023.

El test 2024-2025 es **retrospectivo out-of-time** y se usa solo de forma descriptiva después de congelar la decisión. El holdout de confirmación no se usa hasta Fase 5.

## 2. Modelos comparados

- **M1:** biophysical baseline/fallback.
- **M2:** ignición base con ancla histórica y ecotono.
- **M3:** híbrido socioecológico.

Los tres usan la misma capacidad: 45/60 rondas y 23/31 hojas, seed 42, calibración Platt monótona. La capa de propagación se valida por separado y no participa en la selección M1/M2/M3.

## 3. Estimandos primarios

### 3.1 Uso temporal (nowcast)

- ΔROC-AUC = M3 − M2 en validación 2023, caso-control, por región.
- ΔPR-AUC = M3 − M2 en validación 2023, caso-control, por región.
- Incertidumbre: bootstrap pareado agrupado por fecha, 1.000 réplicas, seed 42.

### 3.2 Uso espacial

- ROC-AUC y PR-AUC en bloques de desarrollo 50 km con buffer de 20 km, evaluados en 2023, por región y modelo.
- Misma incertidumbre agrupada por bloque.

### 3.3 Fallback

- ROC-AUC y PR-AUC en celdas con `freq_hist_1km == 0` en validación 2023, por región y modelo.

### 3.4 Métricas secundarias

- Brier, BSS, ECE.
- Precisión, recall y F0.5 con cutoff congelado en 2023.
- Métricas poblacionales 2023.
- Lead time `t-1` 2023.
- Cost-loss 2023.

## 4. Regla de decisión propuesta y congelada

Margen práctico de no-inferioridad: **δ = 0,01** para ROC-AUC y PR-AUC.

### 4.1 Modelo temporal

M3 se retiene para una región si:

1. ΔROC-AUC (M3 − M2) ≥ 0;
2. límite inferior del IC 95 % pareado de ΔROC-AUC > −0,01;
3. ΔPR-AUC (M3 − M2) ≥ 0.

Si M3 no cumple en una región, se usa M2 en esa región o se documenta un esquema regional. No se fuerza un único modelo global.

### 4.2 Ventaja espacial

M2 se reporta como mejor en bloques no vistos si el IC 95 % pareado de ΔROC-AUC (M2 − M3) excluye 0. Esta ventaja no revierte por sí sola la elección temporal: afecta la afirmación de extrapolación espacial.

### 4.3 Fallback

M1 se retiene como fallback para celdas sin historia si su ROC-AUC 2023 no es inferior al mejor modelo por más de δ y su calibración es aceptable. Si no, se documenta el cambio de fallback.

### 4.4 Propagación

No hay selección de modelo. Se reportan métricas 2023 y luego 2024/2025 por separado.

## 5. Secundarios y exploratorios

Son secundarios:

- estabilidad con seeds 123 y 2026;
- sensibilidad a tamaño de bloque;
- subgrupos por estación;
- benchmark FWI/INPE.

No se usan para cambiar la decisión primaria ni para elegir el resultado más favorable.

## 6. Prohibiciones

- No usar 2024-2025 para seleccionar modelo, features, cutoff o narrativa.
- No usar el holdout de confirmación.
- No recalcular cutoffs en test.
- No agregar features nuevas.
- No cambiar la regla después de ver los resultados.
- No afirmar superioridad universal de M3.

## 7. Artefactos previstos

- `auditoria/FASE2_AUDITORIA_MODELOS.md`
- `tablas/tabla10_validacion_2023.csv`
- `tablas/tabla10_validacion_2023_poblacional.csv`
- `tablas/tabla10_validacion_2023_lead_time.csv`
- `tablas/tabla08b_validacion_espacial_2023.csv`
- `auditoria/FASE2_DECISION_MODELO.md`
- `auditoria/FASE2_DECISION_MODELO.json`

## 8. Criterio de cierre

La Fase 2 científica se cierra cuando:

- la decisión está registrada y congelada;
- el test se reporta solo como retrospectivo;
- el manuscrito, suplemento y README se reconstruyen desde los artefactos;
- el holdout de confirmación sigue intacto;
- los tests y `00_validar_datos.py` pasan.
