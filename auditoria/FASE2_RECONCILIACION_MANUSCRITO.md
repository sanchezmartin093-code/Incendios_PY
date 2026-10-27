# Fase 2 - Reconciliación del manuscrito y artefactos

**Fecha:** 2026-10-27  
**Base de selección:** validación 2023 + holdout espacial de desarrollo 2023.  
**Test 2024–2025:** solo descriptivo; no usado para selección.

## 1. Afirmaciones corregidas

| Afirmación anterior | Evidencia actual | Texto nuevo |
|---|---|---|
| “Test ciego 2024–2025” | El test ya fue observado durante el desarrollo. | “Test retrospectivo out-of-time 2024–2025”. |
| “M3/M1 gating” | M1 no supera a M2 en celdas sin historia; M2 es el fallback validado. | “M3/M2: M3 para historia observada, M2 para celdas sin historia; M1 baseline”. |
| “M3 pierde frente a M1 en bloques espaciales” | En 2023, M2 0.6773/0.7470 > M3 0.6496/0.7277 > M1 0.6456/0.7176. | M2 es el comparador espacial; M3 no es extrapolador; M1 no es fallback. |
| “M1 es el fallback correcto” | M1 sin historia: 0.6650/0.7018 vs M2 0.7677/0.7440 y M3 0.7626/0.7421. | M1 rechazado como fallback operativo. |
| Conteos 17/24/33-34 variables | Booster real: M1 24, M2 35, M3 44/45. | Conteos corregidos. |
| 62/66 tests | Suite actual: 71 passed, 2 warnings. | 71 tests, `pytest`. |
| Métricas de test pre-auditoría | Rerun actual: M3 0.8021/0.7534; propagación 0.9039/0.7859. | Tablas y texto actualizados. |
| Precisión poblacional 0.062/0.206 | Rerun actual: 0.082/0.198. | Texto actualizado. |
| “Datos archivados en DOI actual” | DOI v1.0.0 es pre-auditoría; el release actual es Fase 3. | Se declara actualización pendiente en Fase 3. |

## 2. Evidencia de selección

- `tablas/tabla10_validacion_2023.csv`
- `tablas/tabla10_validacion_2023_fallback.csv`
- `tablas/tabla10_validacion_2023_poblacional.csv`
- `tablas/tabla10_validacion_2023_lead_time.csv`
- `tablas/tabla10_validacion_2023_propagacion.csv`
- `tablas/tabla10_comparaciones_pareadas_2023.csv`
- `tablas/tabla10_comparaciones_fallback_2023.csv`
- `tablas/tabla08b_validacion_espacial_2023.csv`
- `tablas/tabla08b_comparaciones_pareadas_2023.csv`

## 3. Test descriptivo

- `tablas/tabla02_metricas_por_anio.csv`: 2024/2025 separados.
- `tablas/tabla07_evaluacion_poblacional_por_anio.csv`: población por año.
- `tablas/tabla08_validacion_espacial_por_anio.csv`: test espacial descriptivo; no selección.
- `tablas/tabla09_evaluacion_lead_time_por_anio.csv`: lead time por año.
- `tablas/tabla04_benchmarking_internacional.csv`: benchmark actual.

## 4. Cautelas que quedaron en el texto

- 90.65 % de celdas de test presentes en train por solapamiento temporal.
- 2025 usa fuel 2023 con lag 2.
- La cobertura conformal es empírica, no garantía formal bajo cambio temporal.
- El lock tiene versiones exactas pero no hashes de ruedas.
- No se usa el holdout de confirmación.

## 5. Estado

Manuscrito ES/EN, suplemento ES/EN, README, figuras principales y productos de alerta actualizados. La release/DOI queda para Fase 3.
