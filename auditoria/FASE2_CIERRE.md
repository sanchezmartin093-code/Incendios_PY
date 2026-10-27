# Fase 2 científica - Cierre

**Fecha:** 2026-10-27  
**Rama:** `audit/fase0-freeze`  
**Protocolo:** `auditoria/FASE2_PROTOCOLO_SELECCION.md`  
**Decisión:** `auditoria/FASE2_DECISION_MODELO.md`

## 1. Qué se hizo

1. Se congeló el protocolo de selección antes de ejecutar la validación.
2. Se auditó la identidad de M1/M2/M3 y la capacidad de los boosters.
3. Se evaluó la validación temporal 2023.
4. Se evaluó el holdout espacial de desarrollo en 2023.
5. Se evaluó el fallback en celdas sin historia.
6. Se evaluó lead time `t-1` 2023 y propagación 2023.
7. Se aplicó la regla preespecificada con δ = 0.01.
8. Se cambió la arquitectura operativa de M3/M1 a M3/M2.
9. Se actualizaron manuscrito ES/EN, suplemento ES/EN, README, figuras y productos de alerta.
10. Se regeneraron y verificaron los artefactos.

## 2. Resultado de selección

- M3 se retiene para nowcast temporal en Chaco y Oriental.
- M2 es el mejor modelo en bloques espaciales 2023 y el fallback operativo.
- M1 se rechaza como fallback: en celdas sin historia queda muy por debajo de M2/M3.
- Propagación se mantiene sin cambios.

## 3. Evidencia

- `tablas/tabla10_validacion_2023.csv`
- `tablas/tabla10_validacion_2023_fallback.csv`
- `tablas/tabla10_validacion_2023_poblacional.csv`
- `tablas/tabla10_validacion_2023_lead_time.csv`
- `tablas/tabla10_validacion_2023_propagacion.csv`
- `tablas/tabla10_comparaciones_pareadas_2023.csv`
- `tablas/tabla10_comparaciones_fallback_2023.csv`
- `tablas/tabla08b_validacion_espacial_2023.csv`
- `tablas/tabla08b_comparaciones_pareadas_2023.csv`

## 4. Cambios de código y documentación

- `src/alerta.py`: M3 para historia, M2 para celdas sin historia; M1 diagnóstico.
- `scripts/09_servicio_alerta.py`: exporta `probabilidad_m2` y `fallback_m2`.
- `tests/test_alerta.py`: verifica M2 y la columna de fallback.
- `scripts/26_validacion_2023.py`
- `scripts/27_validacion_espacial_2023.py`
- `scripts/29_lead_time_2023.py`
- `scripts/30_validacion_propagacion_2023.py`
- Manuscrito, suplemento, README, figuras y alertas regeneradas.

## 5. Verificación

- `pytest`: **71 passed, 2 warnings**.
- `00_validar_datos.py`: **0 FAIL**.
- Quarto: manuscrito y suplemento renderizan en ES/EN.
- Holdout de confirmación: checksums sin cambios; no usado.
- Test 2024–2025: reportado como retrospectivo, no usado para selección.

## 6. Limitaciones que quedan explícitas

- El test 2024–2025 es retrospectivo y el 90.65 % de sus celdas aparecen en train por solapamiento temporal.
- 2025 usa fuel 2023 con lag 2.
- La cobertura conformal es empírica.
- El lock tiene versiones exactas pero no hashes de ruedas.
- La release/DOI actual es pre-auditoría; la actualización corresponde a Fase 3.

## 7. Decisión de cierre

Fase 2 científica cerrada. La siguiente fase es la release/DOI y la preparación de submission (Fase 3), no otra selección de modelo.
