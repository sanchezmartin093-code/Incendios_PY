# Fase 1 - Cierre de correcciones P0

**Fecha:** 2026-10-27  
**Rama:** `audit/fase0-freeze`  
**Base congelada:** `fase0-freeze-2026-10-27`  
**Estado:** correcciones P0 ejecutadas; resultados todavía retrospectivos.

> **Documento histórico.** Las métricas de este archivo fueron reemplazadas por el rerun de Fase 2.0. Ver `auditoria/FASE2_0_CIERRE.md`.

## Actualización Fase 2.0

Resueltos desde este cierre:

- C1: holdout de confirmación implementado y reservado para Fase 5.
- H2: dataset particionado por año y reproducible (8/8 hashes idénticos).
- H3: procedencia climática por variable y warm-up de 60 días.
- H5: CI con lock exacto; hashes de ruedas en excepción formal.
- H8: métricas separadas 2024/2025.

Siguen para Fase 2/release:

- C4/C8: manuscrito, suplemento y narrativa espacial.
- C5/H1: release, DOI y manifiesto.
- H9/H10: conteos y figuras/texto final.

## 1. Correcciones implementadas

| ID | Corrección | Evidencia |
|---|---|---|
| C2 | Lead time `t-1` ahora desplaza tambien `fdd_heladas_60d` y `deficit_hidro_60d`; las interacciones se recalculan con esos valores. | `scripts/21_evaluacion_lead_time.py`; `tests/test_lead_time.py` |
| C3 | El perfil estatico de MapBiomas usa un unico mapa, por defecto 2023; no procesa los mapas anuales. | `src/biomasa.py`; `scripts/03_procesar_combustible.py`; `tests/test_biomasa.py` |
| C6 | `07` falla si falta el umbral de validacion; la alerta falla si faltan umbrales, prevalencias o fuel anual; `14` no usa fallback hardcodeado. | `scripts/07_evaluacion_test_2024.py`; `src/alerta.py`; `scripts/14_auditoria_suplementaria.py`; `tests/test_alerta.py` |
| C7 | M2 final ahora usa la misma capacidad que M3/M1: 45/60 rondas, 23/31 hojas. | `scripts/06_entrenar_modelos.py`; `scripts/16_ablacion_modelo_hibrido.py`; `scripts/24_entrenar_m1.py` |
| H2 | Targets D2 se ordenan por `fecha, cell_id, task`; el dataset completo no se ordena porque el sort de 915M filas agoto el temp de DuckDB. | `scripts/05d_muestrear_poblacional_d2.py` |
| H4 | Se agrego un test de integracion del plan D2 de produccion (ignicion vs propagacion, buffer y negativos). | `tests/test_produccion_d2.py` |
| H6 | `00_validar_datos` ahora verifica fuel anual, `fuel_lag`, mapas no previos y ausencia de fuel futuro en particiones. | `scripts/00_validar_datos.py` |
| H7 | Se regeneraron las prevalencias D2 desde los conteos posteriores al rerun. | `scripts/19_estimar_prevalencias.py` |

## 2. Rerun ejecutado

Secuencia ejecutada:

`03 -> 04 -> 05d -> 19 -> 05e -> 06 -> 24 -> 16 -> 07 -> 08 -> 13 -> 18 -> 20 -> 21 -> 22 -> 14 -> 09 -> 11 -> 12 -> 10 -> 10b -> 15 -> 23 -> 24_generar_graphical_abstract`

## 3. Resultados actuales (retrospectivos 2024-2025)

### Test caso-control

| Region | Modelo | ROC-AUC | PR-AUC | Precision F0.5 |
|---|---|---:|---:|---:|
| Chaco | M3 ignition | 0.8020 | 0.2841 | 34.02 % |
| Chaco | M2 ignition | 0.7991 | 0.2755 | 33.35 % |
| Chaco | Propagation | 0.9033 | 0.6930 | 83.79 % |
| Oriental | M3 ignition | 0.7538 | 0.2343 | 28.69 % |
| Oriental | M2 ignition | 0.7407 | 0.2154 | 26.05 % |
| Oriental | Propagation | 0.7857 | 0.3756 | 63.34 % |

### Poblacional

| Region | Modelo | ROC-AUC | Precision | Recall |
|---|---|---:|---:|---:|
| Chaco | M3 ignition | 0.8061 | 0.088 % | 24.37 % |
| Chaco | M2 ignition | 0.8077 | 0.090 % | 24.17 % |
| Chaco | Propagation | 0.9077 | 9.82 % | 50.63 % |
| Oriental | M3 ignition | 0.7641 | 0.187 % | 23.77 % |
| Oriental | M2 ignition | 0.7557 | 0.164 % | 23.05 % |
| Oriental | Propagation | 0.7967 | 4.50 % | 19.72 % |

### Espacial

| Region | Modelo | ROC-AUC holdout |
|---|---|---:|
| Chaco | M1 | 0.6528 |
| Chaco | M2 | 0.6858 |
| Chaco | M3 | 0.6743 |
| Oriental | M1 | 0.6887 |
| Oriental | M2 | 0.7135 |
| Oriental | M3 | 0.6991 |

La tabla espacial actual no sostiene la afirmacion del manuscrito de que M3 pierde frente a M1. M2 es el mejor en holdout espacial; M3 sigue siendo el mejor en test caso-control. Esto debe reconciliarse en Fase 2.

### Lead time `t-1`

| Region | Modelo | ROC-AUC | PR-AUC | Precision | Recall |
|---|---|---:|---:|---:|---:|
| Chaco | M1 | 0.6356 | 0.1525 | 16.78 % | 21.49 % |
| Chaco | M3 | 0.8013 | 0.2818 | 32.01 % | 32.52 % |
| Oriental | M1 | 0.6800 | 0.1627 | 18.71 % | 27.33 % |
| Oriental | M3 | 0.7513 | 0.2315 | 29.10 % | 19.77 % |

## 4. Verificacion

- Tests: **70 passed, 2 warnings**.
- `00_validar_datos`: **0 FAIL**.
- Particiones D2: ordenadas por `fecha, cell_id, task`.
- Sin fuel futuro en particiones (`anio_mapa < anio_fila`).
- `03` volvio a generar el perfil estatico solo con 2023.
- Alerta 2024-09-09 regenerada y funcionando con umbrales/prevalencias actuales.

## 5. Pendientes que no son Fase 1

- C1: 2024-2025 sigue siendo retrospectivo; falta implementar y reservar el holdout espacial nuevo.
- C4/C8: manuscrito, suplemento y afirmaciones espaciales siguen desactualizados.
- C5/H1: release/DOI/manifiesto siguen sin corresponder al estado actual.
- H3: procedencia climatica por variable y warm-up de 60 dias.
- H5: CI con lock y chequeos de fuga.
- H8: metricas separadas 2024/2025.
- H9/H10: conteos de variables y figuras/texto final.
- H2: el dataset completo no se ordena; solo los targets que alimentan modelos.

## 6. Decision

Fase 1 se considera cerrada solo para los P0 de codigo. **No se actualiza DOI ni se envia manuscrito.** La Fase 2 debe reconciliar resultados y documentacion antes de cualquier release.
