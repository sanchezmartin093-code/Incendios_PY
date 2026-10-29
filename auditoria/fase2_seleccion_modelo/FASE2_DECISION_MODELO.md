# Fase 2 científica - Decisión de modelo

**Fecha:** 2026-10-27  
**Protocolo:** `auditoria/FASE2_PROTOCOLO_SELECCION.md`  
**Datos de selección:** validación 2023 y holdout espacial de desarrollo 2023.  
**Test 2024-2025:** no usado para esta decisión.

## 1. Validación temporal 2023

| Región | Modelo | ROC-AUC | PR-AUC | F0.5 |
|---|---|---:|---:|---:|
| Chaco | M1 | 0.6354 | 0.1480 | 0.1762 |
| Chaco | M2 | 0.7817 | 0.2452 | 0.2933 |
| Chaco | M3 | 0.7844 | 0.2542 | 0.2979 |
| Oriental | M1 | 0.7041 | 0.1818 | 0.2301 |
| Oriental | M2 | 0.7496 | 0.2221 | 0.2631 |
| Oriental | M3 | 0.7628 | 0.2441 | 0.2875 |

Diferencias pareadas M3 − M2, bootstrap agrupado por fecha, B = 1.000:

| Región | ΔROC-AUC [IC 95 %] | ΔPR-AUC [IC 95 %] |
|---|---|---|
| Chaco | +0.0028 [+0.0001, +0.0054] | +0.0089 [+0.0034, +0.0144] |
| Oriental | +0.0132 [+0.0097, +0.0169] | +0.0221 [+0.0166, +0.0272] |

**Decisión temporal:** M3 se retiene en ambas regiones. Cumple ΔROC ≥ 0, IC inferior > −0.01 y ΔPR ≥ 0.

## 2. Validación espacial de desarrollo 2023

Bloques de 50 km, buffer de 20 km, evaluación 2023:

| Región | Modelo | ROC-AUC | PR-AUC |
|---|---|---:|---:|
| Chaco | M1 | 0.6456 | 0.1668 |
| Chaco | M2 | 0.6773 | 0.1760 |
| Chaco | M3 | 0.6496 | 0.1630 |
| Oriental | M1 | 0.7176 | 0.1574 |
| Oriental | M2 | 0.7470 | 0.1820 |
| Oriental | M3 | 0.7277 | 0.1608 |

Diferencias pareadas M2 − M3, bootstrap agrupado por bloque, B = 1.000:

| Región | ΔROC-AUC [IC 95 %] | ΔPR-AUC [IC 95 %] |
|---|---|---|
| Chaco | +0.0277 [+0.0076, +0.0510] | +0.0130 [+0.0046, +0.0227] |
| Oriental | +0.0193 [+0.0130, +0.0272] | +0.0212 [+0.0136, +0.0290] |

**Decisión espacial:** M2 es el modelo más robusto en bloques no vistos. M3 no debe presentarse como extrapolador espacial. M1 tampoco es un modelo espacial fuerte.

## 3. Fallback en celdas sin historia

Subconjunto `freq_hist_1km == 0`, validación 2023:

| Región | Modelo | ROC-AUC | PR-AUC |
|---|---|---:|---:|
| Chaco | M1 | 0.6650 | 0.0841 |
| Chaco | M2 | 0.7677 | 0.1437 |
| Chaco | M3 | 0.7626 | 0.1459 |
| Oriental | M1 | 0.7018 | 0.0740 |
| Oriental | M2 | 0.7440 | 0.0892 |
| Oriental | M3 | 0.7421 | 0.0949 |

M1 es inferior al mejor modelo por 0.098 ROC-AUC en Chaco y 0.040 en Oriental. **M1 no se sostiene como fallback operativo.**

M2 vs M3 en celdas sin historia:

- Chaco: ΔROC = +0.0051 [+0.0013, +0.0093]; ΔPR = −0.0023 [−0.0071, +0.0027].
- Oriental: ΔROC = +0.0019 [−0.0047, +0.0082]; ΔPR = −0.0058 [−0.0104, −0.0017].

La diferencia ROC está dentro de δ = 0.01 y no es operativamente material. M2 conserva una ventaja espacial y una ligera ventaja en Chaco; M3 conserva mejor PR en Oriental. Se elige **M2 como fallback** por su robustez espacial.

## 4. Lead time 2023

Clima `t-1` (persistencia, 24 h):

| Región | Modelo | ROC-AUC | PR-AUC |
|---|---|---:|---:|
| Chaco | M1 | 0.6342 | 0.1467 |
| Chaco | M2 | 0.7818 | 0.2444 |
| Chaco | M3 | 0.7831 | 0.2523 |
| Oriental | M1 | 0.6965 | 0.1765 |
| Oriental | M2 | 0.7463 | 0.2167 |
| Oriental | M3 | 0.7613 | 0.2384 |

M3 conserva la ventaja temporal con `t-1`. El sistema sigue siendo nowcast/hindcast, no pronóstico numérico.

## 5. Propagación 2023

| Región | ROC-AUC | PR-AUC | F0.5 |
|---|---:|---:|---:|
| Chaco | 0.8942 | 0.6418 | 0.6913 |
| Oriental | 0.8022 | 0.4173 | 0.4879 |

No hay selección de modelo para propagación; se mantiene la capa táctica.

## 6. Decisión final

1. **Ignición temporal:** M3 en Chaco y Oriental.
2. **Fallback operativo:** M2 para celdas con `freq_hist_1km == 0`.
3. **M1:** baseline biofísico y comparador; **no** fallback operativo.
4. **Espacial:** M2 es el comparador espacial más fuerte; M3 no se presenta como extrapolador.
5. **Propagación:** sin cambios.

La arquitectura operativa cambia de **M3/M1** a **M3/M2**.

## 7. Consecuencias

- Actualizar `src/alerta.py` y `scripts/09_servicio_alerta.py`.
- Actualizar manuscrito, suplemento y README: eliminar “M3/M1 gating” y la afirmación de que M1 es el fallback correcto.
- Reportar el fallback M1 como hipótesis rechazada, no como resultado positivo.
- Mantener el test 2024-2025 como descriptivo, nunca como confirmación.
- No tocar el holdout de confirmación.

## 8. Estado

Decisión congelada. Cualquier cambio posterior debe registrarse como enmienda y no puede basarse en el test 2024-2025.

## 9. Ejecución

- `src/alerta.py` usa M3 para celdas con historia y M2 para celdas sin historia; M1 queda como diagnóstico.
- `scripts/09_servicio_alerta.py` exporta `probabilidad_m2` y `fallback_m2`.
- `tests/test_alerta.py` verifica la carga de M2 y la columna de fallback.
- Manuscrito ES/EN, suplemento, README, figuras y productos de alerta regenerados.
- El holdout de confirmación no se usó.
