# Fase 2.0 - Cierre de deuda técnica previa a Fase 2

**Fecha:** 2026-10-27  
**Estado:** deuda técnica cerrada; Fase 2 científica todavía no iniciada.  
**No se actualizó DOI, manuscrito ni release.**

## 1. Holdout de confirmación

Implementado en `src/holdout.py` y `scripts/25_definir_holdout_confirmacion.py`.

- Bloques de 50 km.
- 20 % de bloques elegibles.
- Buffer de 20 km.
- Independiente del holdout de desarrollo de `20`.
- Generado solo desde la grilla, sin labels ni predicciones.
- Guardado en `datos/processed/holdout_confirmacion/`.

Resultado:

- Bloques totales: 201.
- Bloques de desarrollo: 44.
- Bloques de confirmación: 31.
- Celdas de confirmación: 68.719.
- Celdas de buffer: 95.031.

Metadata y hashes: `auditoria/FASE2_HOLDOUT_META.json`.

Regla: **no se evalúa hasta Fase 5**.

## 2. Dataset determinista por particiones

`04_construir_dataset.py` ahora escribe:

`datos/processed/dataset_unificado_candidatos/dataset_YYYY.parquet`

Cada partición:

- está ordenada por `cell_id, fecha`;
- se escribe por año;
- evita el sort monolítico que agotó el temp de DuckDB.

Resultado:

- 8 particiones: 2018-2025.
- 915.561.114 filas totales.
- 397.448 celdas.
- 7.901,79 MB.
- Manifiesto con filas y SHA-256 por partición.

Reproducibilidad:

- Se hizo una segunda build completa.
- Las 8 particiones dieron hashes **idénticos**.
- Ver `auditoria/FASE2_REPRO_BUILD_CHECK.txt`.
- Manifiesto copiado a `auditoria/FASE2_DATASET_MANIFEST.json`.

## 3. Procedencia climática

Creado `datos/processed/clima_provenance.csv` y copia versionada en `auditoria/FASE2_CLIMA_PROVENANCE.csv`.

- Temperatura, precipitación, viento, ráfaga y VPD: CDS ERA5-Land vía `lote_clima_*`.
- ET0: Hargreaves-Samani sobre Tmax/Tmin/latitud de CDS.
- `swvl1`: CDS `volumetric_soil_water_layer_1`.
- PEV: cacheado, solo diagnóstico; no feature.
- Open-Meteo: no usado en el clima final.
- Verificación de una fila base contra CDS 2018-01-01: diferencia máxima 0.0.

## 4. Warm-up de 60 días

Se excluyó del train:

`2018-01-01` a `2018-03-01`

Primera fecha de train: `2018-03-02`.

Conteos:

- `train_ignicion`: 1.003.981 filas; 91.271 positivos.
- `train_propagacion`: 5.492.223 filas; 499.293 positivos.
- `val_ignicion`: 188.507.
- `val_propagacion`: 832.766.
- `test_ignicion`: 376.552.
- `test_propagacion`: 1.437.414.

Sin fuel futuro en train, val ni test (`anio_mapa < anio_fila`).

## 5. CI con lock

- `requirements-lock.txt` regenerado con pins exactos del entorno.
- CI ahora instala `requirements-lock.txt`, ejecuta `pip check` y `pytest`.
- Workflow actualizado en `.github/workflows/ci.yml`.
- **Pendiente:** hashes de ruedas. `pip-compile --generate-hashes` agotó el tiempo de red; no se inventaron hashes. El lock es exacto en versiones, pero no está en modo `--require-hashes`.

## 6. Métricas separadas 2024/2025

Se generaron tablas nuevas:

- `tabla02_metricas_por_anio.csv`
- `tabla04_benchmarking_por_anio.csv`
- `tabla07_evaluacion_poblacional_por_anio.csv`
- `tabla08_validacion_espacial_por_anio.csv`
- `tabla09_evaluacion_lead_time_por_anio.csv`

Resultados principales:

### Ignición M3, caso-control

| Region | Año | fuel_lag | ROC-AUC | PR-AUC | Precision F0.5 |
|---|---:|---:|---:|---:|---:|
| Chaco | 2024 | 1 | 0.8030 | 0.2820 | 32.65 % |
| Chaco | 2025 | 2 | 0.8011 | 0.2874 | 33.06 % |
| Oriental | 2024 | 1 | 0.7532 | 0.2316 | 28.84 % |
| Oriental | 2025 | 2 | 0.7539 | 0.2377 | 31.89 % |

### Propagación

| Region | Año | fuel_lag | ROC-AUC | PR-AUC | Precision F0.5 |
|---|---:|---:|---:|---:|---:|
| Chaco | 2024 | 1 | 0.9069 | 0.6937 | 82.42 % |
| Chaco | 2025 | 2 | 0.8982 | 0.6915 | 87.19 % |
| Oriental | 2024 | 1 | 0.7911 | 0.3872 | 64.59 % |
| Oriental | 2025 | 2 | 0.7785 | 0.3578 | 65.83 % |

### Poblacional M3

| Region | Año | ROC-AUC | Precision | Recall |
|---|---:|---:|---:|---:|
| Chaco | 2024 | 0.8074 | 0.083 % | 34.95 % |
| Chaco | 2025 | 0.8024 | 0.080 % | 19.99 % |
| Oriental | 2024 | 0.7607 | 0.181 % | 22.50 % |
| Oriental | 2025 | 0.7651 | 0.223 % | 18.90 % |

### Espacial

| Region | Modelo | ROC-AUC holdout |
|---|---|---:|
| Chaco | M1 | 0.6536 |
| Chaco | M2 | 0.6905 |
| Chaco | M3 | 0.6751 |
| Oriental | M1 | 0.6862 |
| Oriental | M2 | 0.7159 |
| Oriental | M3 | 0.7006 |

M2 sigue siendo el mejor en holdout espacial. La narrativa del manuscrito debe corregirse en Fase 2.

## 7. Verificación

- Tests: **71 passed, 2 warnings**.
- `00_validar_datos`: **0 FAIL**.
- Dataset: 8/8 particiones idénticas entre builds.
- Holdout: 31 bloques, 68.719 celdas, disjunto del desarrollo.
- Train empieza en 2018-03-02.
- Alertas regeneradas.
- Figuras principales regeneradas.

## 8. Pendientes legítimos

- Hashes de ruedas en el lock.
- Manuscrito, suplemento, README y DOI.
- Uso del holdout nuevo, solo en Fase 5.
- C3, FWI components, población/carreteras, SPI/SPEI, LFMC y NWP.
- Fase 2 científica: reconciliar M2/M3/fallback y actualizar narrativa.

## 9. Decisión

La deuda técnica previa queda cerrada con una excepción explícita: los hashes del lock no se generaron por timeout de red. No se avanza a Fase 2 hasta aceptar esa excepción o completar los hashes.
