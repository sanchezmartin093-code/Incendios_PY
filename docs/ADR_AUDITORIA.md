# ADR de auditoría y corrección — Incendios_PY

**Fecha:** 2026-10-26  
**Estado:** Semana 1 ejecutada parcialmente; D1–D4 aprobadas.  
**Alcance:** estado actual del *working tree* (no un commit reproducible).

## Decisiones aprobadas (2026-10-26)

- **D1:** MapBiomas real, con normalización y reentrenamiento.
- **D2:** ignición estricta + propagación restringida al dominio t-1.
- **D3:** FWI/INPE completos.
- **D4:** población post-gating del periodo de test.

## Contexto

La auditoría detectó que el repositorio mezcla versiones:

- `train/val/test` y los modelos usan el perfil sintético.
- `combustible_mapbiomas_1km.parquet` actual es MapBiomas real.
- `dataset_unificado_candidatos.parquet` actual usa el perfil real.
- La alerta carga el perfil real y el JSON de prevalencias del dataset sintético.
- El filtro de confianza FIRMS comparaba `"low"` contra valores reales `l/n/h`.
- La precipitación diaria se agregaba con `nanmax` en lugar de sumar incrementos horarios.
- El manifiesto anterior tenía hashes desactualizados y archivos inexistentes.

## Decisiones requeridas

### D1. Perfil de combustible

**Propuesta:** usar MapBiomas real y reentrenar; actualizar el manuscrito a la colección/año efectivamente usados. El archivo real disponible es `mapbiomas_paraguay_collection2_2023.tif` (Colección 2, 2023), no la Colección 4 declarada.

**Alternativa:** mantener el perfil sintético solo como demostración de software; en ese caso:
- normalizarlo para que las fracciones sumen 100;
- agregar una máscara de agua/urbano;
- eliminar conclusiones ecológicas, SHAP de biomasa, proxy INPE y alerta operativa.

**No ejecutar** `03 → 04 → 05 → 06 → 16` hasta aprobar D1.

### D2. Definición de targets

**Propuesta:** separar formalmente:

- `target_ignicion`: celda sin fuego en la misma celda en los últimos N días y sin fuego t-1 dentro de R km.
- `target_propagacion`: celdas dentro de R km de un fuego t-1, preferentemente excluyendo igniciones nuevas.

**Alternativa:** renombrar el target actual a “ocurrencia de fuego” y retirar “ignición primaria” del título, abstract y conclusiones.

**No ejecutar** el reentrenamiento hasta aprobar D2.

### D3. Benchmark

**Propuesta A:** implementar FWI completo (FFMC, DMC, DC, BUI, FWI) e INPE real.

**Propuesta B:** renombrar el benchmark a “ISI” y “proxy INPE”; eliminar ECE/BSS de índices no calibrados y suavizar las conclusiones.

**No ejecutar** `13` ni `16` como benchmark definitivo hasta aprobar D3.

### D4. Marco poblacional

**Propuesta:** población = celdas-día elegibles post-gating del periodo de test; usar la misma prevalencia en evaluación, tablas y alerta. Reportar siempre case-control y población por separado.

**No ejecutar** `18` ni `19` hasta aprobar D4.

## Cambios de Semana 1 ya aplicados

1. **FIRMS**
   - `src/firms.py`: el filtro de baja confianza ahora reconoce `l` y `low`.
   - `tests/test_firms.py`: fixture con valores reales `n/l/h`; exige que `l` y `low` se eliminen.
   - Impacto medido sobre el crudo actual: 2.176.151 → 1.977.568 detecciones y 958.857 → 892.533 celdas-día al excluir `l`.

2. **Clima**
   - `src/clima_cds.py`: nueva `agregar_precipitacion_diaria`; distingue acumulado desde las 00 UTC de incrementos horarios y suma en el segundo caso.
   - `src/clima.py`: `calcular_indices_fwi_y_sequedad` valida continuidad diaria por nodo y falla ante gaps.
   - `tests/test_clima_cds.py`: casos acumulado, incrementos horarios y NaN.
   - `tests/test_clima.py`: test de gap temporal.
   - Verificación sobre `era5_land_py_2024_Q3.nc` (nodo de prueba): 92 días continuos, máximo diario 156,9 mm y total trimestral 500,5 mm; antes se tomaba el máximo horario, no la suma diaria.

3. **Fase 0**
   - `fuentes/MANIFIESTO_DATOS_v2.csv`: inventario con hashes actuales y notas de linaje.
   - `scripts/00_verificar_manifiesto.py`: verificación de existencia/hash; falla ante `AUSENTE` o `MISMATCH`.

## Estado de datos al cierre de Semana 1

| Artefacto | Estado | Acción pendiente |
|---|---|---|
| `focos_viirs_diarios.parquet` | Regenerado: 892.533 celdas-día, sin `l/low` | Deduplicar overpasses y validar fecha local |
| `clima_era5_diario.parquet` | v2 CDS 3-hourly: 1.870.080 filas, 640 nodos, 2.922 días, máx 237,38 mm/día, 0 valores >300 mm | Revisar la resolución 3-hourly y actualizar el texto |
| `combustible_mapbiomas_1km.parquet` | Real (Colección 2, 2023) | Normalizar, documentar colección y reentrenar |
| `dataset_unificado_candidatos.parquet` | v2: 915.562.849 filas, 867.354 fuegos, 397.336 celdas, 2.919 días, prevalencia 0,000947 | Implementar targets D2 y reconstruir train/val/test |
| `train/val/test` | v1 sintéticos + columnas D2 | Remuestrear por target y reentrenar |
| Modelos y tablas | v1 sintéticos | Reentrenar y regenerar |

### Clima CDS v2

- La solicitud horaria trimestral excedía el límite de costo de CDS.
- Se usó resolución **3-hourly** trimestral (32 archivos NetCDF, conservados).
- La precipitación diaria se reconstruye descartando el primer paso (acumulado del día anterior) y usando el último valor de la cola si es monótona.
- Verificación: 1.870.080 filas, 640 nodos, 2.922 días, máximo diario 237,38 mm y 0 valores >300 mm.
- Limitación: la resolución 3-hourly puede subestimar picos intradiarios de VPD/viento; el manuscrito debe declararlo.

## Targets D2 aplicados

Definiciones implementadas en `src/targets.py`:

- `target_ignicion = 1`: `fuego=1`, sin fuego t-1 en 25 km, sin fuego en la misma celda en los últimos 10 días.
- `target_ignicion = 0`: sin fuego y sin fuego t-1 en 25 km.
- `target_propagacion = 1`: `fuego=1` con fuego t-1 en 25 km.
- `target_propagacion = 0`: sin fuego con fuego t-1 en 25 km.
- Fuera de dominio: `null`.

Conteos sobre las particiones actuales (aún muestreadas con el esquema viejo):

| Partición | target_ign 1 | target_ign 0 | target_ign null | target_prop 1 | target_prop 0 | target_prop null |
|---|---:|---:|---:|---:|---:|---:|
| train | 75.828 | 3.396.852 | 2.963.467 | 519.140 | 2.436.053 | 3.480.954 |
| val | 14.035 | 602.471 | 400.334 | 76.725 | 321.929 | 618.186 |
| test | 28.694 | 1.040.354 | 772.099 | 135.513 | 633.401 | 1.072.233 |

### Remuestreo poblacional D2 (definitivo)

`scripts/05d_muestrear_poblacional_d2.py` construyó un plan desde `dataset_unificado` v2 completo:

- 9.363.090 filas planificadas.
- 143.970 positivos de ignición y 707.220 de propagación.
- Negativos muestreados del dominio D2 completo, no de la muestra vieja.
- Ratio 1:10 exacto por fecha/región.

| Archivo | Positivos | Negativos | Ratio |
|---|---:|---:|---:|
| `train_ignicion.parquet` | 92.604 | 926.040 | 1:10,00 |
| `val_ignicion.parquet` | 17.134 | 171.340 | 1:10,00 |
| `test_ignicion.parquet` | 34.232 | 342.320 | 1:10,00 |
| `train_propagacion.parquet` | 500.849 | 5.008.490 | 1:10,00 |
| `val_propagacion.parquet` | 75.697 | 756.970 | 1:10,00 |
| `test_propagacion.parquet` | 130.674 | 1.306.740 | 1:10,00 |

Los archivos están en `datos/processed/targets_poblacionales/`. Los antiguos `datos/processed/targets/` quedan **deprecados** porque usaban la muestra vieja con buffer same-day.

**Hecho:** `06/07/16/13/18/20/21` usan `targets_poblacionales` y `target_*`; modelos y tablas 2–10 regenerados.

**Alerta actualizada:** `src/alerta.py` usa prevalencias D2 por target y selecciona M3/Propagación según `focos_25km_t1`; ya no usa el máximo entre capas. Se regeneraron:

- alertas 2024-09-09 y 2025-12-31,
- boletines JSON/Markdown,
- GeoTIFF de ambas fechas,
- dashboard interactivo.

**Figuras actualizadas:** fig01 macro-regiones, fig01 arquitectura, fig02 ROC/PR, fig03 mapa operativo, fig04 SHAP, fig05 benchmarking y fig07 ablación. fig06 es conceptual y no cambió.

**Pendiente:** `14` (S3/S5), `tabla00`, manuscrito, suplemento, README y PROTOCOLO.

## Reentrenamiento D2

Modelos reentrenados sobre `targets_poblacionales`:

| Modelo | Chaco ROC-AUC | Oriental ROC-AUC |
|---|---:|---:|
| M3 ignición híbrida | 0,7852 | 0,7480 |
| M2 ignición base | 0,7733 | 0,7323 |
| Propagación táctica | 0,8988 | 0,7798 |

Evaluación poblacional (tabla 7):

- M3 precisión: 0,062 % Chaco / 0,242 % Oriental.
- M2 precisión: 0,047 % / 0,165 %.
- Propagación: 9,853 % / 4,183 %.

Holdout espacial (tabla 8): M1 0,6346/0,6195; M3 0,6246/0,5721.

## Impacto del clima v2 en el gating

- Dataset v2: **915.562.849** celdas-día candidatas (antes 707,1 M con clima v1).
- Reducción por gating: **21,7 %** (antes 39,2 % declarado en el manuscrito).
- Retención de detecciones: **867.354 / 892.533 = 97,17 %** (antes 89,94 %).
- Días sin candidatos: 2018-06-02, 2018-11-23 y 2019-05-23 (todos los nodos con VPD <0,8 kPa o lluvia 48 h ≥10 mm).
- Los números de gating del manuscrito quedan obsoletos.

## Lo que NO se hizo todavía

- No se reentrenó ningún modelo.
- No se remuestreó por target; las particiones actuales conservan el muestreo viejo.
- No se regeneraron tablas/figuras.
- No se modificó el manuscrito ni el material suplementario.
- `preparar_matrices_xy` y los scripts de modelado siguen usando `fuego`; la integración con `target_*` se hará al reentrenar.
- No se ejecutó `03`, `05`, `06`, `16`, `13`, `18`, `19`, `20` ni `21`.

## Criterios de aceptación de Semana 1

- [x] El filtro de baja confianza FIRMS elimina `l` y `low`.
- [x] La precipitación diaria usa suma de incrementos horarios cuando corresponde.
- [x] La falta de continuidad temporal falla explícitamente.
- [x] El manifiesto v2 verifica hashes sin fallos (21 artefactos, 0 fallos).
- [x] La suite de tests pasa (54 tests).
- [ ] D1–D4 aprobadas.
- [ ] Datos de focos y clima regenerados con las correcciones.
- [ ] Manifiesto v2 reemplaza al manifiesto anterior y se integra al CI.

## Próximo gate

**G1:** datos v2 congelados y hasheados. Hasta entonces, no se reentrena ni se actualizan resultados del manuscrito.
