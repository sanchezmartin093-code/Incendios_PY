# Fase 2 - Auditoría de identidad de modelos y datos

**Fecha:** 2026-10-27  
**Objetivo:** verificar que la comparación M1/M2/M3 usa contratos y capacidades consistentes.

## 1. Contratos de features

| Modelo | Chaco | Oriental | Booster coincide |
|---|---:|---:|---:|
| M1 biophysical | 24 | 24 | Sí |
| M2 ignición base | 35 | 35 | Sí |
| M3 híbrido | 44 | 45 | Sí |

M3 Oriental incluye `freq_hist_1km` además de las variables compartidas; M3 Chaco no. Esto ya estaba definido en `src/modelo.py` y se mantiene.

## 2. Capacidad y calibración

- M1: 45 rondas Chaco, 60 Oriental; Platt monótono.
- M2: 45 rondas Chaco, 60 Oriental; Platt monótono.
- M3: 45 rondas Chaco, 60 Oriental; Platt monótono.
- Seed base: 42.
- Hojas: 23 Chaco, 31 Oriental.
- No se detectaron modelos con capacidad reducida.

## 3. Umbrales congelados

Los umbrales existen para M1/M2/M3 y fueron calculados en validación:

- M1 Chaco cutoff F0.5 = 0.1387.
- M1 Oriental cutoff F0.5 = 0.1675.
- M2 Chaco cutoff F0.5 = 0.2071.
- M2 Oriental cutoff F0.5 = 0.1983.
- M3 Chaco cutoff F0.5 = 0.2218.
- M3 Oriental cutoff F0.5 = 0.2186.

## 4. Holdout de confirmación

Checksums previos guardados en `auditoria/FASE2_HOLDOUT_CHECKSUM_PRE.txt`:

- `bloques.parquet`: `babe5d38...dd85b`
- `celdas_buffer.parquet`: `22ede473...a8fc`
- `celdas_holdout.parquet`: `0947d7c2...bdc7`

Coinciden con `holdout_confirmacion.meta.json`. No se usará hasta Fase 5.

## 5. Conclusión

La comparación M2/M3 es válida en capacidad y contrato de features. Las diferencias de rendimiento no pueden atribuirse a distinta capacidad. La narrativa antigua con conteos 17/24/33-34 está desactualizada y debe corregirse en el manuscrito.
