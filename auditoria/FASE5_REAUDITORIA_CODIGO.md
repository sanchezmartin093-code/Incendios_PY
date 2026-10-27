# Fase 5 - Reauditoría de código del candidato r2

**Fecha:** 2026-10-27  
**Candidato vigente:** `submission-candidate-2026-10-27-r2`  
**Commit del tag:** `7390cca`  
**Commit de correcciones:** `e621f95`  
**Estado:** reauditoría aprobada; sin errores de código bloqueantes.

## 1. Correcciones aplicadas

### 1.1 Verificación de checksums

- `scripts/32_verificar_candidato.py` ahora es el verificador de solo lectura.
- `scripts/31_congelar_candidato.py` incluye en su corpus a `scripts/31` y `scripts/32`.
- El CSV de checksums se escribe con `csv.DictWriter` y escaping correcto.
- El script 31 falla si faltan artefactos críticos.
- La instrucción incorrecta de “reejecutar 31 para verificar” quedó corregida en el freeze r2.

### 1.2 Partición de confirmación

Se agregó en `src/holdout.py`:

- `etiquetar_particion_confirmacion(...)`
- `filtrar_particion_confirmacion(...)`

Reglas:

- `desarrollo`: no está en holdout ni en buffer.
- `buffer_confirmacion`: buffer de 20 km; excluido de train y calibración.
- `confirmacion`: holdout; solo evaluación.

Se agregaron tests de:

- exclusión de holdout y buffer;
- evaluación solo en holdout;
- rechazo de solapamiento holdout/buffer.

### 1.3 Protocolo

Se creó `auditoria/FASE5_PROTOCOLO_CONFIRMACION.md` con:

- entrenamiento 2018–2022 sin holdout ni buffer;
- calibración 2023 sin holdout ni buffer;
- evaluación primaria 2024–2025 en bloques de confirmación;
- evaluación única;
- cutoffs congelados antes de mirar confirmación;
- M3/M2 como modelo operativo y M1 como baseline;
- prohibición de reentrenar después de la confirmación.

## 2. Verificación ejecutada

| Control | Resultado |
|---|---|
| `compileall src scripts tests` | OK |
| `pip check` | Sin dependencias rotas |
| `pytest -q tests` | **73 passed, 2 warnings** |
| `00_validar_datos.py` | **0 FAIL** |
| Verificador de checksums | **99/99 archivos OK** |
| Bytes | **9.306.571.614** |
| Corpus SHA-256 | `a8dbd415c97ff8d3433b73af4243137b6033b4ac2676bbe18a5d6984143300c3` |
| Holdout de confirmación | Sin cambios |
| Scripts de selección | No leen test ni confirmación |

## 3. Hallazgos restantes

No quedan errores de código detectados en los caminos ejecutados. Persisten
bloqueantes que no son de código:

1. `fuentes/MANIFIESTO_DATOS_v2.csv` sigue obsoleto; falta manifiesto v3.
2. `CITATION.cff`, `.zenodo.json`, `title_page.qmd` y `cover_letter.qmd` siguen
   con DOI/versión/fecha pre-auditoría.
3. `requirements-lock.txt` no tiene hashes de ruedas; excepción documentada.
4. No se generaron PDFs finales de submission.
5. La Fase 5 todavía no se ejecutó; el holdout permanece intacto.

## 4. Conclusión

El código está listo para iniciar la Fase 5 con el candidato r2. No se reentrenó
ni se evaluó el holdout en esta reauditoría. El siguiente paso es ejecutar el
protocolo de confirmación y no el release.
