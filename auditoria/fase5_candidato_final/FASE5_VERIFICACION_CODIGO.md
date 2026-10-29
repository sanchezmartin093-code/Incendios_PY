> **Nota histórica:** este informe corresponde al candidato r1. La reauditoría vigente está en `auditoria/FASE5_REAUDITORIA_CODIGO.md` (candidato r2).

# Fase 5 - Verificación de código previa al reentrenamiento

**Fecha:** 2026-10-27  
**Candidato:** `submission-candidate-2026-10-27`  
**Commit del tag:** `8a4fc86`  
**Estado:** verificación aprobada; sin errores bloqueantes.

## 1. Controles ejecutados

| Control | Resultado |
|---|---|
| `git status` | Limpio antes de la verificación |
| `compileall` sobre `src/`, `scripts/`, `tests/` | OK |
| `pip check` | Sin dependencias rotas |
| `pytest -q tests` | **71 passed, 2 warnings** |
| `00_validar_datos.py` | **0 FAIL** |
| Checksums del candidato | **97/97 OK** |
| Bytes verificados | **9.306.557.815** |
| Corpus SHA-256 | `a6adff5a60ab19d88168825cf4b2f0455a9e3dffdb1bc6f8ce6136a865df5b7b` |
| Holdout de confirmación | Checksums idénticos; no modificado |
| Scripts de selección | No leen `test_*.parquet` ni el holdout de confirmación |

## 2. Verificación de modelos

- M1: 24 features, 45/60 rondas.
- M2: 35 features, 45/60 rondas.
- M3: 44/45 features, 45/60 rondas.
- Calibradores Platt monótonos presentes.
- Umbrales y prevalencias D2 presentes.
- Las features de los boosters coinciden con el contrato de `src/modelo.py`.

## 3. Hallazgos

### H1 - Instrucción de verificación incorrecta en el documento congelado

`FASE5_CANDIDATO_FREEZE.md` sugería reejecutar `scripts/31_congelar_candidato.py`
para verificar. Ese script **regenera** checksums, no los valida.

**Corrección:** se agregó `scripts/32_verificar_candidato.py`, de solo lectura.
La verificación correcta es:

```bash
python scripts/32_verificar_candidato.py
```

El documento congelado no se modificó para no alterar el tag del candidato.

### H2 - El script de congelamiento no está en su propio corpus

`scripts/31_congelar_candidato.py` no se incluye en la lista de checksums.
No afecta a datos, modelos ni tablas, pero el manifiesto de código del release
debe incluirlo explícitamente.

### H3 - Lock sin hashes

`requirements-lock.txt` tiene versiones exactas pero **0 hashes de ruedas**.
La excepción está aceptada en `FASE2_0_LOCK_EXCEPTION.md`; debe decidirse antes
del release si se reintenta o se publica como limitación.

### H4 - Manifiesto de datos v2 obsoleto

`fuentes/MANIFIESTO_DATOS_v2.csv` todavía contiene el dataset monolítico viejo,
el perfil sintético y targets deprecados. No es un error de código, pero es un
bloqueante de release. Se requiere `MANIFIESTO_DATOS_v3.csv` generado desde los
archivos actuales.

### H5 - Metadatos de release aún en v1.0.0

`CITATION.cff`, `.zenodo.json`, `title_page.qmd` y `cover_letter.qmd` conservan
DOI/versión/fecha pre-auditoría. No afecta el candidato científico, pero impide
publicar o enviar sin actualización.

### H6 - Confirmación requiere reentrenamiento

Los modelos actuales vieron **61.634 de 68.719** celdas del holdout de
confirmación y **85.039** celdas de su buffer. La Fase 5 no puede evaluar los
modelos actuales directamente: debe reentrenar excluyendo holdout + buffer,
recalcular priors y fijar calibración/cutoffs solo con desarrollo 2023.

### H7 - Calibración espacial y buffer

En `scripts/27_validacion_espacial_2023.py`, la calibración usa validación 2023
excluyendo el holdout, pero no excluye explícitamente el buffer. Para Fase 5,
el protocolo debe separar:

- `train_confirmacion`: sin holdout ni buffer;
- `cal_confirmacion`: desarrollo 2023 sin holdout ni buffer;
- `eval_confirmacion`: holdout, una sola vez.

No es un fallo del análisis de selección ya cerrado, pero sí una decisión que
debe quedar fijada antes del reentrenamiento.

## 4. Conclusión

No se encontraron errores de sintaxis, imports, tests ni integridad de datos.
El candidato científico está intacto y verificado.

**No se inicia el reentrenamiento hasta escribir y congelar
`auditoria/FASE5_PROTOCOLO_CONFIRMACION.md` con la separación
train/calibración/evaluación y la regla de una sola evaluación.**
