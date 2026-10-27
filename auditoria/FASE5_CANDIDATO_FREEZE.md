# Fase 5 - Congelamiento del candidato científico

**Fecha:** 2026-10-27  
**Estado:** candidato r2 congelado y verificado.  
**Tag:** `submission-candidate-2026-10-27-r2`  
**Rama:** `audit/fase0-freeze`

## 1. Identificadores

- Commit científico de Fase 2: `15f3022`
- Commit de correcciones Fase 5: `e621f95`
- Commit que recibirá el tag r2: el commit de este documento y los checksums.
- Git tree del candidato: `2033ad28e452d10edc889d750627827c4fcb8f83`
- Git archive SHA-256: `6988bdbff3e0b6533b83fa758dc3cd8e708036b5f65cd4bcf0610ed54b804de6`
- Corpus científico SHA-256: `a8dbd415c97ff8d3433b73af4243137b6033b4ac2676bbe18a5d6984143300c3`

Este tag reemplaza a `submission-candidate-2026-10-27` para el trabajo de Fase 5.
El tag anterior queda como histórico.

## 2. Universo congelado

Se congelaron **99 archivos**, **9.306.571.614 bytes**:

- dataset anual particionado 2018–2025 y su manifiesto;
- targets `train/val/test` de ignición y propagación;
- grilla, clima, combustible anual, desmonte, focos, FWI, INPE, conteos y plan de muestreo;
- modelos, calibradores, umbrales y prevalencias;
- tablas de selección, comparaciones pareadas y decisión;
- holdout de confirmación;
- código de selección, alerta, modelo, holdout y tests;
- scripts de congelamiento y verificación;
- `requirements-lock.txt`, `pyproject.toml` y workflow de CI.

Archivos:

- `auditoria/FASE5_CANDIDATO_CHECKSUMS.csv`
- `auditoria/FASE5_CANDIDATO_CHECKSUMS.json`
- `auditoria/FASE5_CANDIDATO_GIT_ARCHIVE.sha256`

## 3. Universo NO congelado

No forman parte de este checksum científico:

- manuscrito, suplemento, cover letter y title page;
- README y metadatos de release;
- `CITATION.cff` y `.zenodo.json`;
- DOI, versión y notas de release;
- PDFs/DOCX de submission.

Esos archivos cambiarán después de la confirmación y de la actualización de DOI.

## 4. Regla de congelamiento

Después del tag r2:

1. No se reentrena M1/M2/M3 ni propagación.
2. No se recalculan umbrales ni prevalencias.
3. No se toca el holdout de confirmación.
4. No se cambia el dataset ni los targets.
5. Cualquier cambio científico requiere un nuevo candidato y un nuevo tag.

## 5. Hecho crítico para Fase 5

El holdout de confirmación todavía no se usó, pero **los modelos actuales no son válidos para evaluarlo directamente**:

| Conjunto | Celdas | Presentes en `train_ignicion` |
|---|---:|---:|
| Holdout de confirmación | 68.719 | 61.634 |
| Buffer de confirmación | 95.031 | 85.039 |

Por tanto, la Fase 5 exige:

- excluir holdout + buffer del entrenamiento;
- excluir holdout + buffer de la calibración 2023;
- recalcular priors causales sin esos fuegos;
- reentrenar con hiperparámetros congelados;
- fijar cutoffs solo con desarrollo 2023;
- evaluar una única vez el holdout, incluido el modelo operativo M3/M2.

## 6. Verificación

Verificación de solo lectura:

```bash
python scripts/32_verificar_candidato.py
```

No reejecutar `scripts/31_congelar_candidato.py` para verificar: ese script
regenera los checksums. El script 31 queda incluido en el corpus, por lo que
cualquier cambio en él invalida este candidato.

## 7. Próximo paso

Ejecutar la Fase 5 según `auditoria/FASE5_PROTOCOLO_CONFIRMACION.md`.
No se inicia el release ni la submission hasta cerrar la confirmación.
