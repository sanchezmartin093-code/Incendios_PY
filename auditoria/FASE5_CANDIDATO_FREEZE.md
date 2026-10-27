# Fase 5 - Congelamiento del candidato científico

**Fecha:** 2026-10-27
**Estado:** candidato r3 congelado y verificado.
**Tag:** `submission-candidate-2026-10-27-r3`
**Rama:** `audit/fase0-freeze`

## 1. Identificadores

- Commit científico de Fase 2: `15f3022`
- **Snapshot científico:** `3a5edeee5beb8e31bff8ef69ac15cfd98ea7c3a7`
  - tree: `40b9e64df05a982e509b2d0c16ac3bc38450598a`
  - archive SHA-256: `3e259b2b68ea1a2fd18a0d2ef51fc54c6d98d17e986ecdafe6e974e61c02c891`
- **Tag r3:** `submission-candidate-2026-10-27-r3` → `e2aa676c4c873827a6df35547b10c4d733bf0244`
  - tree: `ed790cdabc8e4ac1530add9c8a385082aad74e45`
  - archive SHA-256: `3af1bb318fae74ed7a8e0da4caeca6712e28097ed38f3db15edf486bb0f6f98c`
- Corpus científico SHA-256: `6c9c8b7ee86ee92b07c498a01d15a33830078144152ace9d56c1ab8adb4c7c6c`
- Archivos congelados: **99**
- Bytes congelados: **9.306.572.469**

El archive del snapshot científico y el archive del tag **no son iguales**:
los checksums se calcularon sobre `3a5edee`; el tag agrega el CSV, el JSON,
el archivo de archive y este documento. Ambos hashes están registrados en
`FASE5_CANDIDATO_CHECKSUMS.json` y en `FASE5_CANDIDATO_GIT_ARCHIVE.sha256`.

Este tag reemplaza a `submission-candidate-2026-10-27-r2` para el trabajo de
Fase 5. El tag r2 (`7390cca`, archive `a768...`) y el tag inicial quedan como
históricos.

## 2. Universo congelado

Se congelaron **99 archivos**, **9.306.572.469 bytes**:

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

Después del tag r3:

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
python scripts/33_verificar_trazabilidad_git.py
```

No reejecutar `scripts/31_congelar_candidato.py` para verificar: ese script
regenera los checksums. El script 31 queda incluido en el corpus, por lo que
cualquier cambio en él invalida este candidato.

`scripts/33_verificar_trazabilidad_git.py` es un verificador externo al
corpus r3: compara commit, tree y archive del snapshot científico y del tag.
