# Fase 5 - Congelamiento del candidato científico

**Fecha:** 2026-10-27
**Estado:** candidato r4 congelado y verificado.
**Tag:** `submission-candidate-2026-10-27-r4`
**Rama:** `audit/fase0-freeze`

## 1. Identificadores

- Commit científico de Fase 2: `15f3022`
- **Snapshot científico:** `14d6ad7e05d331d3e8f5d28246f414bea9ea472a`
  - tree: `d6d9e9a16efb4c6797a48082ce6e5326cc7e1ec9`
  - archive SHA-256: `53ea437fcfcc420e475514b1f4c7bad7e491315264e44b8b3690fdfec7ea6f1e`
- **Tag r4:** `submission-candidate-2026-10-27-r4` → `c6322e11d17a78e0ca743c34ab238d1c13d3d272`
  - tree: `51b78243dbf370e5ab5a2b22ee05931562fd7e3e`
  - archive SHA-256: `bd5d676d6c202db7e8cb2d841be6633043a62d87afb0f249740d70c5e0b2ad96`
- Corpus científico SHA-256: `d367ebb8b95b6d97529463e713186c97d22b955812dbbef8c6f65893d6fddbc5`
- Archivos congelados: **100**
- Bytes congelados: **9.306.577.967**

El archive del snapshot científico y el archive del tag **no son iguales**:
los checksums se calcularon sobre `14d6ad7`; el tag agrega el CSV, el JSON,
el archivo de archive y este documento. Ambos hashes están registrados en
`FASE5_CANDIDATO_CHECKSUMS.json` y en `FASE5_CANDIDATO_GIT_ARCHIVE.sha256`.

Este tag reemplaza a `submission-candidate-2026-10-27-r3` para el trabajo de
Fase 5. Los tags r3, r2 y el inicial quedan como históricos.

## 2. Universo congelado

Se congelaron **100 archivos**, **9.306.577.967 bytes**:

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

Después del tag r4:

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

Verificación integrada de solo lectura:

```bash
python scripts/32_verificar_candidato.py
```

El script 32 verifica archivos, tamaños, corpus y, al final, invoca la
verificación Git/tag de `scripts/33_verificar_trazabilidad_git.py`.

No reejecutar `scripts/31_congelar_candidato.py` para verificar: ese script
regenera los checksums. El script 31 queda incluido en el corpus, por lo que
cualquier cambio en él invalida este candidato.
