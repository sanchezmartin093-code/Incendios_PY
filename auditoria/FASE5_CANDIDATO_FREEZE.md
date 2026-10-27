# Fase 5 - Congelamiento del candidato científico

**Fecha:** 2026-10-27  
**Estado:** candidato científico congelado.  
**Tag previsto:** `submission-candidate-2026-10-27`  
**Rama:** `audit/fase0-freeze`

## 1. Identificadores

- Commit científico de Fase 2: `15f3022`
- Commit del script de congelamiento: `b789b5e`
- Commit que recibirá el tag: el commit de este documento y los checksums.
- Git tree del candidato: `1a417b1443a5445a32d0d0eb24f9651dea64c3e5`
- Git archive SHA-256: `10874f3cc47c4d0c1ddd20a2322718e122a423a85554d4bd3525d8340b45ce6a`
- Corpus científico SHA-256: `a6adff5a60ab19d88168825cf4b2f0455a9e3dffdb1bc6f8ce6136a865df5b7b`

## 2. Universo congelado

Se congelaron **97 archivos**, **9.306.557.815 bytes**:

- dataset anual particionado 2018–2025 y su manifiesto;
- targets `train/val/test` de ignición y propagación;
- grilla, clima, combustible anual, desmonte, focos, FWI, INPE, conteos y plan de muestreo;
- modelos, calibradores, umbrales y prevalencias;
- tablas de selección, comparaciones pareadas y decisión;
- holdout de confirmación;
- código de selección, alerta, modelo, holdout y tests;
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

Después del tag:

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
- recalcular priors causales sin esos fuegos;
- reentrenar con hiperparámetros congelados;
- calibrar y fijar cutoffs solo con desarrollo 2023;
- evaluar una única vez el holdout, incluido el modelo operativo M3/M2.

## 6. Verificación

El candidato se generó desde un repositorio limpio. Para verificar integridad:

```bash
git status --short
git rev-parse HEAD
shasum -a 256 auditoria/FASE5_CANDIDATO_CHECKSUMS.csv
```

La verificación completa se hace reejecutando el script desde el tag:

```bash
python scripts/31_congelar_candidato.py
```

## 7. Próximo paso

Escribir y congelar `auditoria/FASE5_PROTOCOLO_CONFIRMACION.md` antes de ejecutar cualquier evaluación sobre el holdout. No se inicia el release ni la submission hasta cerrar la confirmación.
