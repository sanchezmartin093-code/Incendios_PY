# Fase 0 - Inventario de release y DOI

**Fecha:** 2026-10-27  
**Rama de congelamiento:** `audit/fase0-freeze`  
**Commit pre-auditoria:** `94d5b5e`  
**Tag pre-auditoria:** `pre-audit-annual-fuel-2026-10-27`  
**Release publica existente:** `v1.0.0`

## 1. Estado actual

La release publica `v1.0.0` y sus DOI **no corresponden** al estado anual actual.

- Codigo DOI: `10.5281/zenodo.23675105`
- Data DOI: `10.5281/zenodo.23675279`
- Concept DOI codigo: `10.5281/zenodo.23675104`
- Concept DOI datos: `10.5281/zenodo.23675278`

El commit actual incluye fuel anual MapBiomas, targets D2 regenerados, modelos reentrenados y tablas actualizadas. Nada de eso esta en `v1.0.0`.

## 2. Que falta para una release nueva

1. Commit de cierre de Fase 0 y tag `fase0-freeze-2026-10-27`.
2. Commit de las correcciones de Fase 1.
3. Rerun completo y tablas/figuras definitivas.
4. Manifiesto de datos con SHA-256 actualizado.
5. Modelos, calibradores, umbrales y prevalencias actuales.
6. Targets D2 actuales.
7. Predicciones y salidas de alerta actuales.
8. Model card y data card.
9. `requirements-lock.txt` usado por CI.
10. Verificacion desde clon limpio.
11. Nuevo tag de release, por ejemplo `v1.1.0`.
12. Nuevo DOI de codigo y datos.
13. Actualizacion de `CITATION.cff`, README y manuscrito.

## 3. Archivos ignorados por git que deben archivarse

- `datos/processed/` (dataset, clima, fuel anual, desmonte, targets, predicciones).
- `datos/raw/` (MapBiomas, Hansen, CDS, FIRMS, vectores).
- `modelos/*.booster`.
- `modelos/calibradores/*.joblib` y `*.json`.

No deben subirse a git; deben incluirse en el archivo de datos o en DVC/almacenamiento externo.

## 4. Afirmaciones que no se pueden hacer con la release actual

- “El codigo y los datos actuales estan archivados con DOI.”
- “El test 2024-2025 es ciego.”
- “La release v1.0.0 reproduce las tablas actuales.”
- “El manifiesto de datos actual corresponde al dataset anual.”

## 5. Criterio de cierre

La Fase 3 no se cierra hasta que:

- exista un tag nuevo;
- el DOI nuevo resuelva;
- el archivo contenga codigo, datos, modelos y manifiesto;
- un clon limpio reproduzca las tablas principales;
- README y manuscrito apunten al DOI correcto.
