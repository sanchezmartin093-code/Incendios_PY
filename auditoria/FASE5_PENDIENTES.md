# Pendientes del paper y del release

**Fecha:** 2026-10-27
**Candidato científico:** `submission-candidate-2026-10-27-r4` → `c6322e1`
**Estado:** r4 congelado y verificado. Release v2.0.0 publicado en GitHub y Zenodo. P0 de disponibilidad cerrado.

## Hecho

- Manifiesto actualizado a `fuentes/MANIFIESTO_DATOS_v3.csv` (69 artefactos, hashes verificados).
- DOI conceptuales, versión 2.0.0 y fechas actualizados en CITATION, Zenodo, title page, cover letter, manuscrito y suplemento.
- Tests actualizados de 71 a 76 en manuscrito, suplemento, cover letter y README.
- PDFs y DOCX finales generados en inglés y español, más cover letter y title page.
- Verificación: 76 tests, 0 FAIL en `00_validar_datos.py`, 100/100 checksums r4.
- Release GitHub `v2.0.0`: <https://github.com/sanchezmartin093-code/Incendios_PY/releases/tag/v2.0.0>.
- DOI de versión: código `10.5281/zenodo.23717175`; datos `10.5281/zenodo.23717176`.
- DOI conceptuales: código `10.5281/zenodo.23675104`; datos `10.5281/zenodo.23675278`; ambos resuelven a v2.0.0.
- Asset de datos: `Incendios_PY_data_v2.0.0.zip` (1.185.970.018 bytes; SHA-256 `8e10284d2bffdfcfb6e1727856f11195a67df7533b80733e79cea86d2b840d89`).

## Hecho en P1

- S5, S13, S1–S3 regenerados desde los CSV actuales; figuras suplementarias actualizadas.
- Política operativa **M3/M2 gated** evaluada en test y población; nueva Tabla S17.
- Guía de usuario (`docs/USER_GUIDE.md`) y tutorial (`docs/TUTORIAL.md`) agregados.
- Provenance de Hansen GFC-2025 v1.13 agregada al manuscrito y a Data Availability.
- PDFs con numeración de líneas y doble espacio; highlights en archivo separado (`highlights.docx`).
- Lock sin hashes: excepción aceptada y documentada en `auditoria/FASE2_0_LOCK_EXCEPTION.md`.
- Release verificado: concept DOIs resuelven a v2.0.0 (`23717175`/`23717176`).
- Holdout de confirmación espacial 2023 evaluado con reentrenamiento (Tabla S18).
- Baseline acoplado t-1 implementado como comparador (Tabla S19).
- Sensibilidad D2 completa 3 × 3 (15/25/35 km × 7/10/14 d) con remuestreo y reentrenamiento; Tabla S20. M3 supera a M2 en las 18 comparaciones región × evaluación y el ranking es estable.

## Pendiente científico

1. Validación externa en otro país; el holdout de confirmación espacial 2023 ya se evaluó de forma independiente (Tabla S18).
2. Publicar v2.0.1 con S20, scripts 37–39 y el nuevo asset de datos; v2.0.0 no contiene estos artefactos.

## Nota

La sensibilidad D2 reentrena modelos auxiliares y no modifica los boosters operativos de r4; cualquier cambio en el modelo principal requiere un candidato nuevo.
