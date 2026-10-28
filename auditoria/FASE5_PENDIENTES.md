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

## Pendiente

1. **Decidir el lock sin hashes.**
   `requirements-lock.txt` mantiene versiones exactas y 0 hashes de ruedas; la excepción está documentada.

2. **Verificación final del release.** Concept DOIs resuelven a v2.0.0 (`23717175`/`23717176`).

## Nota

No se reentrena: los boosters, calibradores, targets y tablas del paper no cambiaron. Cualquier cambio científico posterior requiere un candidato nuevo.
