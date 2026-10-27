# Pendientes del paper y del release

**Fecha:** 2026-10-27
**Candidato científico:** `submission-candidate-2026-10-27-r4` → `c6322e1`
**Estado:** r4 congelado y verificado. La ruta de confirmación Fase 5 quedó descartada; el paper se trabaja directo.

## Hecho

- Manifiesto actualizado a `fuentes/MANIFIESTO_DATOS_v3.csv` (69 artefactos, hashes verificados).
- DOI conceptuales, versión 2.0.0 y fechas actualizados en CITATION, Zenodo, title page, cover letter, manuscrito y suplemento.
- Tests actualizados de 71 a 76 en manuscrito, suplemento, cover letter y README.
- PDFs y DOCX finales generados en inglés y español, más cover letter y title page.
- Verificación: 76 tests, 0 FAIL en `00_validar_datos.py`, 100/100 checksums r4.

## Pendiente

1. **Decidir el lock sin hashes.**
   `requirements-lock.txt` mantiene versiones exactas y 0 hashes de ruedas; la excepción está documentada.

2. **Publicar y mergear.**
   Crear el release `v2.0.0`, publicar el código y los datos en Zenodo, asignar los DOI de versión y hacer push/merge a `main`.

3. **Verificación final del release.**
   Comprobar que los DOI conceptuales resuelven a la versión 2.0.0 y que los PDFs/DOCX subidos coinciden con los generados.

## Nota

No se reentrena: los boosters, calibradores, targets y tablas del paper no cambiaron. Cualquier cambio científico posterior requiere un candidato nuevo.
