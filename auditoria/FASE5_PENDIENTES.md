# Fase 5 - Pendientes después de la reauditoría

**Fecha:** 2026-10-27
**Candidato vigente:** `submission-candidate-2026-10-27-r4` → `c6322e1`
**Estado:** r4 congelado y verificado; Fase 5 no ejecutada.

## 1. Bloqueantes antes de reentrenar

Sin bloqueos de código. `scripts/32` integra la verificación Git/tag.

## 2. Resuelto en estas correcciones

- Candidato **r4**: source `14d6ad7`, tag `c6322e1`, corpus `d367ebb8...`,
  100 archivos, 9.306.577.967 bytes.
- `scripts/32` verifica checksums y luego ejecuta `scripts/33`.
- `scripts/33` funciona también en el commit del tag, leyendo commit/tree/archive
  del mensaje del tag cuando el JSON solo tiene los campos `source_*`.
- Candidato **r3**: source `3a5edee`, tag `e2aa676`, corpus `6c9c8b7e...`,
  99 archivos, 9.306.572.469 bytes.
- Se separaron `source_commit`, `source_tree`, `source_archive_sha256`,
  `tag_commit`, `tag_tree` y `tag_archive_sha256`.
- El tag r2 conserva el commit `7390cca` y su archive `a768...`.
- Se quitó `host` del JSON público.
- Se agregó el verificador de trazabilidad `scripts/33_verificar_trazabilidad_git.py`.
- `scripts/31` ahora exige M3/M2/propagación/M1 en Chaco y Oriental,
  con sus calibradores, umbrales, prevalencias d2, holdout y targets.
- `scripts/31` recibe el tag por `--tag` y escribe el CSV con LF.

## 3. Bloqueantes científicos

3. **Ejecutar la Fase 5 una sola vez.**
   Reentrenar excluyendo holdout + buffer, recalcular priors, calibrar y fijar cutoffs solo con desarrollo 2023, y evaluar el holdout en 2024–2025.

4. **Reportar confirmación.**
   Crear `FASE5_CONFIRMACION.md`, `FASE5_CONFIRMACION.json` y `tabla13_confirmacion_*`.

5. **Actualizar el manuscrito con confirmación.**
   Si el resultado es negativo, suavizar o cambiar las afirmaciones; no reajustar el modelo.

## 4. Bloqueantes de release

6. **Generar `MANIFIESTO_DATOS_v3.csv`.**
   El v2 contiene dataset monolítico viejo, perfil sintético y targets deprecados.

7. **Actualizar metadatos.**
   `CITATION.cff`, `.zenodo.json`, `title_page.qmd`, `cover_letter.qmd`, README y manuscrito siguen con DOI/versión/fecha pre-auditoría.

8. **Resolver o aceptar formalmente el lock sin hashes.**
   `requirements-lock.txt` tiene versiones exactas y 0 hashes de ruedas.

9. **Generar PDFs finales.**
   Falta `manuscrito.pdf`; `material_suplementario.pdf` es anterior a la reconciliación.

10. **Merge/push y release.**
    `main` está detrás de la rama; no existe `v2.0.0`; no hay DOI nuevo; no se hizo push.

## 5. Mejoras no bloqueantes

11. Excluir buffer también en la calibración si se reutiliza `scripts/27`.
12. Agregar aserciones para el rol `buffer_confirmacion` en tests.
13. Extender CI a manuscrito ES/EN y suplementos.
14. Evaluar si `00_validar_datos.py` debe validar el contenido del manifiesto v3, no solo que existan hashes.

## 6. Regla

La Fase 5 puede iniciar sobre r4. Cualquier cambio de código posterior a r4 requiere un candidato r5.
